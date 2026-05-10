"""
AgentOS-Kernel — Full Integration Orchestrator.

Wires the three core components into a unified agent execution loop:

  ┌──────────────────────────────────────────────────────────────────┐
  │                         AgentOS                                 │
  │                                                                 │
  │  step(user_message)                                             │
  │    │                                                            │
  │    ├─ 1. Append user_message to L1                              │
  │    ├─ 2. Bounds check: L1→L2 compaction, L2→L3 archival         │
  │    ├─ 3. L3 gated retrieval (BGE embed + cross-encoder)         │
  │    ├─ 4. Assemble mega-prompt (system + L2 + L1 + L3)           │
  │    ├─ 5. vLLM inference (Qwen2.5-72B-AWQ, JSON-constrained)    │
  │    ├─ 6. Rust ToolRuntime.execute_batch()                       │
  │    └─ 7. Append tool result to L1 as observation                │
  │                                                                 │
  │  Models:                                                        │
  │    • Qwen2.5-72B-Instruct-AWQ via vLLM  (~38 GB VRAM)          │
  │    • Qwen2.5-1.5B-Instruct via transformers (~3 GB VRAM)        │
  │    • BAAI/bge-base-en-v1.5 for embeddings (~0.4 GB)             │
  │    • BAAI/bge-reranker-v2-m3 for cross-encoder (~1.1 GB)        │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memory_manager import MemoryManager
from l3_index import L3Index, RetrievedFact

log = logging.getLogger("agentos.core")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Call JSON Schema (for vLLM guided decoding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning about what to do next.",
        },
        "tool": {
            "type": "string",
            "enum": [
                "query_transactions",
                "trace_network",
                "check_watchlist",
                "request_wire_trace",
                "check_device_overlap",
                "assess_risk",
                "file_sar",
            ],
            "description": "The tool to execute.",
        },
        "parameters": {
            "type": "object",
            "description": "Parameters to pass to the tool.",
        },
    },
    "required": ["reasoning", "tool", "parameters"],
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "response": {
            "type": "string",
            "description": "Final answer to the user when no more tools are needed.",
        },
    },
    "required": ["reasoning", "response"],
}

REASONING_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StepResult:
    """Result of a single AgentOS.step() execution."""

    turn_number: int
    user_message: str
    llm_response: Dict[str, Any]
    tool_results: Optional[Dict[str, Any]]
    l1_compacted: bool
    l2_archived_to_l3: bool
    l3_facts_retrieved: int
    prompt_token_count: int
    elapsed_ms: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM Inference Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ReasoningEngine:
    """Lazy-loaded vLLM engine for Qwen2.5-72B-AWQ inference.

    Uses guided JSON decoding to guarantee parseable tool calls.
    """

    _instance: Optional["ReasoningEngine"] = None
    _llm = None
    _tokenizer = None

    @classmethod
    def get(cls) -> "ReasoningEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if ReasoningEngine._llm is None:
            log.info("Loading reasoning model: %s", REASONING_MODEL)
            from vllm import LLM

            ReasoningEngine._llm = LLM(
                model=REASONING_MODEL,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                max_model_len=8192,
                dtype="auto",
                quantization="awq",
                trust_remote_code=True,
            )
            ReasoningEngine._tokenizer = ReasoningEngine._llm.get_tokenizer()
            log.info("Reasoning model loaded")

    def generate(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any] = TOOL_CALL_SCHEMA,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Generate a JSON-constrained response from the reasoning model.

        Args:
            messages: Chat messages (system, user, assistant, tool).
            schema: JSON schema to enforce via guided decoding.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict from the model's response.
        """
        from vllm import SamplingParams

        # Format messages into a prompt using the tokenizer's chat template
        prompt = ReasoningEngine._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=json.dumps(schema),
        )

        outputs = ReasoningEngine._llm.generate([prompt], params)
        response_text = outputs[0].outputs[0].text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            log.error("vLLM JSON decode failed: %s", response_text[:200])
            return {
                "reasoning": "Failed to generate valid JSON",
                "tool": "query_transactions",
                "parameters": {},
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AgentOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = (
    "You are an AML (Anti-Money Laundering) investigation agent. "
    "You have access to tools for tracing networks, checking watchlists, "
    "querying transactions, assessing risk, and filing SARs. "
    "Analyze each alert systematically using available tools. "
    "Always explain your reasoning before taking action.\n\n"
    "Available tools:\n"
    "- query_transactions: Query transaction history {customer_id}\n"
    "- trace_network: Trace entity relationship network {entity_id, depth}\n"
    "- check_watchlist: Screen entity against PEP/sanctions lists {entity_name}\n"
    "- request_wire_trace: Get SWIFT wire trace details {entity_id|transaction_id}\n"
    "- check_device_overlap: Check shared devices/IPs {entity_id}\n"
    "- assess_risk: Generate risk assessment {customer_id}\n"
    "- file_sar: File a Suspicious Activity Report {case_id, narrative}\n\n"
    "Respond with a JSON object containing 'reasoning', 'tool', and 'parameters'."
)


class AgentOS:
    """Unified orchestrator wiring ToolRuntime + MemoryManager + L3Index.

    Args:
        use_mock_llm: If True, use keyword-based mock LLM instead of vLLM.
        use_mock_compactor: If True, use heuristic compaction instead of Qwen 1.5B.
        use_mock_embeddings: If True, use deterministic hash embeddings.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        max_l1_tokens: int = 6000,
        max_l2_tokens: int = 2000,
        max_l2_entities: int = 20,
        compaction_batch_size: int = 4,
        l3_db_path: str = "/tmp/agent_os_l3",
        l3_relevance_threshold: float = 0.50,
        tool_timeout_ms: int = 10000,
        rust_worker_threads: Optional[int] = None,
        use_mock_llm: bool = False,
        use_mock_compactor: bool = False,
        use_mock_embeddings: bool = False,
    ) -> None:
        self._use_mock_llm = use_mock_llm

        # ── Component 1: Rust Async Tool Runtime ──
        try:
            from agent_os_core import ToolRuntime
            self._tool_runtime = ToolRuntime(rust_worker_threads)
            self._rust_available = True
        except ImportError:
            self._tool_runtime = None
            self._rust_available = False

        # ── Component 2: L1/L2 Memory Manager ──
        self._memory = MemoryManager(
            system_prompt=system_prompt,
            max_l1_tokens=max_l1_tokens,
            max_l2_tokens=max_l2_tokens,
            compaction_batch_size=compaction_batch_size,
            use_mock_compactor=use_mock_compactor,
        )
        self._max_l2_entities = max_l2_entities

        # ── Component 3: L3 Persistent Index ──
        self._l3 = L3Index(
            db_path=l3_db_path,
            relevance_threshold=l3_relevance_threshold,
            use_mock=use_mock_embeddings,
        )

        # ── Orchestrator state ──
        self._tool_timeout_ms = tool_timeout_ms
        self._turn_number = 0
        self._total_tool_calls = 0
        self._total_l3_retrievals = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core Agent Loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def step(self, user_message: str) -> StepResult:
        """Execute one full turn of the agent loop.

        1. Append user message to L1
        2. Bounds check: L1→L2 compaction, L2→L3 archival
        3. L3 gated retrieval
        4. Assemble mega-prompt
        5. LLM inference (vLLM or mock)
        6. Execute tools via Rust runtime
        7. Append observations to L1
        """
        step_start = time.perf_counter()
        self._turn_number += 1
        l1_compacted = False
        l2_archived = False

        # ── Step 1: Append user message to L1 ──
        self._memory.add_turn("user", user_message, turn=self._turn_number)

        # ── Step 2: Bounds checking ──
        while self._memory.needs_compaction:
            result = await self._memory.compact_if_needed()
            if result:
                l1_compacted = True
            else:
                break

        if self._memory.l2_token_count > self._memory.max_l2_tokens:
            l2_archived = self._archive_l2_overflow()

        # ── Step 3: L3 gated retrieval ──
        l3_facts = self._l3.retrieve_gated(user_message)
        self._total_l3_retrievals += len(l3_facts)

        # ── Step 4: Assemble mega-prompt ──
        messages = self._memory.get_injected_prompt()
        if l3_facts:
            messages = L3Index.inject_into_prompt(messages, l3_facts)

        # ── Step 5: LLM call ──
        if self._use_mock_llm:
            llm_response = self._mock_llm_call(messages)
        else:
            llm_response = ReasoningEngine.get().generate(messages)

        # ── Step 6: Append LLM response to L1 ──
        self._memory.add_turn(
            "assistant",
            json.dumps(llm_response),
            turn=self._turn_number,
        )

        # ── Step 7: Execute tools & append observations ──
        tool_batch_result = None
        if "tool" in llm_response:
            tool_batch_result = self._execute_tools(llm_response)

            self._memory.add_turn(
                "tool",
                json.dumps(tool_batch_result),
                turn=self._turn_number,
            )

            while self._memory.needs_compaction:
                result = await self._memory.compact_if_needed()
                if result:
                    l1_compacted = True
                else:
                    break

        elapsed = int((time.perf_counter() - step_start) * 1000)

        return StepResult(
            turn_number=self._turn_number,
            user_message=user_message,
            llm_response=llm_response,
            tool_results=tool_batch_result,
            l1_compacted=l1_compacted,
            l2_archived_to_l3=l2_archived,
            l3_facts_retrieved=len(l3_facts),
            prompt_token_count=self._memory.total_prompt_tokens,
            elapsed_ms=elapsed,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # L2 → L3 Archival Bridge
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _archive_l2_overflow(self) -> bool:
        """Archive the oldest L2 entities to L3 when L2 exceeds budget."""
        scratchpad = self._memory._l2_scratchpad
        if not scratchpad:
            return False

        archived_any = False
        entities_by_size = sorted(
            scratchpad.keys(),
            key=lambda e: len(scratchpad[e]),
            reverse=True,
        )

        for entity_id in entities_by_size:
            if self._memory.l2_token_count <= self._memory.max_l2_tokens:
                break
            facts = scratchpad.pop(entity_id)
            self._l3.archive_facts(entity_id, facts)
            archived_any = True

        if archived_any:
            self._memory._l2_token_count = self._memory._count_tokens(
                self._memory._format_l2_preamble()
            )

        return archived_any

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Mock LLM (testing only)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _mock_llm_call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Keyword-based mock LLM. For testing without GPU."""
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"].lower()
                break

        if any(kw in user_msg for kw in ["investigate", "alert", "review"]):
            return {
                "reasoning": "Starting investigation — reviewing alert details.",
                "tool": "query_transactions",
                "parameters": {"customer_id": "CUST-A"},
            }
        elif any(kw in user_msg for kw in ["owner", "beneficial", "who owns", "korev"]):
            return {
                "reasoning": "Tracing beneficial ownership chain.",
                "tool": "trace_network",
                "parameters": {"entity_id": "CUST-A", "depth": 2},
            }
        elif any(kw in user_msg for kw in ["risk", "assessment", "score"]):
            return {
                "reasoning": "Compiling risk assessment from gathered evidence.",
                "tool": "assess_risk",
                "parameters": {"customer_id": "CUST-A"},
            }
        elif any(kw in user_msg for kw in ["watchlist", "pep", "sanction", "screen"]):
            return {
                "reasoning": "Screening against global watchlists.",
                "tool": "check_watchlist",
                "parameters": {"entity_name": "Viktor Korev"},
            }
        elif any(kw in user_msg for kw in ["device", "ip", "mule"]):
            return {
                "reasoning": "Checking device/IP overlap with known mules.",
                "tool": "check_device_overlap",
                "parameters": {"entity_id": "CUST-A"},
            }
        elif any(kw in user_msg for kw in ["file", "sar", "report"]):
            return {
                "reasoning": "Evidence warrants filing a SAR.",
                "response": "SAR filing recommended based on structuring, PEP, and shell network.",
            }
        else:
            return {
                "reasoning": f"Processing: {user_msg[:100]}",
                "tool": "query_transactions",
                "parameters": {"customer_id": "CUST-A"},
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tool Execution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _execute_tools(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls via the Rust ToolRuntime."""
        tool_name = llm_response.get("tool", "")
        params = llm_response.get("parameters", {})
        params_json = json.dumps(params)

        self._total_tool_calls += 1

        if self._rust_available and self._tool_runtime is not None:
            result_json = self._tool_runtime.execute_one(
                tool_name, params_json, self._tool_timeout_ms
            )
            return json.loads(result_json)
        else:
            return {
                "tool": tool_name,
                "status": "success",
                "data": {"mock": True, "params": params},
                "elapsed_ms": 0,
                "error": None,
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Diagnostics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_full_state(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of the entire OS state."""
        mem_stats = self._memory.get_stats()
        l3_stats = self._l3.get_stats()
        return {
            "turn_number": self._turn_number,
            "total_tool_calls": self._total_tool_calls,
            "total_l3_retrievals": self._total_l3_retrievals,
            "rust_runtime_available": self._rust_available,
            "llm_mode": "mock" if self._use_mock_llm else REASONING_MODEL,
            "memory": mem_stats,
            "l3_index": l3_stats,
        }

    def get_prompt_preview(self, max_chars: int = 500) -> str:
        """Get a preview of the current assembled prompt."""
        messages = self._memory.get_injected_prompt()
        lines = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:max_chars]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def cleanup(self) -> None:
        """Clean up L3 database. For testing."""
        self._l3.drop()

    def __repr__(self) -> str:
        state = self.get_full_state()
        llm = "mock" if self._use_mock_llm else "qwen-72b"
        return (
            f"AgentOS(turns={state['turn_number']}, "
            f"tools={state['total_tool_calls']}, "
            f"rust={'✓' if state['rust_runtime_available'] else '✗'}, "
            f"llm={llm}, "
            f"L1={state['memory']['l1_tokens']}/{state['memory']['l1_budget']}tok, "
            f"L2={state['memory']['l2_entities']}ent, "
            f"L3={state['l3_index']['rows_in_table']}rows)"
        )
