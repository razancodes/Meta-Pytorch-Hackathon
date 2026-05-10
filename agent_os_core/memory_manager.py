"""
AgentOS-Kernel — L1/L2 Cognitive Cache Compaction Engine.

Implements a two-tier memory hierarchy for long-context agentic reasoning:

  L1 (Active Working Memory)
    ─ Sliding window of raw conversation turns
    ─ Capped by token budget (default 6,000 tokens)
    ─ Oldest turns evicted to compaction when budget exceeded

  L2 (Structured Scratchpad)
    ─ Entity → [facts] key-value store
    ─ Populated by the compaction model (Qwen2.5-1.5B-Instruct)
    ─ Injected as a structured preamble in every prompt

The compaction model can be set to:
  - Real: Qwen2.5-1.5B-Instruct via transformers (requires GPU)
  - Mock: Heuristic extraction with simulated latency (for testing)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import tiktoken

log = logging.getLogger("agentos.memory")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Role = Literal["system", "user", "assistant", "tool"]

COMPACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Message:
    """Single conversation turn."""

    role: Role
    content: str
    token_count: int = 0  # Populated by MemoryManager
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class CompactionResult:
    """Output of the compaction model — entities and their facts."""

    entities: Dict[str, List[str]]  # entity_name → [fact_1, fact_2, ...]
    source_turn_range: Tuple[int, int]  # (first_turn_idx, last_turn_idx)
    timestamp: float = field(default_factory=time.time)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compaction Engine (Real & Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


COMPACTION_PROMPT_TEMPLATE = """You are a precise fact extraction engine. Given a sequence of conversation turns from an AML investigation, extract all factual information as structured JSON.

Rules:
1. Group facts by entity (customer IDs, entity names, transaction IDs, alert IDs).
2. Each fact should be a single, self-contained statement.
3. Remove duplicates. Keep the most specific version of each fact.
4. Use the entity's canonical ID as the key (e.g., "CUST-A", "Viktor Korev", "TXN-001").
5. For general task context, use "__task__" as the entity.

Output ONLY valid JSON in this exact format:
{{"entities": {{"ENTITY_ID": ["fact 1", "fact 2"], "ENTITY_ID_2": ["fact 3"]}}}}

Conversation turns to extract from:
{turns}

JSON output:"""


class CompactorEngine:
    """Lazy-loaded Qwen2.5-1.5B-Instruct for L1→L2 compaction."""

    _instance: Optional["CompactorEngine"] = None
    _model = None
    _tokenizer = None

    @classmethod
    def get(cls) -> "CompactorEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if CompactorEngine._model is None:
            log.info("Loading compaction model: %s", COMPACTOR_MODEL)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            CompactorEngine._tokenizer = AutoTokenizer.from_pretrained(
                COMPACTOR_MODEL, trust_remote_code=True
            )
            CompactorEngine._model = AutoModelForCausalLM.from_pretrained(
                COMPACTOR_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            log.info("Compaction model loaded on device: %s", CompactorEngine._model.device)

    def extract_entities(self, turns_text: str) -> Dict[str, List[str]]:
        """Run the compaction model to extract entities and facts.

        Args:
            turns_text: Formatted string of conversation turns.

        Returns:
            Dict mapping entity IDs to lists of facts.
        """
        import torch

        prompt = COMPACTION_PROMPT_TEMPLATE.format(turns=turns_text)

        inputs = CompactorEngine._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(CompactorEngine._model.device)

        with torch.no_grad():
            outputs = CompactorEngine._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=CompactorEngine._tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = CompactorEngine._tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON from response
        try:
            # Handle cases where model wraps in markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response)
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed["entities"]
            elif isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, IndexError, KeyError):
            log.warning("Compaction model returned unparseable JSON: %s", response[:200])

        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MemoryManager:
    """Two-tier cognitive cache for long-context agentic reasoning.

    Args:
        system_prompt: Base system prompt (always pinned in L1).
        max_l1_tokens: Token budget for L1 sliding window.
        max_l2_tokens: Token budget for L2 scratchpad.
        compaction_batch_size: Turns to evict per compaction cycle.
        use_mock_compactor: If True, use heuristic extraction. If False, use Qwen2.5-1.5B.
    """

    def __init__(
        self,
        system_prompt: str,
        max_l1_tokens: int = 6000,
        max_l2_tokens: int = 4000,
        compaction_batch_size: int = 4,
        encoding_name: str = "cl100k_base",
        use_mock_compactor: bool = False,
    ) -> None:
        self._encoder = tiktoken.get_encoding(encoding_name)

        # ── System prompt (pinned, never evicted) ──
        self._system_prompt = system_prompt
        self._system_token_count = self._count_tokens(system_prompt)

        # ── L1: Sliding Window (raw conversation turns) ──
        self._l1_messages: List[Message] = []
        self._l1_token_count: int = 0
        self.max_l1_tokens = max_l1_tokens
        self._compaction_batch_size = compaction_batch_size

        # ── L2: Structured Scratchpad (entity → facts) ──
        self._l2_scratchpad: Dict[str, List[str]] = {}
        self._l2_token_count: int = 0
        self.max_l2_tokens = max_l2_tokens

        # ── Compaction config ──
        self._use_mock_compactor = use_mock_compactor
        self._compaction_count: int = 0
        self._total_turns_compacted: int = 0
        self._compaction_running: bool = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Token Counting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoder.encode(text))

    @property
    def l1_token_count(self) -> int:
        """Current L1 token usage (excluding system prompt)."""
        return self._l1_token_count

    @property
    def l1_total_tokens(self) -> int:
        """Total L1 tokens including system prompt."""
        return self._system_token_count + self._l1_token_count

    @property
    def l2_token_count(self) -> int:
        """Current L2 scratchpad token usage."""
        return self._l2_token_count

    @property
    def total_prompt_tokens(self) -> int:
        """Total tokens that get_injected_prompt() will produce."""
        return self._system_token_count + self._l2_token_count + self._l1_token_count

    @property
    def l1_turn_count(self) -> int:
        """Number of conversation turns in L1."""
        return len(self._l1_messages)

    @property
    def l2_entity_count(self) -> int:
        """Number of entities tracked in L2."""
        return len(self._l2_scratchpad)

    @property
    def l2_fact_count(self) -> int:
        """Total number of facts across all L2 entities."""
        return sum(len(facts) for facts in self._l2_scratchpad.values())

    @property
    def needs_compaction(self) -> bool:
        """Whether L1 has exceeded its token budget."""
        return self._l1_token_count > self.max_l1_tokens

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # L1: Add / Evict
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_turn(self, role: Role, content: str, **metadata: Any) -> Message:
        """Add a conversation turn to L1.

        Does NOT automatically trigger compaction — call compact_if_needed()
        after adding turns to control when compaction runs.
        """
        token_count = self._count_tokens(content)
        msg = Message(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata,
        )
        self._l1_messages.append(msg)
        self._l1_token_count += token_count
        return msg

    def _evict_oldest(self, n: int) -> List[Message]:
        """Remove the oldest N non-system turns from L1."""
        n = min(n, len(self._l1_messages))
        evicted = self._l1_messages[:n]
        self._l1_messages = self._l1_messages[n:]

        evicted_tokens = sum(m.token_count for m in evicted)
        self._l1_token_count -= evicted_tokens

        return evicted

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # L2: Compaction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def compact_if_needed(self) -> Optional[CompactionResult]:
        """Check L1 budget and compact to L2 if necessary."""
        if not self.needs_compaction:
            return None
        if self._compaction_running:
            return None

        evicted = self._evict_oldest(self._compaction_batch_size)
        if not evicted:
            return None

        self._compaction_running = True
        try:
            result = await self._compact_to_l2(evicted)
            self._merge_into_l2(result)
            self._compaction_count += 1
            self._total_turns_compacted += len(evicted)
            return result
        finally:
            self._compaction_running = False

    async def _compact_to_l2(self, evicted_messages: List[Message]) -> CompactionResult:
        """Send evicted turns to the compaction model.

        Uses CompactorEngine (Qwen2.5-1.5B) in production mode,
        or heuristic extraction in mock mode.
        """
        if self._use_mock_compactor:
            return await self._mock_compact(evicted_messages)

        # ── Real compaction via Qwen2.5-1.5B-Instruct ──
        turns_text = self._format_turns_for_compaction(evicted_messages)

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None,
            CompactorEngine.get().extract_entities,
            turns_text,
        )

        # Fallback if model returned nothing
        if not entities:
            log.warning("Compaction model returned empty — using heuristic fallback")
            return await self._mock_compact(evicted_messages)

        return CompactionResult(
            entities=entities,
            source_turn_range=(0, len(evicted_messages) - 1),
        )

    @staticmethod
    def _format_turns_for_compaction(messages: List[Message]) -> str:
        """Format evicted messages as a readable conversation for the compactor."""
        lines = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content[:500]  # Truncate long tool results
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _mock_compact(self, evicted_messages: List[Message]) -> CompactionResult:
        """Heuristic entity/fact extraction. For testing without GPU."""
        # No artificial sleep — immediate extraction
        entities: Dict[str, List[str]] = {}

        for msg in evicted_messages:
            content = msg.content

            if msg.role == "tool":
                try:
                    data = json.loads(content)
                    self._extract_facts_from_tool_result(data, entities)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif msg.role == "assistant":
                for token in content.split():
                    cleaned = token.strip("\"',{}[]():;")
                    if cleaned.startswith(("ENT_", "CUST-", "TXN-", "ALT-")):
                        entities.setdefault(cleaned, []).append(
                            f"Referenced in assistant response at step {msg.metadata.get('step', '?')}"
                        )
            elif msg.role == "user":
                if len(content) > 20:
                    entities.setdefault("__task__", []).append(
                        f"User instruction: {content[:200]}"
                    )

        if not entities:
            combined = " | ".join(m.content[:100] for m in evicted_messages)
            entities["__context__"] = [f"Compacted context: {combined[:300]}"]

        return CompactionResult(
            entities=entities,
            source_turn_range=(0, len(evicted_messages) - 1),
        )

    @staticmethod
    def _extract_facts_from_tool_result(
        data: Any, entities: Dict[str, List[str]], prefix: str = ""
    ) -> None:
        """Recursively extract entity/fact pairs from tool result JSON."""
        if isinstance(data, dict):
            entity_keys = (
                "entity_id", "customer_id", "entity", "entity_name",
                "trace_target", "target",
            )
            entity_id = None
            for key in entity_keys:
                if key in data and isinstance(data[key], str):
                    entity_id = data[key]
                    break

            if entity_id:
                facts = []
                for k, v in data.items():
                    if k in entity_keys or k == "source":
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        facts.append(f"{k}: {v}")
                    elif isinstance(v, list) and len(v) <= 5:
                        facts.append(f"{k}: {json.dumps(v)}")
                if facts:
                    entities.setdefault(entity_id, []).extend(facts)
            else:
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        MemoryManager._extract_facts_from_tool_result(
                            v, entities, prefix=f"{prefix}{k}."
                        )
        elif isinstance(data, list):
            for item in data[:5]:
                MemoryManager._extract_facts_from_tool_result(
                    item, entities, prefix=prefix
                )

    def _merge_into_l2(self, result: CompactionResult) -> None:
        """Merge compaction results into the L2 scratchpad, deduplicating facts."""
        for entity, new_facts in result.entities.items():
            existing = self._l2_scratchpad.get(entity, [])
            existing_set = set(existing)

            for fact in new_facts:
                if fact not in existing_set:
                    existing.append(fact)
                    existing_set.add(fact)

            self._l2_scratchpad[entity] = existing

        # Recompute L2 token count
        self._l2_token_count = self._count_tokens(self._format_l2_preamble())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prompt Assembly
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _format_l2_preamble(self) -> str:
        """Format the L2 scratchpad as a structured text preamble."""
        if not self._l2_scratchpad:
            return ""

        lines = ["[ACCUMULATED KNOWLEDGE — Structured Scratchpad]"]
        for entity, facts in sorted(self._l2_scratchpad.items()):
            if entity.startswith("__"):
                label = entity.strip("_").replace("_", " ").title()
                lines.append(f"\n## {label}")
            else:
                lines.append(f"\n## Entity: {entity}")
            for fact in facts:
                lines.append(f"  • {fact}")
        lines.append("\n[END SCRATCHPAD]\n")

        return "\n".join(lines)

    def get_injected_prompt(self) -> List[Dict[str, str]]:
        """Assemble the full prompt for the reasoning model.

        Returns a message list:
          [0] system  — base prompt + L2 scratchpad preamble
          [1..N] user/assistant/tool — L1 sliding window
        """
        l2_preamble = self._format_l2_preamble()
        if l2_preamble:
            system_content = f"{self._system_prompt}\n\n{l2_preamble}"
        else:
            system_content = self._system_prompt

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        for msg in self._l1_messages:
            messages.append(msg.to_dict())

        return messages

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Diagnostics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_stats(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the memory manager state."""
        return {
            "l1_turns": self.l1_turn_count,
            "l1_tokens": self._l1_token_count,
            "l1_total_tokens": self.l1_total_tokens,
            "l1_budget": self.max_l1_tokens,
            "l1_utilization": f"{self._l1_token_count / self.max_l1_tokens * 100:.1f}%",
            "l2_entities": self.l2_entity_count,
            "l2_facts": self.l2_fact_count,
            "l2_tokens": self._l2_token_count,
            "l2_budget": self.max_l2_tokens,
            "system_prompt_tokens": self._system_token_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "compactions_run": self._compaction_count,
            "turns_compacted": self._total_turns_compacted,
            "needs_compaction": self.needs_compaction,
            "compactor": "mock" if self._use_mock_compactor else COMPACTOR_MODEL,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        mode = "mock" if self._use_mock_compactor else "qwen-1.5b"
        return (
            f"MemoryManager("
            f"L1={stats['l1_tokens']}/{stats['l1_budget']}tok "
            f"[{stats['l1_turns']} turns], "
            f"L2={stats['l2_tokens']}/{stats['l2_budget']}tok "
            f"[{stats['l2_entities']} entities, {stats['l2_facts']} facts], "
            f"compactor={mode})"
        )
