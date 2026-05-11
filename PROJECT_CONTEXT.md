# Memex OS-Agent Benchmark — Project Context

> Complete technical reference for the Memex AML Investigation Environment.
> **Hackathon:** Meta / Hugging Face OpenEnv · **Team:** MuazTPM
> **Unified Model Family:** Qwen 2.5 (7B training · 1.5B compaction · 72B inference)

---

## 1. Problem Statement

Money laundering costs **$800B–$2T annually** (2–5% of global GDP), with ~90% going undetected. Existing AML systems cost **$274B/year** globally. A real investigation is a **multi-step, partially observable** process: an analyst pulls profiles, traces networks, waits for wire results, cross-references sanctions lists, then decides whether to file a SAR.

This process has three properties that make it a compelling RL environment:

| Property | Why It Matters |
|----------|---------------|
| **Partial Observability** | The agent never sees all data at once — evidence gets evicted from attention |
| **Long Horizon** | 10–25 tool calls across multiple data sources |
| **Asymmetric Consequences** | FN = −2.0 (missed laundering), FP = −0.75 (false SAR), TP = +1.0, TN = +0.5 |

Memex is an OpenEnv-compatible POMDP environment where an LLM must **operate** an OS — managing Virtual Memory, handling Interrupts, and self-updating its Kernel — while solving these investigations.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MEMEX SYSTEM OVERVIEW                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  OpenEnv SDK  │◄──►│  FastAPI Server   │◄──►│  AMLEnvironment   │  │
│  │  (openenv.yaml)│   │ (openenv_server)  │    │ (18-tool dispatch) │  │
│  └──────────────┘    └──────────────────┘    └────────┬──────────┘  │
│                                                        │              │
│                              ┌─────────────────────────┼──────┐      │
│                              │                         ▼      │      │
│                        ┌─────┴──────┐          ┌──────────────┐      │
│                        │   Grader    │          │ StateManager │      │
│                        │ (rewards)   │          │  (OS kernel) │      │
│                        └────────────┘          └──────────────┘      │
│                                                        │              │
│                     ┌──────────────┬───────────────┬────┘              │
│                     ▼              ▼               ▼                  │
│               ┌──────────┐  ┌──────────┐   ┌────────────┐            │
│               │ Virtual  │  │  Async   │   │  Kernel    │            │
│               │ Memory   │  │Interrupts│   │  Updates   │            │
│               │ (RAM/Disk)│  │(Wire Trace)│  │(Sys Prompt)│            │
│               └──────────┘  └──────────┘   └────────────┘            │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  TRAINING LAYER                                                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ train_grpo │  │  self_play  │  │ eval     │  │  curriculum   │  │
│  │ (TRL+Unsloth)│ │(L vs D PPO) │  │ harness  │  │  (PLR engine) │  │
│  └────────────┘  └─────────────┘  └──────────┘  └───────────────┘  │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  PRODUCTION LAYER (agent_os_core/)                                   │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  AgentOS  │  │ MemoryManager │  │ L3Index  │  │ ToolRuntime  │  │
│  │(orchestr.)│  │  (L1/L2 cache)│  │(LanceDB) │  │ (Rust/Tokio) │  │
│  └──────────┘  └───────────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. OS Mechanics — The Three Pillars

### 3.1 Virtual Memory (RAM → Disk Paging)

The agent's context window holds only the **last 2 observations** (RAM = 2 slots). Evidence gets evicted as new observations arrive. The agent can persist critical findings via `write_to_case_file` (capped at 3 rewarded writes/episode).

| Concept | Implementation | File |
|---------|---------------|------|
| RAM capacity | 2-slot sliding window | `state_manager.py` |
| Disk persistence | Unlimited reads, 3 rewarded writes | `state_manager.py` |
| Page fault | −0.05 when referencing evicted, unsaved data | `graders/grader.py` |
| Page hit | +0.10 when reading previously saved data | `graders/grader.py` |

### 3.2 Async Interrupts (Wire Traces)

`request_wire_trace` starts a background job with a 2–4 step ETA. The agent receives a `job_id` and must continue investigating. Calling `retrieve_async_result` before ETA triggers an async timeout (−0.10).

| Concept | Implementation |
|---------|---------------|
| Job scheduling | `state_manager.py:_start_async_job()` |
| ETA tracking | Step counter, random 2–4 step delay |
| Premature retrieval | −0.10 penalty |
| Successful retrieval | Wire trace data injected into observation |

### 3.3 Kernel Updates (Self-Modifying System Prompt)

The agent searches `compliance_manual` for AML regulations, then calls `update_system_prompt` to inject a compliance rule into its own active directives. Only **6 valid kernel modes** exist (defined in `state_manager.py:KERNEL_MODES`), preventing prompt injection.

---

## 4. Tool Roster (18 Tools)

| Category | Tools | Count |
|----------|-------|:-----:|
| **Domain Investigation** | `review_alert`, `get_customer_profile`, `query_transactions`, `check_watchlist`, `trace_network`, `check_source_of_funds`, `check_market_price`, `assess_risk`, `check_device_overlap`, `verify_customs_invoice`, `query_beneficial_ownership` | 11 |
| **OS Mechanic** | `write_to_case_file`, `request_wire_trace`, `retrieve_async_result`, `search_compliance_manual`, `update_system_prompt` | 5 |
| **Terminal** | `file_sar`, `close_alert` | 2 |

### AML Typologies

| ID | Typology | Description |
|----|----------|-------------|
| `easy` | Structuring | Sub-$10K cash deposits (smurfing) |
| `medium` | Layering | Fan-out through shell companies |
| `hard` | Trade-Based ML | Over-invoicing phantom shipments |

---

## 5. Reward Design

### 5.1 Per-Step Dense Rewards (from `graders/grader.py`)

| Event | Reward | Cap |
|-------|-------:|-----|
| Action cost | −0.02 | Every step |
| Redundant tool call | −0.03 | Per duplicate |
| Page fault | −0.05 | Per fault |
| Async timeout | −0.10 | Per premature retrieval |
| Successful disk write | +0.10 | 3/episode |
| Kernel injection | +0.15 | 2/episode |
| Investigation bonus | +0.02–0.05 | First use per tool type |

### 5.2 Terminal Score (Composite)

| Component | Weight | Source |
|-----------|--------|--------|
| Detection (TP=+1.0, TN=+0.5, FP=−0.75, FN=−2.0) | 1.0 | `grader.py` |
| Entity F1 + Findings match | 0.5 | Ground truth comparison |
| Typology accuracy | 0.3 | Exact match |
| Efficiency (steps used / max steps) | 0.2 | Step counter |
| OS mechanics utilization | 0.2 | Disk/async/kernel usage |

### 5.3 GRPO Decomposed Rewards (from `train_grpo.py`)

| Function | Scores | Anti-Gaming |
|----------|--------|-------------|
| **R1** Format Compliance | Valid ```json with known tool | −2.0 for degenerate output |
| **R2** Investigation Quality | Tool category diversity | −0.3 for empty params |
| **R3** Environment Execution | Full multi-step env.step() | Deterministic seeding |
| **R4** OS Mechanics | Unique OS tool usage | Dedup via seen_tools set |

### 5.4 Anti-Gaming Measures

1. Hard caps (3 writes, 2 kernel injections)
2. Deduplication (R4 uses `seen_tools` set)
3. Action cost (−0.02 per step prevents padding)
4. Redundancy penalty (−0.03 for duplicate calls)
5. "Always SAR" trap: E[R_always_SAR] = 0.475 < E[R_reasonable] ≈ 0.68
6. Unique procedural IDs per episode prevent memorization

---

## 6. Unified Qwen 2.5 Architecture

All tiers use the **Qwen 2.5 family** for unified ChatML prompt formatting and consistent tokenization:

| Tier | Model | Purpose | Quantization | VRAM |
|------|-------|---------|-------------|------|
| Training | `unsloth/Qwen2.5-7B-Instruct` | GRPO policy learning | 4-bit NF4 + LoRA r=16 | ~14 GB |
| Compaction | `Qwen/Qwen2.5-1.5B-Instruct` | L1→L2 fact extraction | FP16 | ~3 GB |
| Production | `Qwen/Qwen2.5-72B-Instruct-AWQ` | Real-time inference | AWQ via vLLM | ~38 GB |

**Why unified?** Mixing tokenizers (e.g., Llama + Qwen) causes silent token drift in the L1 sliding window, where precise token counting is critical for eviction decisions.

---

## 7. Training Pipeline

### 7.1 GRPO (Primary Path)

Uses TRL's `GRPOTrainer` — no critic network needed. Advantage is computed from group comparison:

```
A_i = (r_i - mean(r_group)) / std(r_group)
```

| Parameter | Value |
|-----------|-------|
| Model | `unsloth/Qwen2.5-7B-Instruct` (4-bit NF4) |
| LoRA rank | 16 (all attention + MLP projections) |
| Group size (G) | 4 completions per prompt |
| Learning rate | 5e-6 (cosine decay) |
| KL penalty (β) | 0.04 |
| Prompts | 250 unique procedural scenarios |
| Epochs | 2 |
| Gradient accumulation | 8 steps |
| Max completion length | 2048 tokens |
| Compute dtype | float16 (required by Unsloth 4-bit) |
| Hardware | NVIDIA A100 (40 GB) |

### 7.2 Self-Play (Adversarial Curriculum)

Two-agent pipeline in `self_play.py`:
- **Launderer LLM** generates evasive AML scenarios as structured JSON
- **Defender LLM** investigates using the full 18-tool environment
- Zero-sum: Launderer reward = −Defender score
- 9-check validation gate ensures scenario quality

### 7.3 Curriculum Learning (PLR)

`curriculum/plr_engine.py` implements Prioritized Level Replay:
- TD-error-based level sampling prioritizes harder typologies
- Regret oracle (`curriculum/oracle.py`) estimates proxy regret
- Adaptive difficulty scaling during training

---

## 8. AgentOS-Kernel (Production Inference)

### 8.1 Overview

The `agent_os_core/` directory contains the production inference runtime for bare-metal A100 80GB deployment. Zero API calls — everything runs locally.

### 8.2 3-Tier Cognitive Cache

Solves **context starvation** (Lost in the Middle problem):

```
┌─────────────────────────────────────────────────────────┐
│  L1 — Raw Conversation Window (6K tokens)               │
│  │  Sliding window of user/assistant/tool turns          │
│  │  Token-counted via tiktoken (cl100k_base)             │
│  │  Overflow → evict oldest N turns → compact to L2      │
│                                                          │
│  L2 — Structured Scratchpad (4K tokens)                  │
│  │  Entity→facts map: {"CUST-A": ["PEP", "offshore"]}   │
│  │  Compacted by Qwen2.5-1.5B (JSON extraction)         │
│  │  Injected at PROMPT START (high attention position)   │
│  │  Deduplicates facts across compaction cycles          │
│  │  Overflow → archive oldest entities to L3             │
│                                                          │
│  L3 — LanceDB Persistent Archive (unbounded)            │
│     BGE-base-en-v1.5 embeddings (768-dim)                │
│     Cross-encoder gate (BGE-reranker-v2-m3)              │
│     Only injected when relevance score > 0.50            │
│     Injected at PROMPT END (high attention position)     │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Rust Tool Runtime

`agent_os_core/src/lib.rs` — Tokio-based concurrent executor exposed via PyO3:

- **Mock mode**: Built-in handlers with simulated AML data + latency
- **HTTP mode**: Dispatches to `http://<base_url>/tools/<name>` via reqwest
- Timeout + panic safety per tool call
- Batch execution with concurrent `tokio::spawn`
- GIL-bypass: Rust runtime runs outside Python's GIL

### 8.4 VRAM Budget (A100 80GB)

| Component | Model | VRAM |
|-----------|-------|------|
| Reasoning Engine | Qwen2.5-72B-Instruct-AWQ (vLLM) | ~38 GB |
| Compaction Engine | Qwen2.5-1.5B-Instruct | ~3 GB |
| Embedder | BAAI/bge-base-en-v1.5 | ~0.4 GB |
| Reranker | BAAI/bge-reranker-v2-m3 | ~1.1 GB |
| Tool Runtime | Rust/Tokio (PyO3) | 0 GB |
| **Total** | | **~42.5 GB** |
| **Headroom** | | **~37.5 GB ✓** |

---

## 9. Data Flow — Complete Episode

```
1. Client calls POST /reset with task_id
2. openenv_server.py → AMLEnvironment.reset()
3. ScenarioGenerator produces procedural scenario (seeded)
4. StateManager initializes: RAM=[], Disk=[], async_jobs={}, kernel_mode=default
5. Initial observation returned (alert summary)

INVESTIGATION LOOP:
6. Agent receives observation
7. Agent generates tool call (JSON)
8. Client calls POST /step with AMLAction
9. AMLEnvironment dispatches to tool handler
10. StateManager applies OS mechanics:
    - RAM eviction check (2-slot window)
    - Async job ETA tracking
    - Kernel mode validation
11. Grader computes step reward (dense signal)
12. Observation + reward returned

TERMINAL:
13. Agent calls file_sar or close_alert
14. Grader computes composite terminal score
15. Episode complete — final reward returned
```

---

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| **Environment** | |
| `models.py` | Pydantic contracts: `AMLAction`, `AMLObservation`, `AMLState`, `AGUIState` |
| `state_manager.py` | OS kernel: Virtual Memory, Interrupts, Kernel Updates |
| `server/aml_environment.py` | Core env: 18-tool dispatch, reward calculation, scenario lifecycle |
| `openenv_server.py` | FastAPI server: `/reset`, `/step`, `/state`, `/health` |
| `client.py` | HTTP client with typed methods for all 18 tools |
| `inference.py` | Standalone ReAct inference agent (OpenAI-compatible) |
| `openenv.yaml` | OpenEnv environment manifest |
| **Training** | |
| `train_grpo.py` | ★ GRPO training (TRL + Unsloth, 4 reward functions) |
| `self_play.py` | Two-agent self-play orchestrator |
| `eval_harness.py` | Multi-typology benchmark suite |
| `demo_eval.py` | 1MDB demo evaluation with AGUI replay |
| **Scenarios & Grading** | |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario engine |
| `scenarios/compliance_manual.py` | Searchable compliance knowledge base |
| `scenarios/adversary_agent.py` | Adversarial scenario generator |
| `graders/grader.py` | Dense reward engine |
| `curriculum/plr_engine.py` | Prioritized Level Replay |
| **AgentOS-Kernel** | |
| `agent_os_core/agent_os.py` | Production orchestrator (vLLM + cache + Rust) |
| `agent_os_core/memory_manager.py` | L1/L2 cognitive cache with LLM compaction |
| `agent_os_core/l3_index.py` | L3 LanceDB index with cross-encoder gating |
| `agent_os_core/src/lib.rs` | Rust Tokio async runtime (PyO3 bindings) |
| **Infrastructure** | |
| `Dockerfile` | HF Spaces deployment (Python 3.11-slim, port 7860) |
| `requirements.txt` | Runtime dependencies |
| `.hfignore` | Excludes `agent_os_core/` from HF Space |

---

## 11. Deployment

### HuggingFace Spaces (Live Demo)
```bash
openenv push --ignore-file .hfignore
# → https://huggingface.co/spaces/MuazTPM/aml_investigation_env
```

### Docker
```bash
docker build -t memex . && docker run -p 7860:7860 memex
```

### Local Development
```bash
pip install -r requirements.txt
uvicorn openenv_server:app --host 0.0.0.0 --port 8000
python tests/test_smoke.py  # 8/8 tests
```

### AgentOS-Kernel (A100 Bare Metal)
```bash
cd agent_os_core
pip install tiktoken lancedb numpy pyarrow maturin vllm transformers sentence-transformers
maturin develop --release
python test_integration.py  # 16/16 tests (mock mode)
```

---

## 12. Deep Dive — StateManager Internals

The `StateManager` class (`state_manager.py`) is the OS kernel. It is instantiated once per episode at `reset()` and mutated on every `step()`.

### Internal State

```python
self._ram: deque[str]              # maxlen=2, observation summaries
self._disk: List[str]              # agent-written case file entries
self._async_jobs: Dict[str, AsyncJobInfo]  # job_id → status/ETA
self._kernel: List[str]            # base directive + injected modes
self._job_counter: int             # monotonic job ID counter
```

### 6 Valid Kernel Modes

The agent cannot inject arbitrary text. Only these 6 modes are accepted by `inject_directive()`:

| Mode | When to Inject | Effect |
|------|---------------|--------|
| `enhanced_due_diligence` | High-risk customer profile detected | Heightened scrutiny directives |
| `structuring_detection` | Sub-$10K cash pattern identified | CTR threshold awareness |
| `trade_based_ml_detection` | Invoice/customs anomaly found | Price manipulation rules |
| `sanctions_screening` | PEP/watchlist hit returned | OFAC/EU screening procedures |
| `mule_ring_detection` | Device/IP overlap detected | Mule network tracing rules |
| `high_risk_jurisdiction` | BVI/Cayman/FATF jurisdiction seen | Jurisdiction risk escalation |

Invalid modes raise `ValueError` — this prevents prompt injection attacks.

### RAM Eviction Mechanics

```
RAM_CAPACITY = 2  (deliberately low to force memory management)

On each step:
  1. New observation pushed to RAM deque (maxlen=2)
  2. If len(RAM) was already 2, oldest observation is silently evicted
  3. If agent later references an evicted entity ID not saved to disk → Page Fault
  4. Entity IDs tracked via regex: CUST*, ENT_*, TXN-*, REQ-*, ACC-*, ALERT-*
```

---

## 13. Deep Dive — Procedural Scenario Generator

`scenarios/procedural_generator.py` (~2000 lines) builds unique scenarios on every `reset()`.

### Generation Pipeline

```
ScenarioGenerator.generate(difficulty, typology, force_clean)
  │
  ├── Roll clean vs suspicious (30% clean by default)
  │
  ├── If clean: _gen_clean() → legitimate customer, normal txns, false alarm alert
  │
  ├── If suspicious, dispatch by typology:
  │   ├── _gen_structuring() → sub-$10K cash deposits, non-cash occupation
  │   ├── _gen_layering()    → shell companies, multi-hop fan-out, offshore jurisdictions
  │   └── _gen_tbml()        → phantom invoices, over-invoicing, customs mismatches
  │
  ├── _inject_noise() → add decoy profiles + clean transactions (scaled by difficulty)
  │
  └── Return GeneratedScenario conforming to BaseScenario contract
```

### Noise Scaling by Difficulty

| Difficulty | Decoy Profiles | Decoy Transactions |
|:----------:|:--------------:|:------------------:|
| Easy | 0 | 0 |
| Medium | 2 | 4 |
| Hard | 3 | 8 |

### Anti-Memorization

Every scenario uses procedurally generated:
- Entity IDs (`_uid("CUST", 4)` → e.g., `CUSTX3RU`)
- Names (from 44×35 = 1,540 name combinations)
- Company names (from 25×22 = 550 combinations)
- Transaction amounts, dates, timestamps
- Network graph topology (edge count, depth)
- Device fingerprints, IP addresses, MAC addresses
- Customs invoice details (HS codes, weights, values)

### Phase 3 Data Pillars

Each scenario also generates:
- **Device Fingerprints**: shared device IDs / VPN IPs for mule ring detection
- **Customs Invoices**: HS codes, declared values vs. market prices for TBML
- **Beneficial Ownership**: multi-hop UBO chains with ownership percentages

---

## 14. Deep Dive — AgentOS step() Loop

The production orchestrator (`agent_os_core/agent_os.py`) executes this sequence on every user message:

```
async def step(user_message) → StepResult:
  │
  ├─ 1. APPEND: Add user_message to L1 sliding window
  │     └─ memory.add_turn("user", user_message)
  │
  ├─ 2. BOUNDS CHECK:
  │     ├─ While L1 > 6K tokens:
  │     │     └─ Evict oldest 4 turns → Qwen2.5-1.5B compaction → merge into L2
  │     └─ If L2 > 4K tokens:
  │           └─ Archive oldest entities to L3 LanceDB
  │
  ├─ 3. L3 RETRIEVAL:
  │     ├─ Embed user_message via BGE-base-en-v1.5
  │     ├─ Vector search in LanceDB (top-5 by L2 distance)
  │     └─ Cross-encoder gate (BGE-reranker-v2-m3, threshold > 0.50)
  │
  ├─ 4. MEGA-PROMPT ASSEMBLY:
  │     ├─ [SYSTEM] base_prompt + L2 scratchpad preamble
  │     ├─ [USER/ASSISTANT/TOOL] L1 sliding window
  │     └─ [SYSTEM] L3 retrieved context (before last user msg)
  │
  ├─ 5. LLM INFERENCE:
  │     └─ vLLM Qwen2.5-72B-AWQ with guided_json=TOOL_CALL_SCHEMA
  │        → Guarantees valid JSON with reasoning + tool + parameters
  │
  ├─ 6. APPEND RESPONSE: Add LLM output to L1 as assistant turn
  │
  └─ 7. TOOL EXECUTION:
        ├─ Rust ToolRuntime.execute_one(tool_name, params_json, timeout_ms)
        ├─ Append tool result to L1 as tool turn
        └─ Re-check L1 bounds (may trigger another compaction)
```

### JSON Schema Constrained Generation

vLLM enforces this schema — the model **cannot** produce malformed output:

```json
{
  "reasoning": "string (step-by-step thinking)",
  "tool": "enum [query_transactions, trace_network, ...]",
  "parameters": "object (tool-specific params)"
}
```

---

## 15. Deep Dive — Grader Internals

The `AMLGrader` (`graders/grader.py`) is stateless — it receives flags from the environment.

### Terminal Score Composition

```
R_terminal = w_detect * R_detect
           + w_entity * R_entity_f1
           + w_typology * R_typology
           + w_efficiency * R_efficiency
           + w_os * R_os
```

Where:
- **R_detect**: TP=+1.0, TN=+0.5, FP=−0.75, FN=−2.0
- **R_entity_f1**: 2×(P×R)/(P+R) over predicted vs ground-truth entity sets
- **R_typology**: +1.0 if exact match, 0.0 otherwise
- **R_efficiency**: 1.0 − (steps_used / MAX_STEPS), bounded [0, 1]
- **R_os**: composite of page fault count, disk writes, async usage, kernel updates

### "Always SAR" Trap — Formal Proof

```
Given: 70% suspicious, 30% clean scenario mix

E[R_always_SAR] = 0.7 × R_TP + 0.3 × R_FP
               = 0.7 × 1.0 + 0.3 × (−0.75)
               = 0.70 − 0.225
               = 0.475

E[R_reasonable] ≈ 0.7 × 0.8 + 0.3 × 0.4
               ≈ 0.56 + 0.12
               = 0.68

Since 0.475 < 0.68, the lazy "always file SAR" policy is dominated.
```

### Investigation Progress Bonuses

First-use bonuses per tool type (small, non-exploitable):

| Tool | Bonus |
|------|------:|
| `review_alert` | +0.03 |
| `get_customer_profile` | +0.02 |
| `query_transactions` | +0.02 |
| `check_watchlist` | +0.02 |
| `trace_network` | +0.02 |
| `check_source_of_funds` | +0.02 |
| `write_to_case_file` | +0.03 |
| `file_sar` / `close_alert` | +0.05 |
| **Total cap** | **~0.19** |

---

## 16. Deep Dive — Inference Agent (ReAct Loop)

`inference.py` implements a standalone ReAct inference agent compatible with any OpenAI-format API:

```
Configuration:
  API_BASE_URL = env("API_BASE_URL")  # Any OpenAI-compatible endpoint
  MODEL_NAME   = env("MODEL_NAME")    # e.g., "gpt-4o-mini" or local vLLM
  AML_ENV_URL  = env("AML_ENV_URL")   # http://localhost:8000

Loop per task (easy/medium/hard):
  1. POST /reset → initial observation
  2. Build system prompt with dynamic kernel directives
  3. For up to 25 steps:
     a. Send conversation to LLM → get tool call JSON
     b. POST /step with AMLAction → observation + reward
     c. Append observation to conversation history
     d. If terminal → break
  4. Log final score
```

The system prompt includes the full ReAct framework, OS mechanics documentation, investigation protocol, and all 18 tool signatures with parameter types.

---

## 17. Pydantic Data Contracts

`models.py` defines the typed contracts used across the entire system:

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `AMLAction` | Agent → Environment | `tool: str`, `parameters: dict` |
| `AMLObservation` | Environment → Agent | `observation: str`, `reward: float`, `done: bool`, `info: dict` |
| `AMLState` | Full episode state | `scenario_id`, `step_count`, `accumulated_reward`, `tool_call_hashes` |
| `AGUIState` | Frontend visualization | `ram_usage`, `disk_entries`, `async_jobs`, `kernel_directives`, `graph_data` |
| `RAMUsage` | Memory monitor | `capacity: int`, `used: int`, `entries: list` |
| `AsyncJobInfo` | Wire trace tracking | `job_id`, `status`, `eta_steps`, `result` |
| `CurriculumState` | PLR metadata | `difficulty`, `typology`, `episode_number` |

---

## 18. Frontend — Glass Box Visualizer

The Next.js frontend (`frontend/`) provides real-time investigation visualization:

| Panel | Technology | What It Shows |
|-------|-----------|---------------|
| **3D Threat Map** | react-globe.gl | Geographic transaction flows with arc animations |
| **Entity Graph** | Cytoscape.js (cola layout) | Entity relationships, shell company connections |
| **RAM Monitor** | Custom React component | 2-slot context window occupancy |
| **Disk Storage** | Custom React component | Persisted case file entries |
| **Kernel Directives** | Custom React component | Active compliance modes |
| **Investigation Timeline** | Scrollable terminal | Step-by-step tool calls and observations |

**Design**: Brutalist aesthetic — JetBrains Mono, neon orange accents (`#FF6B00`), terminal-style layout. Chosen for the "intelligence terminal" vibe and AML interpretability requirements.

---

## 19. OpenEnv API Reference

The FastAPI server (`openenv_server.py`) exposes the OpenEnv-compatible REST API:

| Endpoint | Method | Request Body | Response |
|----------|--------|-------------|----------|
| `/health` | GET | — | `{"status": "healthy"}` |
| `/reset` | POST | `{"task_id": "easy\|medium\|hard"}` | `AMLObservation` (initial alert) |
| `/step` | POST | `AMLAction` (tool + params) | `AMLObservation` (result + reward + done) |
| `/state` | GET | — | `AMLState` (full episode snapshot) |
| `/agui` | GET | — | `AGUIState` (frontend visualization payload) |

The server runs on port 7860 (HF Spaces convention) and supports both the OpenEnv SDK client and direct HTTP usage.

---

## 20. Testing Strategy

### Smoke Tests (`tests/test_smoke.py`) — 8 tests
- Environment creation and reset
- Tool dispatch (all 18 tools)
- OS mechanics (RAM eviction, async jobs, kernel injection)
- Terminal actions (file_sar, close_alert)
- Reward computation
- Procedural scenario generation
- OpenEnv contract compliance

### AgentOS-Kernel Tests (4 test files, 21+ tests total)

| File | Tests | What It Validates |
|------|:-----:|-------------------|
| `test_runtime.py` | 5 | Rust PyO3 bindings, mock/HTTP modes, batch execution, timeout handling |
| `test_memory.py` | 5 | L1 token counting, eviction, L2 compaction, fact deduplication |
| `test_l3.py` | 5 | LanceDB archival, vector search, cross-encoder gating, prompt injection |
| `test_integration.py` | 6 | Full AgentOS step() loop, L1→L2→L3 cascade, mock mode end-to-end |

All AgentOS tests support `use_mock=True` for CPU-only execution (no GPU required).

---

## 21. Glossary

| Term | Definition |
|------|-----------|
| **SAR** | Suspicious Activity Report — the terminal "positive" decision |
| **POMDP** | Partially Observable Markov Decision Process |
| **GRPO** | Group Relative Policy Optimization — RL algorithm from DeepSeek |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning |
| **QLoRA** | Quantized LoRA — 4-bit NF4 base + FP16 LoRA adapters |
| **AWQ** | Activation-aware Weight Quantization — inference quantization |
| **vLLM** | High-throughput LLM serving engine with PagedAttention |
| **ChatML** | Chat Markup Language — Qwen's prompt format |
| **PLR** | Prioritized Level Replay — curriculum learning algorithm |
| **ReAct** | Reason + Act framework for tool-using LLM agents |
| **Page Fault** | Penalty when referencing evicted, unsaved data (−0.05) |
| **Async Timeout** | Penalty for premature wire trace retrieval (−0.10) |
| **Kernel Mode** | One of 6 valid compliance directives injectable into system prompt |
| **L1/L2/L3** | Three-tier cognitive cache (sliding window / scratchpad / vector archive) |
| **Entity F1** | Precision-recall F1 score over predicted vs ground-truth entity sets |
| **CTR** | Currency Transaction Report — $10K reporting threshold |
| **PEP** | Politically Exposed Person |
| **UBO** | Ultimate Beneficial Owner |
| **TBML** | Trade-Based Money Laundering |
| **SWIFT** | Society for Worldwide Interbank Financial Telecommunication |
| **GIL** | Global Interpreter Lock — bypassed by Rust/PyO3 tool runtime |

---

## 22. References

1. **TIPS** (ICLR 2026) — Xi R. et al. Dense per-step reward shaping via potential-based functions. → [arXiv](https://arxiv.org/abs/2503.02197)
2. **DeepSeekMath** (2024) — Shao et al. Introduced GRPO. → [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
3. **DeepSeek-R1** (2025) — Pure RL emergent reasoning. → [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
4. **ReAct** (ICLR 2023) — Yao et al. Interleaved reasoning + tool-use. → [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
5. **Prioritized Level Replay** (ICML 2021) — Jiang et al. TD-error curriculum. → [arXiv:2010.03934](https://arxiv.org/abs/2010.03934)
6. **LoRA** (2021) — Hu et al. Low-rank adaptation. → [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
7. **QLoRA** (2023) — Dettmers et al. 4-bit NF4 + LoRA. → [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
8. **TRL** — HuggingFace. GRPOTrainer implementation. → [GitHub](https://github.com/huggingface/trl)
9. **Unsloth** — Daniel & Michael Han. Fast LoRA with Triton kernels. → [GitHub](https://github.com/unslothai/unsloth)
10. **Lost in the Middle** (2023) — Liu et al. Context position bias in LLMs. → [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
