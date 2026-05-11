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

## 12. References

1. **TIPS** (ICLR 2026) — Dense per-step reward shaping
2. **DeepSeekMath** (2024) — GRPO algorithm
3. **DeepSeek-R1** (2025) — Pure RL emergent reasoning
4. **ReAct** (ICLR 2023) — Interleaved reasoning + tool-use
5. **Prioritized Level Replay** (ICML 2021) — Curriculum learning
6. **LoRA** (2021) — Parameter-efficient fine-tuning
7. **QLoRA** (2023) — 4-bit NF4 quantization + LoRA
8. **TRL** — GRPOTrainer implementation
9. **Unsloth** — Fast LoRA/QLoRA with Triton kernels
10. **Lost in the Middle** (2023) — Context position bias in LLMs
