# Memex: Project Context

> Living document tracking the current state of the Memex OS-Agent Benchmark.
> Last updated: 2026-04-25

## What Is Memex?

Memex is a **B2B enterprise benchmarking environment** built for the Meta / Hugging Face OpenEnv Hackathon. It tests whether an LLM can function as a Turing-complete Operating System over long-horizon tasks, using complex Anti-Money Laundering (AML) investigations as the obstacle course.

The "OS" metaphor is an **architectural framework**, not a literal operating system. It solves the LLM context window bottleneck by giving the agent explicit tools to manage its own memory, handle async operations, and self-improve its decision rules.

---

## Current Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Models** | `models.py` | Single source of truth for Pydantic types: `AMLAction`, `AMLObservation`, `AMLState` (with `submitted_typology`, `entities_flagged`, `decision_action`, `async_poll_count`), `AsyncJobInfo`, `AGUIState`, `CurriculumState` |
| **State Manager** | `state_manager.py` | OS mechanics engine: RAM eviction (2-slot context), Disk persistence, Async Job Queue, Kernel Directives. Entity ID regex matches procedural alphanum patterns (`CUST[A-Z0-9]{3,6}`, `ENT`, `TXN`) |
| **Procedural Generator** | `scenarios/procedural_generator.py` | Dynamic POMDP scenario builder: 3 typologies × 3 difficulties, unique IDs per episode |
| **Environment** | `server/aml_environment.py` | Core environment: 18 tool handlers + State Manager integration + scenario injection support for self-play |
| **Launderer Env** | `server/launderer_env.py` | One-step MDP for Launderer training: robust 3-strategy JSON extraction, schema validation, shaped reward tiers, VRAM-safe frozen Defender scoring |
| **Grader** | `graders/grader.py` | Dense reward: per-step micro-rewards + terminal composite score [-1.0, +1.0]. Recognizes `false_positive` as clean-scenario typology |
| **OpenEnv Server** | `openenv_server.py` | Production FastAPI entrypoint — OpenEnv SDK `create_app()` + standalone fallback |
| **Legacy Server** | `server/app.py` | FastAPI HTTP server (dual-mode: OpenEnv / standalone, used by trainers) |
| **Client** | `client.py` | HTTP client with all 18 tool wrappers |
| **Inference** | `inference.py` | ReAct agent loop with OS-mechanic awareness |

### Self-Play Training Pipeline

| Component | File | Purpose |
|-----------|------|---------|
| **Self-Play Orchestrator** | `self_play.py` | **Production training path.** Alternating best-response: Warmup → [Launderer → Defender] × N rounds. Population-based checkpoint management. Linear mix ratio schedule (0.3 → 0.7). Real score propagation from trainers |
| **Defender PPO** | `train_defender_ppo.py` | Step-level PPO with GAE (γ=0.99, λ=0.95). Mixed-mode training (procedural + Launderer scenarios). Entity F1, typology accuracy, decision label (TP/TN/FP/FN) tracking. Terminal reward de-duplication to prevent GAE double-counting |
| **Launderer PPO** | `train_launderer_ppo.py` | Single-step MDP PPO. Generates evasive scenario JSON, scored against frozen Defender. VRAM-safe full unload/reload cycle. KL auto-revert with advantage variance guard |
| **Standalone PPO** | `train_ppo.py` | Legacy step-level PPO (Unsloth 4-bit + LoRA, L4-optimized, `--use-plr`) |
| **GRPO** | `train_grpo.py` | Group Relative Policy Optimization (**EXPERIMENTAL** — no clipping, gradient coupling, requires `--experimental`) |
| **PPO 70B** | `train_ppo_70b.py` | Multi-GPU DeepSpeed ZeRO-3 PPO for 70B on A100 cluster (proof of scalability) |
| **DPO** | `train_dpo.py` | Offline DPO continuous learning from user preference pairs |
| **Adversary (Legacy)** | `train_adversary.py` | GAN-style heuristic adversarial loop (**DEPRECATED** — use `self_play.py`) |

### Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| **LoRA Hot-Swap** | `hotswap.py` | Zero-downtime LoRA adapter reload into running models |
| **PLR Engine** | `curriculum/plr_engine.py` | Prioritized Level Replay: regret-weighted scenario sampling buffer |
| **Proxy Oracle** | `curriculum/oracle.py` | Regret computation: `1.0 - protagonist_score` |
| **Demo** | `demo_eval.py` | 1MDB-inspired demo with AGUI replay capture |
| **Eval Harness** | `eval_harness.py` | Checkpoint benchmarking across typology/difficulty grid |
| **Adversary Agent** | `scenarios/adversary_agent.py` | LLM-backed evasive scenario generator (local Llama-3.1-8B or procedural fallback) |
| **Compliance Manual** | `scenarios/compliance_manual.py` | Searchable AML rule corpus for kernel updates |
| **Tests** | `tests/test_smoke.py` | 8 end-to-end smoke tests |
| **PLR Tests** | `tests/test_plr.py` | PLR engine unit tests |

### OS Mechanics

1. **Virtual Memory (RAM Eviction)**
   - Agent context window = last 2 observations only
   - `write_to_case_file(content)` → pages data to persistent disk (+0.10 reward, hard-capped at 3 per episode)
   - Page Fault: referencing evicted data NOT on disk → -0.05 penalty

2. **Interrupts (Async Queue)**
   - `request_wire_trace(entity_id)` → background job with 2–4 step ETA
   - `retrieve_async_result(job_id)` → fetch when ETA=0
   - Async Timeout: premature retrieval → -0.10 penalty
   - `async_poll_count` tracked in `AMLState` for correct success rate computation

3. **Kernel Updates (Self-Improvement)**
   - `search_compliance_manual(query)` → find AML rules
   - `update_system_prompt(rule)` → inject into active kernel directives (+0.15 reward, hard-capped at 2 per episode)

### Tool Roster (18 Total)

**Domain Investigation Tools (10):** `review_alert`, `get_customer_profile`, `query_transactions`, `check_watchlist`, `trace_network`, `check_source_of_funds`, `check_market_price` (TBML commodity pricing), `assess_risk`, `file_sar` / `close_alert` (terminal)

**Phase 3 — FinCEN Investigation Tools (3):** `check_device_overlap` (mule-ring detection), `verify_customs_invoice` (TBML phantom shipments), `query_beneficial_ownership` (UBO shell-layer tracing)

**OS-Mechanic Tools (5):** `write_to_case_file`, `request_wire_trace`, `retrieve_async_result`, `search_compliance_manual`, `update_system_prompt`

### Reward System

**Per-step:** action cost (-0.02), redundancy (-0.03), unique tool (+0.03), page fault (-0.05), async timeout (-0.10), disk write (+0.10, max 3/episode), kernel inject (+0.15, max 2/episode)

**Terminal:** decision accuracy (0.30) + typology (0.15) + findings coverage (0.20) + entity F1 (0.15) + UBO identification (0.05) + Phase 3 pillar tools (0.05) + efficiency (0.10) → mapped to [-1.0, +1.01] (including OS micro-rewards)

### Scenario Generation

3 ML typologies: **Structuring** (smurfing), **Layering** (shell companies), **Trade-Based ML** (invoice manipulation)
3 difficulties: **Easy** (1 culprit, minimal noise), **Medium** (3+ entities, moderate noise), **Hard** (5+ entities, deep networks, decoys)

All entity IDs are procedurally generated per episode — no memorization possible.

---

## Test Status

```
✓ 9/9 typology/difficulty combos generate valid scenarios
✓ Unique IDs confirmed across episodes (anti-memorization)
✓ Compliance manual search works
✓ Noise scales with difficulty
✓ Full episode: structuring (easy) → score +1.01
✓ Full episode: layering (medium) → score +1.01
✓ Full episode: trade-based ML (hard) → score +1.01
✓ PLR curriculum engine: buffer, sampling, metrics, save/load
```

**All 8/8 smoke tests pass.**

---

## Frontend Architecture: AGUI (Agentic Graphical User Interface)

The backend is invisible by design — an LLM managing RAM eviction and async queues produces no visual signal. The **AGUI** solves this by emitting a strict JSON `agui_state` payload after every environment step, which the Next.js frontend consumes to render a real-time **5-Panel Tactical Dashboard**:

```
┌─────────────────────────────────────┬─────────────────────────────────┐
│        ENTITY GRAPH (Panel 1)       │   SYSTEM RESOURCES (Panel 3)    │
│        (60% Width)                  │    - RAM & Disk Storage         │
│                                     │    - Active Async Processes     │
│  [3D Globe / Flat Map Toggle]       │    - Kernel Directives          │
│  react-globe.gl vectors             ├─────────────────────────────────┤
│  Cytoscape.js Cola Physics          │   AGENT TERMINAL (Panel 4)      │
│                                     │    - ReAct scratchpad reasoning │
│  (Smooth Incremental Updates)       │    - Tool execution logs        │
│                                     ├─────────────────────────────────┤
│   *GLOBAL THREAT MAP (Panel 2)      │ CURRICULUM ENGINE (Panel 5, opt)│
│    is toggled with Entity Graph     │    - PLR Regret Metrics         │
│                                     │    - Scenario Difficulty Gauge  │
└─────────────────────────────────────┴─────────────────────────────────┘
```

**Data contract:** `state_manager.py` builds the `AGUIState` object (defined in `models.py`) containing `ram_usage` (capacity string + active context list), `disk_storage` (list of persisted findings), `async_jobs` (job IDs with ETAs and statuses), `kernel_directives` (base + agent-injected rules), and `curriculum` (PLR buffer state: enabled flag, buffer_size, mean_regret, max_regret, difficulty gauge, diversity score). This payload is nested inside `observation.metadata.agui_state` on every `/step` response.

**Frontend replay:** `demo_eval.py` captures per-step AGUI snapshots as `step_NNN.json` files in `demo_output/`. The Next.js frontend reads these sequentially to replay the full investigation as an animated OS simulation.

---

## Two-Agent Self-Play Architecture

The **production adversarial training approach** is two-agent PPO self-play, replacing the legacy heuristic-based `train_adversary.py`.

### Architecture

```
┌─────────────────────────┐                   ┌─────────────────────────┐
│    LAUNDERER AGENT       │                   │     DEFENDER AGENT      │
│ (train_launderer_ppo.py) │    scenario       │ (train_defender_ppo.py) │
│                          │ ───────────────►  │                         │
│  Single-step MDP:        │                   │  Multi-step MDP:        │
│   Generate evasive JSON  │  -defender_score  │   18 investigation      │
│   Schema-validated       │ ◄───────────────  │   tools, 25 steps max   │
│   3-strategy extraction  │   = launderer     │                         │
│                          │     reward        │  Tracks: decision_action│
│  LoRA on Llama-3.1-8B    │                   │  submitted_typology,    │
│  VRAM: ~6 GB loaded      │                   │  entities_flagged       │
└──────────────────────────┘                   └─────────────────────────┘
         ▲                                              │
         │              self_play.py                     │
         │    ┌──────────────────────────────┐           │
         └────┤  Orchestrator               ├───────────┘
              │  Phase 1: Warm-start (20i)  │
              │  Phase 2: Launderer (10i)   │
              │  Phase 3: Defender (15i)    │
              │  × 3 outer rounds           │
              │  Mix: 0.3 → 0.7 (linear)   │
              │  Population best() tracking │
              └──────────────────────────────┘
```

### Key Design Decisions

1. **VRAM-Safe Model Swapping:** Full unload/reload (not LoRA adapter swap) between agents. Zero shared state guaranteed. Both agents use ~6 GB loaded, ~12 GB during training. Peak VRAM never exceeds 13 GB on L4 (24 GB available).

2. **Directional KL Penalty:** All 3 trainers use `kl.clamp(min=0)` (not `kl.abs()`) — only penalizes divergence FROM reference model, does not penalize convergence TOWARD it.

3. **Terminal Reward De-Duplication:** The Defender's GAE advantage computation subtracts prior step reward sum from the terminal composite score to prevent double-counting.

4. **Scenario Injection:** `AMLEnvironment.reset()` accepts an optional `scenario` parameter, allowing Launderer-generated scenarios to be injected during Phase 3 mixed training.

5. **Population Checkpointing:** `self_play.py` maintains a `CheckpointPopulation` that tracks `best_score` per agent per round, selecting the actual best checkpoint for adversarial scoring.

### Launderer Reward Shaping

| Outcome | Reward | Description |
|---------|--------|-------------|
| JSON parse fail | -2.0 | No extractable JSON from model output |
| Schema fail | -1.0 | JSON found but missing required fields |
| Valid, Defender catches | 0.1 – 0.3 | Valid scenario, but Defender succeeds |
| Valid, Defender fails | 0.3 – 1.0 | Maximum: Defender fully fooled |

Formula: `reward = 0.1 + 0.9 × (1.0 - clamp(defender_score, -1, 1)) / 2.0`

### Detection Metrics

The Defender trainer tracks ground-truth metrics persisted directly in `AMLState`:

| Metric | Source Field | Description |
|--------|-------------|-------------|
| Decision label | `st.decision_action` | `"file_sar"` or `"close_alert"` → TP/TN/FP/FN |
| Typology accuracy | `st.submitted_typology` | Exact match vs `ground_truth.typology` |
| Entity F1 | `st.entities_flagged` | Set intersection vs `ground_truth.key_entities` |

---

## L4 Training Pipeline & VRAM Optimization

Training an 8B-parameter language model on consumer GPU hardware (NVIDIA L4, ~24GB VRAM) demands aggressive memory engineering.

### 1. Unsloth 4-bit Quantization + LoRA

The base model (Meta-Llama-3.1-8B-Instruct) is loaded via Unsloth's `FastLanguageModel` in 4-bit NF4 quantization (~5.5GB VRAM). Only the LoRA adapter layers (rank 16, alpha 32) are trainable — roughly 0.92% of total parameters (~42M trainable / 4.58B total).

### 2. The `disable_adapter()` Trick

Standard PPO requires a frozen reference model to compute the KL divergence penalty. Naively, this means loading two copies of the model (~10GB each) — impossible on an L4.

Our solution: **one model, two modes.** During the forward pass, we call `model.disable_adapter()` to temporarily bypass the LoRA layers, producing reference logits from the frozen base weights. Then `model.enable_adapter()` restores the trainable policy. This yields an exact KL penalty with zero additional VRAM overhead.

### 3. VRAM Budget (L4 = 24 GB)

| Component | VRAM |
|-----------|------|
| Base 8B 4-bit (NF4) | ~5.5 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| KV cache (2048 seq) | ~2.0 GB |
| Optimizer (AdamW fp32) | ~1.2 GB |
| Activations (gradient checkpoint) | ~3-6 GB |
| **Total** | **~12-15 GB** |
| **Headroom** | **~9 GB ✓** |

> Only one model is loaded at a time during self-play. The Launderer and Defender never coexist in VRAM.

### 4. PPO Stability Engineering

Both trainers include **11 production-grade safety features:**

- **Mean per-token KL:** `.mean()` not `.sum()` — scale-invariant KL divergence
- **Directional KL:** `kl.clamp(min=0)` — only penalizes divergence from reference
- **Entropy bonus:** `- entropy_coef × H(π)` prevents mode collapse
- **Ratio clamping:** `clamp(log_ratio, -10, 10)` before `exp()` — prevents inf/NaN
- **Return clipping:** `clip(returns, -2.0, +2.0)` — bounds gradient signals
- **Empty response guard:** Dummy EOS prevents NaN from `.mean()` on empty tensor
- **Degenerate response detection:** >80% repeated tokens → -0.15 penalty
- **Fault-tolerant `env.step()`:** try/except → -0.10 penalty for malformed actions
- **Type-safe `parse_action()`:** Forces `params` to dict, catches TypeError/ValueError
- **KL early stopping:** Break PPO epochs if |KL| > 15
- **Auto-Revert ("Time Machine"):** Entropy heartbeat + checkpoint reload with hyperparameter bump

### 5. DPO Continuous Learning Pipeline

A human-in-the-loop feedback system for post-deployment improvement:

- **Frontend:** Prisma-backed SQLite database stores `PreferencePair` records. Next.js API route (`/api/preferences`) exposes POST/GET/PATCH endpoints.
- **DPO Trainer (`train_dpo.py`):** Pulls unconsumed pairs, runs DPO loss against frozen base using `disable_adapter()`, saves updated LoRA adapters.
- **Hot-Swap (`hotswap.py`):** Reloads updated adapters into a live model with zero downtime.

---

## Pitch Strategy: Gym vs. Stage

Memex uses a **dual-data architecture** to satisfy two orthogonal goals: RL generalization and judge persuasion.

### The Gym — `procedural_generator.py` + `self_play.py`

The training environment. `procedural_generator.py` creates mathematically fresh POMDP graphs. `self_play.py` adds adversarial scenarios from the Launderer. Entity IDs are randomly generated — no memorization possible. This forces the PPO agent to learn **transferable OS mechanics**.

### The Stage — `demo_eval.py`

A hardcoded scenario inspired by the **1MDB sovereign wealth fund scandal**: 5 named entities, 8 transactions totaling $681M, ground truth: `file_sar`, typology `layering`.

Two modes:
1. **Scripted** (`--dry-run`): 15 hardcoded steps, perfect +1.01 score.
2. **LLM-driven** (`--model checkpoints/best`): Trained agent investigates autonomously.

Both modes capture AGUI state for frontend replay.

---

## 70B Distributed Training Pipeline (A100 Cluster)

Implemented in `train_ppo_70b.py` using DeepSpeed ZeRO-3.

### VRAM Budget (4× A100-80GB)

| Component | Total | Per-GPU (sharded) |
|-----------|-------|-------------------|
| Model weights (4-bit NF4) | ~35 GB | ~9 GB |
| LoRA adapters (r=32, α=64) | ~150 MB | Replicated on each GPU |
| Optimizer states (fp32 AdamW) | ~280 GB | ~70 GB |
| Gradients (bf16) | ~140 GB | ~35 GB |
| **Peak per-GPU** | — | **~50 GB / 80 GB** |

### Architectural Decisions

1. **Rank-0 Environment:** Only rank 0 runs the AML environment. Trajectories broadcast via `torch.distributed.broadcast_object_list()`.
2. **`disable_adapter()` Under ZeRO-3:** LoRA operates at module level, independent of parameter sharding.
3. **DeepSpeed Engine:** Handles `backward()`, gradient accumulation, all-reduce, and clipping internally.
4. **Distributed Checkpointing:** `engine.save_checkpoint()` saves sharded states per-rank. Tokenizer saved by rank 0 only.

---

## Dependencies

- **Runtime:** Python 3.10+, FastAPI, Pydantic v2, httpx, openai, openenv-core
- **Training (optional):** unsloth, trl, peft, torch, bitsandbytes, accelerate, wandb
- **Validation:** openenv-core, Docker

---

## Deployment

- **HF Spaces:** `openenv push --ignore-file .hfignore` → deploys to `MuazTPM/aml_investigation_env` (Docker, port 7860)
- **OpenEnv CLI:** `openenv serve` → reads `openenv.yaml` → `openenv_server:app`
- **Local:** `uvicorn openenv_server:app --host 0.0.0.0 --port 8000`
- **Docker:** `docker build -t memex . && docker run -p 7860:7860 memex`
- **Smoke Tests:** `python tests/test_smoke.py` → 8/8 tests

---

## Recent Changes

### 2026-04-25

1. **Final audit fixes:** P1-1 (orchestrator score propagation), P2-2 (KL direction: `kl.abs()` → `kl.clamp(min=0)` in all 3 trainers), grader `false_positive` typology alias for clean-scenario TN detection.
2. **Self-play dry-run validated on Colab L4:** Full 3-round orchestrator dry-run completes end-to-end. VRAM peak ~12 GB. Scenario pre-generation: 60/60 valid.
3. **Documentation rewrite:** Updated README, TRAINING, and PROJECT_CONTEXT to reflect self-play as production training path.

### 2026-04-24

1. **Self-play architecture implemented:** `self_play.py` orchestrator, `train_defender_ppo.py` (GAE, mixed scenarios, entity-F1/typology tracking), `train_launderer_ppo.py` (single-step MDP, VRAM-safe scoring), `server/launderer_env.py` (robust JSON extraction, shaped rewards).
2. **Critical audit fixes:** P0-1 (entity ID regex), P0-2 (detection labels / TN via `decision_action`), P1-2 (`async_poll_count`), P1-3 (terminal reward de-duplication), P1-1 (orchestrator score propagation).
3. **AMLState extensions:** Added `submitted_typology`, `entities_flagged`, `decision_action`, `async_poll_count` for ground-truth metric persistence.
4. **Scenario injection:** `AMLEnvironment.reset()` accepts optional `scenario` parameter for mixed-mode training.
5. **Documentation rewrite:** Corrected tool count from 15 to 18, verified terminal reward weights against `grader.py`.
6. **PLR Curriculum Engine:** Added `curriculum/plr_engine.py`, 5th AGUI panel (`CurriculumPanel.tsx`), and `--use-plr` flag.

### 2026-04-23

1. **OpenEnv Server:** Production-grade FastAPI wrapper using OpenEnv SDK `create_app()`. HF Spaces ready (port 7860).
2. **HF Space deployed:** `MuazTPM/aml_investigation_env` live.
3. **Frontend Overhaul:** 2-column AGUI layout, 3D threat mapping via `react-globe.gl`, Cytoscape `cola` physics.
4. **Reward Integrity:** Hard caps on disk writes (3) and kernel injections (2) per episode.

### 2026-04-22

1. **PPO stability overhaul (10 fixes):** Mean KL, entropy bonus, ratio clamping, return clipping, empty response guards.
2. **Auto-Revert:** Entropy heartbeat monitor with checkpoint time machine.

### 2026-04-20

1. **Codebase sanitization:** Removed duplicate types, updated `client.py`, rewrote README.
2. **PPO trainer:** Custom step-level PPO with Unsloth 4-bit + LoRA, L4-optimized.
3. **70B scaling pivot:** `train_ppo_70b.py` with DeepSpeed ZeRO-3.
4. **Demo system:** 1MDB-inspired scenario with AGUI replay capture.
