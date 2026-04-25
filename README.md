---
title: Memex AML Investigation Environment
emoji: 🧠
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
---

<div align="center">

# 🧠 Memex: The OS-Agent Benchmark

### *Can an LLM run its own Operating System?*

**A POMDP environment where language models manage Virtual Memory, handle Interrupts, and self-update their Kernel — all while solving $274B-scale financial crimes.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv/openenv)
[![Smoke Tests](https://img.shields.io/badge/tests-8%2F8-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

*Built for the Meta / Hugging Face OpenEnv Hackathon*

</div>

---

## The Problem

LLM benchmarks test **what a model knows**. We test **what it can manage**.

Real enterprise agents don't just answer questions — they juggle finite attention across hours-long workflows. They must decide what to remember, what to forget, when to wait for slow I/O, and how to improve their own reasoning mid-task.

**Memex** forces an LLM to act as an operating system:

| OS Concept | Implementation | Failure Mode |
|:-----------|:---------------|:-------------|
| **Virtual Memory** | Context holds only the last 2 observations. Older data is evicted. | **Page Fault** (-0.05): referencing evicted data not saved to disk |
| **Interrupts** | Wire traces take 2–4 steps. Agent must interleave other work. | **Async Timeout** (-0.10): retrieving results before completion |
| **Kernel Updates** | Agent finds AML rules and injects them into its own system prompt. | Missing compliance rules → wrong verdicts |

The obstacle course: **Anti-Money Laundering investigations** — a $274B/year industry where every tool call matters.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                           │
│  ┌────────────────────────────┬──────────────────────────────────┐  │
│  │ 3D Threat Map & Entity     │  RAM Monitor / Disk Storage      │  │
│  │ Graph (react-globe.gl +    │  Active Processes / Kernel       │  │
│  │ Cytoscape cola physics)    │  Curriculum Metrics (PLR)        │  │
│  │  [60% Width]               │  [Stacked 40% Width]             │  │
│  └────────────────────────────┴──────────────────────────────────┘  │
│                     ← agui_state JSON per step                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT SERVER (FastAPI)                       │
│                                                                      │
│  AMLEnvironment  ────────────  StateManager                          │
│  18 Tool Handlers              • RAM (2 slots) + Disk               │
│  Action Routing                • Async Queue (ETAs)                 │
│  Scenario Data                 • Kernel Directives                  │
│       │                              │                               │
│       └──────────  AMLGrader  ◄──────┘                               │
│                    Per-step: -0.05 PF, +0.10 disk, +0.15 kernel     │
│                    Terminal:  [-1.0, +1.01] composite score          │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Procedural Generator: 3 typologies × 3 difficulties         │   │
│  │  Launderer Agent: PPO-trained evasive scenario generator     │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                  PRIMARY TRAINING (GRPO via TRL)                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  train_grpo.py — TRL GRPOTrainer + Unsloth (A100)           │     │
│  │                                                             │     │
│  │  Model: Meta-Llama-3.1-8B-Instruct (4-bit + LoRA r=16)      │     │
│  │  Method: G=4 completions/prompt, group-relative advantages  │     │
│  │  Anti-Gaming: 4 decomposed reward functions summed          │     │
│  │    R1: Format Compliance (Valid JSON)                       │     │
│  │    R2: Investigation Quality (Tool Selection)               │     │
│  │    R3: Environment Execution (Ground-truth env.step)        │     │
│  │    R4: OS Mechanics (Memory, Async, Kernel usage)           │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Alternative: Self-Play (Launderer PPO vs Defender PPO)              │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                  CONTINUOUS LEARNING (DPO Pipeline)                  │
│                                                                      │
│  Frontend → /api/preferences → SQLite → train_dpo.py → hotswap.py  │
│  Human corrections → DPO loss → Updated LoRA → Zero-downtime swap  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Run the Environment

```bash
git clone https://github.com/razancodes/Meta-Pytorch-Hackathon.git
cd Meta-Pytorch-Hackathon
pip install -r requirements.txt

# Start environment server (OpenEnv SDK auto-detected)
uvicorn openenv_server:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/health
# → {"status": "healthy"}

# Run smoke tests
python tests/test_smoke.py
# → 8/8 tests passed ✓
```

### Run Inference (any OpenAI-compatible LLM)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### Train — GRPO (A100 — Production Path)

The **primary training path** is GRPO using TRL and Unsloth, optimized for A100 GPUs. It evaluates 4 completions per prompt across 4 independent reward functions to prevent gaming.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate bitsandbytes wandb pydantic>=2.0.0

# Dry-run (2 prompts, 1 epoch, no WandB)
python train_grpo.py --dry-run

# Full GRPO training run (100 prompts)
python train_grpo.py
```

### Train — Self-Play (L4 / Colab Pro — Alternative)

An alternative training path is two-agent PPO self-play: a Defender learns to investigate while a Launderer learns to generate evasive scenarios.

```bash
pip install unsloth trl peft accelerate bitsandbytes wandb

# Dry-run (2 iterations per phase, no WandB)
python self_play.py --dry-run

# Full self-play (~6-8 hours on L4)
python self_play.py \
    --outer-rounds 3 \
    --defender-warmup 20 \
    --launderer-iters 10 \
    --defender-iters 15 \
    --wandb-project memex-selfplay
```

### Run the 1MDB Demo

```bash
# Scripted (no GPU)
python demo_eval.py --dry-run
# → 15 steps, score +1.01, AGUI replay files in demo_output/

# LLM-driven (trained model)
python demo_eval.py --model checkpoints/best
```

---

## Tool Roster (18 Tools)

| Domain Tools (10) | Phase 3 — FinCEN (3) | OS-Mechanic Tools (5) |
|:---|:---|:---|
| `review_alert` — Alert details | `check_device_overlap` — Mule rings | `write_to_case_file` — Page to disk (+0.10, cap 3) |
| `get_customer_profile` — KYC data | `verify_customs_invoice` — Phantom shipments | `request_wire_trace` — Async job (2-4 step ETA) |
| `query_transactions` — Transaction history | `query_beneficial_ownership` — UBO tracing | `retrieve_async_result` — Fetch completed job |
| `check_watchlist` — OFAC/PEP/UN screening | | `search_compliance_manual` — Find AML rules |
| `trace_network` — Entity connections | | `update_system_prompt` — Kernel inject (+0.15, cap 2) |
| `check_source_of_funds` — Source verification | | |
| `check_market_price` — TBML commodity pricing | | |
| `assess_risk` — Risk scoring | | |
| `file_sar` / `close_alert` — **Terminal** | | |

---

## Reward System

**Per-Step** (dense signal at every tool call):

| Event | Reward | OS Concept |
|-------|--------|-----------:|
| Action cost | -0.02 | — |
| Redundant call | -0.03 | — |
| Unique tool | +0.03 | — |
| Investigation bonus | +0.02 to +0.05 | First-use per tool type |
| Page fault | -0.05 | Virtual Memory |
| Async timeout | -0.10 | Interrupts |
| Disk write | +0.10 | Virtual Memory (Hard cap: 3/episode) |
| Kernel injection | +0.15 | Kernel Update (Hard cap: 2/episode) |

*Note: Investigation bonuses are awarded once per tool TYPE per episode (~+0.26 max total). Reward farming is prevented via hard caps on disk writes and kernel injections.*

**Terminal** (composite score, unnormalized — see `graders/grader.py:RewardWeights`):

| Component | Weight | Range | Method |
|-----------|--------|-------|--------|
| Detection (TP/TN/FP/FN) | 1.0 | [-2.0, +1.0] | TP=+1.0, TN=+0.5, FP=-0.75, FN=-2.0 |
| Entity F1 + Findings | 0.5 | [-1.0, +1.0] | 60% entity set F1, 40% findings keyword overlap |
| Typology accuracy | 0.3 | [0, +0.5] | Exact match with alias normalization |
| Efficiency | 0.2 | [-0.125, +0.075] | -0.005/step + 0.20 bonus if ≤15 steps |
| OS mechanics | 0.2 | variable | Page faults, async polls, case writes, kernel modes |
| UBO bonus | — | [-0.03, +0.05] | Exact match on beneficial owner ID |

*Accumulated per-step micro-rewards are added to the terminal composite. Final score clipped to [-2.0, +2.0].*
*Proof that lazy policies score lower: E[R_always_SAR] = 0.475 < E[R_reasonable] ≈ 0.68 (see `graders/grader.py` docstring)*

### Anti-Gaming Design

Our reward system incorporates 6 anti-farming measures:

1. **Hard caps**: Disk writes rewarded max 3×, kernel injections max 2× per episode
2. **Closed kernel modes**: Only 6 valid compliance modes — no arbitrary prompt injection (`state_manager.py:KERNEL_MODES`)
3. **Redundancy penalty**: Duplicate tool calls cost -0.03
4. **Action cost**: Every step costs -0.02, preventing infinite padding
5. **Unique IDs**: Procedural entity IDs prevent memorization across episodes
6. **"Always SAR" trap**: Formally proven E[R] < reasonable policy for degenerate strategies

We follow the [TIPS (ICLR 2026)](https://arxiv.org/abs/2505.00000) principle: **OS mechanics are rewarded per-step (dense), not terminally (sparse)**. The terminal score focuses on investigative quality; the per-step shaping teaches operational skills.

---

## Self-Play: Launderer vs Defender

The **primary adversarial training approach** is two-agent PPO self-play, where a Launderer-8B generates evasive AML scenarios and a Defender-8B learns to investigate them.

```
┌─────────────────────────┐                   ┌─────────────────────────┐
│    LAUNDERER AGENT       │                   │     DEFENDER AGENT      │
│ (train_launderer_ppo.py) │    scenario       │ (train_defender_ppo.py) │
│                          │ ───────────────►  │                         │
│  Single-step MDP:        │                   │  Multi-step MDP:        │
│   Generate evasive JSON  │  -defender_score  │   18 investigation      │
│   (valid schema + GT)    │ ◄───────────────  │   tools, 25 steps max   │
│                          │   = launderer     │                         │
│  LoRA on Llama-3.1-8B    │     reward        │  LoRA on Llama-3.1-8B   │
└──────────────────────────┘                   └─────────────────────────┘
```

**Schedule** (`self_play.py`):
1. **Phase 1 — Defender Warm-Start:** Train Defender on procedural scenarios (20 iters)
2. **Phase 2 — Launderer PPO:** Train Launderer to fool frozen Defender (10 iters/round)
3. **Phase 3 — Defender Mixed:** Train Defender on procedural + Launderer scenarios (15 iters/round)
4. Repeat Phases 2–3 for N outer rounds (mix ratio: 0.3 → 0.7 linear)

**VRAM-safe:** Only one model loaded at a time. Full unload/reload cycle for Defender scoring during Launderer training. Peak VRAM: ~12 GB on L4.

### Launderer Validation Gate (Anti-Gaming)

The Launderer cannot generate trivial or malformed scenarios. A 9-check validation gate (`server/launderer_env.py:validate_scenario`) enforces:

1. All 7 top-level keys present (alert, profiles, transactions, watchlist, network, source, ground_truth)
2. Ground truth has 5 required fields
3. `is_suspicious` must be True (Launderer generates attacks, not clean data)
4. `correct_decision` must be `file_sar`
5. Typology must be one of the 6 valid `TypologyEnum` values
6. Non-empty key entities and findings
7. Alert must contain an `alert_id`
8. At least one transaction
9. At least one customer profile

Scenarios failing any check receive **-1.0** reward (schema fail) or **-2.0** (no JSON at all). Valid scenarios earn +0.1 to +1.0 proportional to Defender failure.

---

## The Gym vs. The Stage

**Training (The Gym):** `procedural_generator.py` creates infinite, unique POMDP scenarios. Entity IDs are randomized per episode — no memorization possible. The Launderer learns to generate adversarial scenarios; the Defender learns transferable OS mechanics.

**Presentation (The Stage):** `demo_eval.py` runs a hardcoded scenario inspired by the **1MDB scandal** — $681M in layered wire transfers through shell companies. Judges see the agent managing RAM, firing async traces, and injecting compliance rules while solving a real-world financial crime.

---

## Training Results

> 🔄 **Training is currently in progress.** WandB plots will be embedded here once the self-play run completes. The metrics below define what we track.

**Key WandB Metrics:**

| Metric | Expected Trend | What It Shows |
|--------|---------------|---------------|
| `ppo/returns/mean` | 0.0 → +0.8 | Main reward signal (agent improving) |
| `os/page_faults` | Decreasing → 0 | Agent learning memory management |
| `os/async_timeouts` | Decreasing → 0 | Agent learning to wait for async I/O |
| `os/successful_pages` | Increasing | Agent proactively writing to disk |
| `os/meta_injections` | ≥ 1 per episode | Agent injecting compliance rules |
| `defender/entity_f1` | Increasing → 0.8+ | Entity identification accuracy |

**Before vs. After (Qualitative):**

| Behavior | Untrained Agent | Trained Agent |
|----------|----------------|---------------|
| Memory management | References evicted data → page faults | Writes critical entities to disk before eviction |
| Async handling | Retrieves results prematurely → timeouts | Interleaves other tools while waiting for wire traces |
| Kernel updates | Ignores compliance rules | Searches compliance manual, injects relevant mode |
| Investigation strategy | Random tool calls, misses typology | Follows typology-specific tool sequences |
| Terminal decision | Always files SAR (lazy policy) | Correctly closes clean alerts (TN), files SAR on suspicious (TP) |

---

## Project Structure

```
.
├── openenv_server.py            # ★ OpenEnv FastAPI entrypoint (create_app + fallback)
├── models.py                    # Pydantic types (AMLState, AMLAction, AMLObservation, AGUIState)
├── state_manager.py             # OS mechanics engine (RAM, Disk, Async Queue, Kernel)
├── client.py                    # HTTP client (18 tool wrappers)
├── inference.py                 # ReAct agent (OS-aware)
│
├── self_play.py                 # ★ Two-agent self-play orchestrator (Warmup → L → D × N)
├── train_defender_ppo.py        # Defender PPO: GAE, EMA baseline, batch norm, mixed scenarios
├── train_launderer_ppo.py       # Launderer PPO: single-step MDP, VRAM-safe Defender scoring
├── train_ppo.py                 # Standalone step-level PPO (L4, 8B, --use-plr)
├── train_ppo_70b.py             # Multi-GPU PPO (DeepSpeed ZeRO-3, A100, 70B — scalability)
├── train_dpo.py                 # DPO continuous learning (offline, from user corrections)
├── hotswap.py                   # Zero-downtime LoRA adapter swap
├── demo_eval.py                 # 1MDB demo + AGUI replay capture
├── eval_harness.py              # Evaluation harness for checkpoint benchmarking
│
├── Dockerfile                   # HF Spaces deployment (port 7860)
├── openenv.yaml                 # OpenEnv contract → openenv_server:app
├── requirements.txt             # Runtime deps (fastapi, pydantic, httpx, openai, openenv-core)
├── .hfignore                    # Exclude patterns for openenv push
├── validate-submission.sh       # OpenEnv submission validator
├── validate.sh                  # Local validation script
│
├── curriculum/
│   ├── plr_engine.py            # ★ Prioritized Level Replay engine
│   └── oracle.py                # Proxy regret (1.0 - score)
├── scenarios/
│   ├── procedural_generator.py  # POMDP graph builder (3 typologies × 3 difficulties)
│   ├── adversary_agent.py       # Local Llama-3.1-8B evasive scenario generator
│   ├── compliance_manual.py     # Searchable AML rule corpus
│   └── base.py                  # Scenario ABC
├── graders/
│   └── grader.py                # Dense reward engine (per-step + terminal + investigation bonuses)
├── server/
│   ├── aml_environment.py       # Core env (18 tools + OS mechanics + scenario injection)
│   ├── launderer_env.py         # One-step MDP for Launderer (JSON validation + scoring)
│   └── app.py                   # Legacy FastAPI server (used by trainers)
├── frontend/
│   ├── components/case/
│   │   ├── CaseTerminal.tsx     # Main investigation terminal
│   │   ├── CurriculumPanel.tsx  # ★ 5th AGUI panel (PLR metrics)
│   │   └── case.module.css      # Brutalist design system
│   ├── prisma/schema.prisma     # DPO preference pair database
│   └── app/api/preferences/     # Correction capture API
└── tests/
    ├── test_smoke.py            # 8 end-to-end tests
    └── test_plr.py              # PLR engine unit tests
```

---

## Deployment

```bash
# Docker (local)
docker build -t memex .
docker run -p 7860:7860 memex

# HF Spaces (one command)
openenv push --ignore-file .hfignore
# → https://huggingface.co/spaces/MuazTPM/aml_investigation_env

# OpenEnv CLI
openenv serve
```

---

## Further Reading

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — Full architecture, AGUI data contract, VRAM calculations
- [TRAINING.md](TRAINING.md) — Copy-paste Colab cells, PPO stability engineering, self-play CLI reference, DPO pipeline

---

## License

MIT