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
│               SELF-PLAY TRAINING (Two-Agent PPO)                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  self_play.py — Alternating Best-Response Orchestrator      │     │
│  │                                                             │     │
│  │  Phase 1: Defender Warm-Start (procedural scenarios)        │     │
│  │  ┌── Outer Round 1..N:                                      │     │
│  │  │  Phase 2: Launderer PPO vs frozen Defender               │     │
│  │  │  Phase 3: Defender PPO on mixed scenarios                │     │
│  │  └── (mix ratio: 0.3 → 0.7 linear)                         │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────────────┐       │
│  │ train_defender_ppo.py│    │ train_launderer_ppo.py       │       │
│  │ Defender-8B + LoRA   │    │ Launderer-8B + LoRA          │       │
│  │ GAE (γ=0.99, λ=0.95)│ ◄──┤ Single-step MDP              │       │
│  │ Mixed scenarios      │    │ VRAM-safe model swap          │       │
│  │ Entity F1 tracking   │    │ Frozen Defender scoring       │       │
│  └──────────────────────┘    └──────────────────────────────┘       │
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────────────┐       │
│  │ train_ppo.py         │    │ train_grpo.py (EXPERIMENTAL) │       │
│  │ Standalone PPO + PLR │    │ No clipping, |KL|            │       │
│  │ L4 (24GB) / T4 (15GB)│    │ Ablation only                │       │
│  └──────────────────────┘    └──────────────────────────────┘       │
│                                                                      │
│  PLR Curriculum: regret-weighted scenario replay (curriculum/)      │
│  Auto-Revert: entropy heartbeat + checkpoint time machine           │
│  VRAM-safe: only one model loaded at a time (full unload/reload)    │
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

### Train — Self-Play (L4 / Colab Pro)

The **production training path** is two-agent self-play: a Defender learns to investigate AML cases while a Launderer learns to generate evasive scenarios.

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

Or train individual agents:

```bash
# Defender only (procedural scenarios)
python train_defender_ppo.py --scenario-source procedural --iterations 50

# Standalone PPO with PLR curriculum (~2.5 hours)
python train_ppo.py --iterations 50 --episodes 4 --use-plr
```

### Train (70B / A100 Cluster — Proof of Scalability)

```bash
pip install deepspeed unsloth trl peft wandb

# 4× A100-80GB
deepspeed --num_gpus 4 train_ppo_70b.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --iterations 50 --episodes 2
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
| Page fault | -0.05 | Virtual Memory |
| Async timeout | -0.10 | Interrupts |
| Disk write | +0.10 | Virtual Memory (Hard cap: 3/episode) |
| Kernel injection | +0.15 | Kernel Update (Hard cap: 2/episode) |

*Note: Reward farming is prevented via hard caps. After the cap is reached, the agent no longer receives the bonus but still pays the -0.02 action cost.*

**Terminal** (composite score → [-1.0, +1.0]):

| Component | Weight | Method |
|-----------|--------|--------|
| Decision | 0.30 | Exact match: file_sar vs close_alert |
| Typology | 0.15 | Exact match: structuring/layering/TBML |
| Findings | 0.20 | Keyword overlap with semantic aliases |
| Entities | 0.15 | Precision/Recall F1 |
| UBO ID | 0.05 | Exact match: beneficial owner ID |
| Pillar | 0.05 | Phase 3 checks coverage |
| Efficiency | 0.10 | Steps used vs optimal path |

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

---

## The Gym vs. The Stage

**Training (The Gym):** `procedural_generator.py` creates infinite, unique POMDP scenarios. Entity IDs are randomized per episode — no memorization possible. The Launderer learns to generate adversarial scenarios; the Defender learns transferable OS mechanics.

**Presentation (The Stage):** `demo_eval.py` runs a hardcoded scenario inspired by the **1MDB scandal** — $681M in layered wire transfers through shell companies. Judges see the agent managing RAM, firing async traces, and injecting compliance rules while solving a real-world financial crime.

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
├── train_defender_ppo.py        # Defender PPO: GAE, mixed scenarios, entity-F1/typology tracking
├── train_launderer_ppo.py       # Launderer PPO: single-step MDP, VRAM-safe Defender scoring
├── train_ppo.py                 # Standalone step-level PPO (L4, 8B, --use-plr)
├── train_grpo.py                # GRPO (EXPERIMENTAL — ablation only, --experimental)
├── train_ppo_70b.py             # Multi-GPU PPO (DeepSpeed ZeRO-3, A100, 70B)
├── train_dpo.py                 # DPO continuous learning (offline, from user corrections)
├── train_adversary.py           # Heuristic adversarial loop (DEPRECATED — use self_play.py)
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
│   └── grader.py                # Dense reward engine (per-step + terminal composite)
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

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — Full architecture, AGUI data contract, VRAM calculations, 70B scaling analysis
- [TRAINING.md](TRAINING.md) — Copy-paste Colab cells, PPO stability engineering, self-play CLI reference, DPO pipeline

---

## License

MIT