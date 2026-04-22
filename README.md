---
title: MetaHack
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
---

<div align="center">

# 🧠 Memex: The OS-Agent Benchmark

### *Can an LLM run its own Operating System?*

**A POMDP environment where language models manage Virtual Memory, handle Interrupts, and self-update their Kernel — all while solving $274B-scale financial crimes.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv/openenv)
[![Smoke Tests](https://img.shields.io/badge/tests-7%2F7-brightgreen)]()
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
│  ┌───────────────┬──────────────┬──────────────┬─────────────────┐  │
│  │  RAM Monitor  │ Disk Storage │   Active     │    Kernel       │  │
│  │  2/2 █████    │ findings...  │  Processes   │   Directives    │  │
│  │  (evicts old) │ (persistent) │  REQ-001 ✓   │ [BASE] + [INJ]  │  │
│  └───────────────┴──────────────┴──────────────┴─────────────────┘  │
│                     ← agui_state JSON per step                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT SERVER (FastAPI)                       │
│                                                                      │
│  AMLEnvironment  ────────────  StateManager                          │
│  15 Tool Handlers              • RAM (2 slots) + Disk               │
│  Action Routing                • Async Queue (ETAs)                 │
│  Scenario Data                 • Kernel Directives                  │
│       │                              │                               │
│       └──────────  AMLGrader  ◄──────┘                               │
│                    Per-step: -0.05 PF, +0.10 disk, +0.15 kernel     │
│                    Terminal:  [-1.0, +1.0] composite score           │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Procedural Generator: 3 typologies × 3 difficulties         │   │
│  │  Unique entity IDs per episode → anti-memorization           │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                     TRAINING INFRASTRUCTURE                          │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────┐     │
│  │  train_ppo.py      │    │  train_ppo_70b.py                │     │
│  │  T4 (15GB VRAM)    │    │  A100 × 4-8 (DeepSpeed ZeRO-3)  │     │
│  │  8B + 4-bit + LoRA │    │  70B + 4-bit + LoRA (r=32)      │     │
│  │  disable_adapter() │    │  disable_adapter() + sharding    │     │
│  │  Peak: ~10 GB      │    │  Peak: ~50 GB/GPU               │     │
│  └────────────────────┘    └──────────────────────────────────┘     │
│                                                                      │
│  Step-Level PPO: every tool call gets a reward signal               │
│  KL via LoRA toggle: no second model copy needed                    │
│  Auto-Revert: entropy heartbeat + checkpoint time machine           │
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

# Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/health
# → {"status": "ok", "env": "aml_investigation_env"}

# Run smoke tests
python tests/test_smoke.py
# → 7/7 tests passed ✓
```

### Run Inference (any OpenAI-compatible LLM)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### Train (T4 / Colab / Kaggle)

```bash
pip install unsloth trl peft wandb

# Dry-run (2 iterations, no GPU needed for env)
python train_ppo.py --dry-run

# Real training (~2.5 hours on T4)
python train_ppo.py --iterations 50 --episodes 4
```

### Train (70B / A100 Cluster)

```bash
pip install deepspeed unsloth trl peft wandb

# 4× A100-80GB
deepspeed --num_gpus 4 train_ppo_70b.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --iterations 50 --episodes 2

# With CPU offloading (2× A100)
deepspeed --num_gpus 2 train_ppo_70b.py --offload-optimizer --offload-params
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

## Tool Roster (15 Tools)

| Domain Tools (9) | OS-Mechanic Tools (6) |
|:---|:---|
| `review_alert` — Alert details | `write_to_case_file` — Page to disk (+0.10) |
| `get_customer_profile` — KYC data | `request_wire_trace` — Async job (2-4 step ETA) |
| `query_transactions` — Transaction history | `retrieve_async_result` — Fetch completed job |
| `check_watchlist` — OFAC/PEP/UN screening | `search_compliance_manual` — Find AML rules |
| `trace_network` — Entity connections | `update_system_prompt` — Kernel inject (+0.15) |
| `check_source_of_funds` — Source verification | |
| `check_market_price` — Trade price comparison | |
| `assess_risk` — Risk scoring | |
| `file_sar` / `close_alert` — **Terminal** | |

---

## Reward System

**Per-Step** (dense signal at every tool call):

| Event | Reward | OS Concept |
|-------|--------|-----------|
| Action cost | -0.02 | — |
| Redundant call | -0.03 | — |
| Unique tool | +0.03 | — |
| Page fault | -0.05 | Virtual Memory |
| Async timeout | -0.10 | Interrupts |
| Disk write | +0.10 | Virtual Memory |
| Kernel injection | +0.15 | Kernel Update |

**Terminal** (composite score → [-1.0, +1.0]):

| Component | Weight | Method |
|-----------|--------|--------|
| Decision | 0.30 | Exact match: file_sar vs close_alert |
| Typology | 0.15 | Exact match: structuring/layering/TBML |
| Findings | 0.25 | Keyword overlap with semantic aliases |
| Entities | 0.15 | Precision/Recall F1 |
| Efficiency | 0.15 | Steps used vs optimal path |

---

## The Gym vs. The Stage

**Training (The Gym):** `procedural_generator.py` creates infinite, unique POMDP scenarios. Entity IDs are randomized per episode — no memorization possible. The agent must learn transferable OS mechanics.

**Presentation (The Stage):** `demo_eval.py` runs a hardcoded scenario inspired by the **1MDB scandal** — $681M in layered wire transfers through shell companies. Judges see the agent managing RAM, firing async traces, and injecting compliance rules while solving a real-world financial crime.

---

## Project Structure

```
.
├── models.py                    # Pydantic types (single source of truth)
├── state_manager.py             # OS mechanics engine
├── client.py                    # HTTP client (15 tool wrappers)
├── inference.py                 # ReAct agent (OS-aware)
├── train_ppo.py                 # PPO trainer (T4, 8B)
├── train_ppo_70b.py             # PPO trainer (A100 cluster, 70B)
├── train_dpo.py                 # DPO continuous learning (offline)
├── hotswap.py                   # Zero-downtime LoRA adapter swap
├── demo_eval.py                 # 1MDB demo + AGUI replay
├── Dockerfile                   # HF Spaces deployment
├── openenv.yaml                 # OpenEnv contract
├── scenarios/
│   ├── procedural_generator.py  # POMDP graph builder
│   ├── compliance_manual.py     # Searchable rule corpus
│   └── base.py                  # Scenario ABC
├── graders/
│   └── grader.py                # Dense reward engine
├── server/
│   ├── app.py                   # FastAPI (OpenEnv-compatible)
│   └── aml_environment.py       # Core env (15 tools + OS mechanics)
├── frontend/
│   ├── prisma/schema.prisma     # DPO preference pair database
│   └── app/api/preferences/     # Correction capture API
└── tests/
    └── test_smoke.py            # 7 end-to-end tests
```

---

## Docker

```bash
docker build -t memex .
docker run -p 8000:8000 memex
```

---

## Further Reading

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — Full architecture, AGUI data contract, VRAM calculations, 70B scaling analysis
- [TRAINING.md](TRAINING.md) — Copy-paste Colab/Kaggle cells, stability engineering features, DPO pipeline setup

---

## License

MIT