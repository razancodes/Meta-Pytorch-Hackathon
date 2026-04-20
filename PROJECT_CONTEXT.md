# Memex: Project Context

> Living document tracking the current state of the Memex OS-Agent Benchmark.
> Last updated: 2026-04-20

## What Is Memex?

Memex is a **B2B enterprise benchmarking environment** built for the Meta / Hugging Face OpenEnv Hackathon. It tests whether an LLM can function as a Turing-complete Operating System over long-horizon tasks, using complex Anti-Money Laundering (AML) investigations as the obstacle course.

The "OS" metaphor is an **architectural framework**, not a literal operating system. It solves the LLM context window bottleneck by giving the agent explicit tools to manage its own memory, handle async operations, and self-improve its decision rules.

---

## Current Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Models** | `models.py` | Single source of truth for all Pydantic types: `AMLAction`, `AMLObservation`, `AMLState`, `AsyncJobInfo`, `AGUIState` |
| **State Manager** | `state_manager.py` | OS mechanics engine: RAM eviction (2-slot context), Disk persistence, Async Job Queue, Kernel Directives |
| **Procedural Generator** | `scenarios/procedural_generator.py` | Dynamic POMDP scenario builder: 3 typologies × 3 difficulties, unique IDs per episode |
| **Compliance Manual** | `scenarios/compliance_manual.py` | Searchable AML rule corpus for kernel updates |
| **Environment** | `server/aml_environment.py` | Core environment: 15 tool handlers + State Manager integration |
| **Grader** | `graders/grader.py` | Dense reward: per-step micro-rewards + terminal composite score [-1.0, +1.0] |
| **Server** | `server/app.py` | FastAPI HTTP server (dual-mode: OpenEnv / standalone) |
| **Client** | `client.py` | HTTP client with all 15 tool wrappers |
| **Inference** | `inference.py` | ReAct agent loop with OS-mechanic awareness |
| **PPO Trainer** | `train_ppo.py` | Custom step-level PPO (Unsloth 4-bit + LoRA, T4-optimized) |
| **Demo** | `demo_eval.py` | 1MDB-inspired demo with AGUI replay capture |
| **Tests** | `tests/test_smoke.py` | 7 end-to-end smoke tests |

### OS Mechanics

1. **Virtual Memory (RAM Eviction)**
   - Agent context window = last 2 observations only
   - `write_to_case_file(content)` → pages data to persistent disk (+0.10 reward)
   - Page Fault: referencing evicted data NOT on disk → -0.05 penalty

2. **Interrupts (Async Queue)**
   - `request_wire_trace(entity_id)` → background job with 2–4 step ETA
   - `retrieve_async_result(job_id)` → fetch when ETA=0
   - Async Timeout: premature retrieval → -0.10 penalty

3. **Kernel Updates (Self-Improvement)**
   - `search_compliance_manual(query)` → find AML rules
   - `update_system_prompt(rule)` → inject into active kernel directives (+0.15)

### Tool Roster (15 total)

**Domain (9):** review_alert, get_customer_profile, query_transactions, check_watchlist, trace_network, check_source_of_funds, check_market_price, assess_risk, file_sar, close_alert

**OS-Mechanic (6):** write_to_case_file, request_wire_trace, retrieve_async_result, search_compliance_manual, update_system_prompt

### Reward System

**Per-step:** action cost (-0.02), redundancy (-0.03), unique tool (+0.03), page fault (-0.05), async timeout (-0.10), disk write (+0.10), kernel inject (+0.15)

**Terminal:** decision accuracy (0.30) + typology (0.15) + findings coverage (0.25) + entity F1 (0.15) + efficiency (0.15) → mapped to [-1.0, +1.0]

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
```

**All 7/7 smoke tests pass.**

---

## Dependencies

- **Runtime:** Python 3.10+, FastAPI, Pydantic v2, httpx, openai
- **Training (optional):** unsloth, trl, peft, torch, wandb
- **Validation:** openenv-core, Docker

---

## Deployment

- **HF Spaces:** Docker-based, auto-deploys via `Dockerfile` in root
- **Local:** `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- **Validation:** `bash validate.sh` (3 checks: ping, Docker build, openenv validate)

---

## Recent Changes (2026-04-20)

1. **Codebase sanitization:** Removed duplicate type definitions from `app.py`, updated `client.py` with all 15 tools, rewrote README to reflect Memex OS-Agent architecture
2. **Git consolidation:** Resolved stuck rebase, recovered all OS-mechanic features to `main`
3. **PPO trainer:** Custom step-level PPO with Unsloth 4-bit + LoRA, T4-optimized (peak ~10GB VRAM)
4. **Demo system:** 1MDB-inspired scenario with AGUI replay capture for frontend visualization
