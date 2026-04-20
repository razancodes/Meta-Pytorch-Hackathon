---
title: MetaHack
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# Memex: The OS-Agent Benchmark

An [OpenEnv](https://github.com/openenv/openenv)-compatible reinforcement learning environment for Anti-Money Laundering (AML) compliance investigation — built for the **Meta / Hugging Face OpenEnv Hackathon**.

Memex tests whether an LLM can function as a **Turing-complete Operating System** over long-horizon tasks: managing a finite context window (Virtual Memory), handling asynchronous background tasks (Interrupts), and self-improving its own decision rules (Kernel Updates) — all while solving complex financial crimes.

---

## 🎯 Why This Matters

Most LLM benchmarks evaluate models on static QA or single-turn generation. **Memex** tests something fundamentally harder: **multi-step reasoning under OS-level resource constraints** — the exact capability gap that separates a language model from a production-grade autonomous agent.

| Capability Tested | Why It's Hard |
|:---|:---|
| **Virtual Memory Management** | The agent's context window holds only the last 2 observations. Older data is evicted. The agent must `write_to_case_file` to page critical findings to disk before eviction. |
| **Interrupt Handling** | Background wire traces take 2–4 steps. The agent must interleave other investigation work while waiting, then retrieve results at the right time. |
| **Self-Improvement** | The agent starts with basic directives. It must search a compliance manual and inject relevant rules into its own system prompt (kernel update). |
| **Sequential Decision-Making** | A 9-phase investigation protocol — each step's output informs the next. No single "correct prompt." |
| **Contextual Tool Selection** | 15 tools with overlapping use cases. The agent must maximize information gain per step under a 25-step budget. |
| **Calibrated Terminal Judgment** | An irreversible, graded decision (file SAR vs. close alert). Premature or poorly justified decisions are heavily penalized. |

Real-world AML compliance costs global financial institutions **over $274 billion annually** (LexisNexis 2023). This environment provides a repeatable, deterministic benchmark to measure agent reasoning under realistic constraints.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INFERENCE LOOP                          │
│  ┌──────────┐    ┌────────────┐    ┌─────────────────────┐  │
│  │ LLM Call │───▶│ JSON Parse │───▶│ POST /step          │  │
│  │ (ReAct)  │    │ tool+params│    │ {tool, parameters}  │  │
│  └────▲─────┘    └────────────┘    └──────────┬──────────┘  │
│       │                                       │             │
│       │   ┌──────────────────────────────┐    │             │
│       └───│ Observation + Reward + AGUI  │◀───┘             │
│           └──────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ENVIRONMENT SERVER (FastAPI)                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               AMLEnvironment + StateManager           │  │
│  │                                                       │  │
│  │  15 Tool Handlers ──▶ Procedural Scenario Data        │  │
│  │  OS Mechanics:                                        │  │
│  │    • RAM (2-slot context) + Disk (persistent pages)   │  │
│  │    • Async Queue (wire traces with ETAs)              │  │
│  │    • Kernel (mutable system prompt directives)        │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │          AMLGrader (Dense Rewards)              │  │  │
│  │  │                                                 │  │  │
│  │  │  Per-Step: cost, redundancy, page faults,       │  │  │
│  │  │           async timeouts, disk writes, kernel   │  │  │
│  │  │  Terminal: decision + typology + findings F1    │  │  │
│  │  │           + entity F1 + efficiency → [-1, +1]   │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Procedural Scenario Generator                 │  │
│  │                                                       │  │
│  │  3 Typologies × 3 Difficulties = 9 combos            │  │
│  │  Unique entity IDs per episode (anti-memorization)    │  │
│  │  Noise injection scales with difficulty               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Procedural generation** — every `reset()` creates a fresh scenario with unique entity IDs, preventing memorization during RL training.
- **Dense reward signal** — per-step micro-rewards for OS mechanic usage (disk writes, kernel updates) plus terminal composite score for investigation quality.
- **Deterministic grading** — no LLM-as-judge. Scores use exact-match checks, keyword-overlap with semantic aliases, and precision/recall F1.
- **Dual-mode server** — uses `openenv-core` when available, gracefully degrades to standalone FastAPI.

---

## Project Structure

```
.
├── models.py                    # Single source of truth for all Pydantic types
├── state_manager.py             # OS mechanics: RAM, Disk, Async Queue, Kernel
├── client.py                    # HTTP client with all 15 tool wrappers
├── inference.py                 # ReAct agent inference loop (OS-aware)
├── train_ppo.py                 # Custom PPO trainer (T4-optimized, Unsloth + LoRA)
├── demo_eval.py                 # 1MDB demo with AGUI replay capture
├── openenv.yaml                 # OpenEnv spec
├── Dockerfile                   # Container for HF Spaces deployment
├── scenarios/
│   ├── __init__.py              # get_scenario() registry
│   ├── base.py                  # BaseScenario ABC
│   ├── procedural_generator.py  # Dynamic POMDP graph builder
│   └── compliance_manual.py     # Searchable AML rule corpus
├── graders/
│   ├── __init__.py
│   └── grader.py                # Dense reward: per-step + terminal scoring
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI server (OpenEnv-compatible)
│   └── aml_environment.py       # Core environment (15 tools + OS mechanics)
└── tests/
    └── test_smoke.py            # 7 end-to-end smoke tests
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Verify

```bash
curl http://localhost:8000/health
# → {"status": "ok", "env": "aml_investigation_env"}
```

### 4. Run inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export AML_ENV_URL="http://localhost:8000"

python inference.py
```

### 5. Run smoke tests

```bash
python tests/test_smoke.py
# → 7/7 tests passed ✓
```

---

## Available Tools (15)

### Domain Investigation Tools (9)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `review_alert` | `alert_id` (optional) | Get full alert details |
| `get_customer_profile` | `customer_id` | Get KYC profile |
| `query_transactions` | `customer_id`, filters | Get transaction history |
| `check_watchlist` | `entity_name`, `list_type` | Screen against OFAC/PEP/UN |
| `trace_network` | `entity_id`, `depth` | Trace connected entities |
| `check_source_of_funds` | `transaction_id` | Verify source documentation |
| `check_market_price` | `commodity` | Compare invoiced vs market prices |
| `assess_risk` | `customer_id` | Get computed risk score |
| `file_sar` | `findings`, `typology`, `entities_involved` | **TERMINAL**: File a SAR |
| `close_alert` | `reason`, `findings` | **TERMINAL**: Close as false positive |

### OS-Mechanic Tools (6)

| Tool | Parameters | OS Concept | Reward |
|------|-----------|------------|--------|
| `write_to_case_file` | `content` | Virtual Memory → Disk Page | +0.10 |
| `request_wire_trace` | `entity_id` or `transaction_id` | Interrupt → Async Enqueue | — |
| `retrieve_async_result` | `job_id` | Interrupt → Result Fetch | -0.10 if premature |
| `search_compliance_manual` | `query`, `category`, `max_results` | Kernel → Rule Lookup | — |
| `update_system_prompt` | `rule` | Kernel → Meta-Injection | +0.15 |

---

## Reward System

### Per-Step Rewards
| Event | Reward |
|-------|--------|
| Action cost (every step) | -0.02 |
| Redundant call (same tool+params) | -0.03 |
| Unique tool call | +0.03 |
| Page fault (reference evicted data not on disk) | -0.05 |
| Async timeout (premature retrieval) | -0.10 |
| Successful disk write | +0.10 |
| Kernel meta-injection | +0.15 |

### Terminal Score [-1.0, +1.0]
| Component | Weight |
|-----------|--------|
| Decision correctness (file_sar / close_alert) | 0.30 |
| Typology identification | 0.15 |
| Key findings coverage (semantic matching) | 0.25 |
| Entity precision/recall (F1) | 0.15 |
| Step efficiency (vs optimal path) | 0.15 |

---

## Scenario Generation

The **procedural generator** creates a fresh POMDP graph on every `reset()`:

| Typology | What It Generates |
|----------|-------------------|
| **Structuring** | Multiple sub-threshold cash deposits ($9K–$9.9K), no cash-intensive occupation |
| **Layering** | Fan-out through shell companies, shared addresses, PEP connections, rapid transfer chains |
| **Trade-Based ML** | Over/under-invoiced trade transactions, market price aberrations, related-party beneficial ownership |

Each difficulty level (easy/medium/hard) scales the number of decoy entities, noise transactions, and network complexity. Entity IDs are unique per episode to prevent memorization during RL training.

---

## Docker

```bash
docker build -t memex-env .
docker run -p 8000:8000 memex-env
```

---

## Python Client

```python
from client import AMLEnvironmentClient

with AMLEnvironmentClient() as c:
    obs = c.reset(task_id="easy")
    obs = c.review_alert()
    obs = c.get_customer_profile("CUST001")
    obs = c.write_to_case_file("PEP confirmed for CUST001")
    obs = c.search_compliance_manual("structuring threshold")
    obs = c.update_system_prompt("CTR threshold is $10,000")
    obs = c.query_transactions("CUST001")
    obs = c.file_sar(
        findings=["multiple_sub_threshold_deposits", "no_source_documentation"],
        typology="structuring",
        entities_involved=["CUST001"],
    )
    print(obs)
```

---

## OpenEnv Spec

```yaml
spec_version: 1
name: aml_investigation_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## License

MIT