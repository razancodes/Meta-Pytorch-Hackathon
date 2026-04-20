# AML Investigation Environment — Complete Project Context

> **Purpose of this document:** Authoritative reference for any new agent or developer picking up this project. Read this before touching a single file.

---

## 1. What This Project Is

This repository implements the **AML Investigation Environment** — a stateful, episodic **reinforcement learning benchmark environment** for Anti-Money Laundering (AML) financial crime investigation.

It was built as a submission to the **Meta PyTorch Hackathon** (also internally called "MetaHack"), submitted to **HuggingFace Spaces** under the `MetaHack` Space. The environment is also designed to be **OpenEnv-compatible**, meaning it follows the [OpenEnv](https://github.com/openenv/openenv) open standard for agent benchmark environments.

**The core idea:** an AI agent plays the role of a *Senior Financial Crime Compliance Investigator*. It is handed a transaction monitoring alert, must use structured investigation tools to gather evidence across multiple steps, and ultimately makes an irreversible terminal decision: either **file a Suspicious Activity Report (SAR)** or **close the alert as a false positive**. The agent is scored not just on whether it makes the right call, but on the *quality and efficiency* of the entire investigation.

This is **not** a gravitational lensing / physics project despite the repository name `AML-Meta-Pytorch`. The repository was repurposed from an earlier ML research project; the current code is entirely focused on AML compliance RL.

---

## 2. Why It Exists (Motivation)

Most LLM benchmarks test static question-answering. This environment tests something qualitatively harder:

| Capability | Why It's Hard |
|---|---|
| **Sequential Decision-Making** | Multi-phase investigation — triage → due diligence → network analysis → determination. No single correct prompt. |
| **Contextual Tool Selection** | 9 domain-specific tools with overlapping use cases. Agent must pick the highest-information-gain tool at each step. |
| **Cross-Reference Reasoning** | Evidence is scattered across tool outputs (e.g., connecting a beneficial owner to a PEP status found via watchlist). |
| **Calibrated Terminal Judgment** | The agent makes an irreversible, graded binary decision. Premature decisions are penalized. |
| **Efficiency Under Budget** | MAX_STEPS = 25. Redundant tool calls cost reward. Agent must maximize evidence per step. |

Real-world AML compliance costs financial institutions **$274 billion annually** (LexisNexis 2023). A benchmark that accurately measures agent investigative ability has significant commercial and societal value.

---

## 3. Repository Structure

```
AML-Meta-Pytorch/
│
├── models.py                  # Pydantic data models (AMLAction, AMLObservation, AMLState)
├── client.py                  # Python HTTP client wrapper (AMLEnvironmentClient)
├── inference.py               # Baseline LLM agent inference script (ReAct loop)
├── __init__.py                # Package init
│
├── openenv.yaml               # OpenEnv spec declaration (spec_version: 1)
├── pyproject.toml             # Build system config (hatchling), project metadata
├── requirements.txt           # Minimal pip dependencies
├── Dockerfile                 # Docker image definition (python:3.11-slim, port 8000)
├── validate.sh                # Local validation script
├── validate-submission.sh     # Full OpenEnv submission validation script
├── server.log                 # Last server run log (development artifact)
│
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI HTTP server (OpenEnv-compatible endpoints)
│   └── aml_environment.py     # AMLEnvironment class — core RL environment logic
│
├── scenarios/
│   ├── __init__.py            # get_scenario(task_id) factory function
│   ├── base.py                # BaseScenario ABC (abstract base class)
│   ├── easy.py                # Task 1: Structuring Detection (CUST001 / John Doe)
│   ├── medium.py              # Task 2: Layering Through Shell Companies (CUST002 / GlobalTrade LLC)
│   └── hard.py                # Task 3: Trade-Based Money Laundering (CUST003 / NovaTech Industries)
│
└── graders/
    ├── __init__.py
    └── grader.py              # AMLGrader — deterministic scoring engine
```

---

## 4. Technology Stack & Environment

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | FastAPI |
| ASGI Server | Uvicorn |
| Data Validation | Pydantic v2 |
| HTTP Client | httpx |
| LLM Client | openai (SDK, v1+) |
| Build System | Hatchling |
| Packaging | `pyproject.toml` / `uv.lock` (uv lockfile present) |
| Containerization | Docker (python:3.11-slim base) |
| Deployment Target | HuggingFace Spaces (Docker SDK, port 8000) |
| OpenEnv Integration | `openenv-core` (optional; gracefully degrades to standalone FastAPI) |

**Key dependency note:** `openenv-core` is listed both in `requirements.txt` and `pyproject.toml`. The code uses a dual-import pattern throughout — it tries `from openenv.core import ...` and falls back to plain Python ABCs/FastAPI if not installed. This means the environment works in two modes:
1. **Full OpenEnv mode**: When `openenv-core` is installed, `create_app()` from `openenv.core.env_server.http_server` is used to build the FastAPI app.
2. **Standalone mode**: Manual FastAPI app construction with identical endpoint signatures — used for local development and HuggingFace Spaces.

---

## 5. Architecture Deep Dive

### 5.1 The RL Loop

```
Agent (LLM via ReAct)
    │
    │  POST /reset  {"task_id": "easy"|"medium"|"hard"}
    ▼
Environment Server (FastAPI)
    │  Returns: initial_alert + available_tools + episode_id
    │
    │  POST /step  {"action": {"tool": "...", "parameters": {...}}}
    ▼
AMLEnvironment.step()
    ├── Guard checks (reset?, already done?, max steps exceeded?)
    ├── Redundancy detection (MD5 hash of tool+params vs history)
    ├── Step reward: +0.05 (unique call) or -0.02 (redundant call)
    ├── Tool dispatch → one of 9 handler methods
    │       └── Reads from current Scenario's data layer
    │           Updates AMLState evidence flags
    ├── If terminal (file_sar / close_alert):
    │       └── AMLGrader.grade() → composite final score [0.001, 0.999]
    └── Returns AMLObservation (tool_result, available_tools, message, reward, done)
```

### 5.2 AMLEnvironment (`server/aml_environment.py`)

The core environment class. Key attributes:
- `MAX_STEPS = 25` — hard budget per episode
- `_current_scenario` — the loaded scenario object (EasyScenario / MediumScenario / HardScenario)
- `_state: AMLState` — tracks all evidence flags and accumulated reward

**9 tool handlers:**
1. `_handle_review_alert` — Returns full alert dict; sets `state.alert_reviewed = True`
2. `_handle_get_customer_profile` — Returns KYC profile by `customer_id` (exact + fuzzy match); sets `state.customer_profiled = True`
3. `_handle_query_transactions` — Filters transactions by `customer_id`, date range, amount; sets `state.transactions_queried = True`
4. `_handle_check_watchlist` — Screens entity against OFAC/PEP/UN lists; appends to `state.watchlist_checked`
5. `_handle_trace_network` — Returns entity relationship graph at depth 1 or 2; sets `state.network_traced = True`
6. `_handle_check_source_of_funds` — Returns documentation status for a transaction ID; appends to `state.source_checked`
7. `_handle_assess_risk` — Computes a deterministic risk score from current `AMLState` flags; sets `state.risk_assessed = True`
8. `_handle_file_sar` (**TERMINAL**) — Invokes `AMLGrader.grade()`, marks `state.decision_made = True`, returns final score
9. `_handle_close_alert` (**TERMINAL**) — Same as above with `decision="close_alert"`

**Redundancy detection:** `_compute_hash(tool, params)` uses MD5 on the JSON-serialized `{"tool": ..., "params": ...}` to produce a stable hash. If this hash is in `state.tool_call_hashes`, the call is penalized.

### 5.3 AMLState (`models.py`)

Pydantic model tracking the full episode state:

```python
episode_id: str
step_count: int
task_id: str
alert_reviewed: bool
customer_profiled: bool
transactions_queried: bool
watchlist_checked: List[str]   # entity names screened
network_traced: bool
source_checked: List[str]      # transaction IDs checked
risk_assessed: bool
decision_made: bool
findings: List[str]            # findings submitted at terminal action
accumulated_reward: float
tool_call_hashes: List[str]    # for redundancy detection
```

### 5.4 Scenario System (`scenarios/`)

Each scenario is a Python class inheriting from `BaseScenario` (ABC). Each scenario provides:
- `initial_alert` → dict shown to agent at episode start
- `customer_profiles` → dict mapping `customer_id` → KYC data
- `transactions` → list of transaction dicts
- `watchlist_results` → dict mapping entity name → watchlist screening result
- `network_graph` → dict mapping entity ID → connected entities
- `source_of_funds` → dict mapping transaction ID → source verification result
- `ground_truth` → dict with `correct_decision`, `typology`, `key_entities`, `key_findings`, optionally `excluded_entities`
- `market_data` (optional, used in Hard scenario) → commodity price reference data

The factory function `get_scenario(task_id)` in `scenarios/__init__.py` instantiates the right class.

### 5.5 AMLGrader (`graders/grader.py`)

The deterministic scoring engine. **No LLM-as-judge.** Two main methods:

**`grade_step(tool, params, state, call_hash)`** → per-step reward:
- `-0.02` for redundant calls
- `+0.05` for unique calls

**`grade(task_id, decision, findings, entities_flagged, typology, state)`** → final composite score in `[0.001, 0.999]`:

| Component | Weight | Mechanism |
|---|---|---|
| Decision correctness | 0.30 | Exact match: `decision == gt["correct_decision"]` |
| Typology correctness | 0.15 | Case-insensitive exact match |
| Key findings coverage | 0.25 | Proportional fuzzy match (3-tier strategy) |
| Entity F1 score | 0.15 | Precision/Recall F1, penalizes excluded entities |
| Step efficiency | 0.15 | Linear decay from optimal step count to MAX_STEPS |

**Optimal step counts per task:**
- Easy: 5 steps
- Medium: 8 steps
- Hard: 10 steps

**Findings fuzzy matching (3-tier):**
1. **Keyword overlap**: ≥50% of ground-truth finding keywords appear in any single agent finding string
2. **Semantic alias table**: Curated synonyms (e.g., `"no_source_documentation"` ↔ `["undocumented", "no_business_justification"]`)
3. **Substring fallback**: Any ground-truth keyword appears anywhere in the combined agent findings

---

## 6. The Three Investigation Scenarios

### Task 1: Structuring Detection (`task_id="easy"`)

**Customer:** John Doe (CUST001), individual, Retail Store Clerk, annual income $38K  
**Account:** Personal Checking, 3 years old, Medium KYC risk tier  
**Alert:** `ALERT-2024-0042` — 5 cash deposits totalling $47,900 over 5 days (Jan 3–7, 2024), all below $10K CTR threshold, at the same branch (Downtown Retail — Branch 12)

**Transaction breakdown:**
- TXN-001-A: $9,500 (Jan 3)
- TXN-001-B: $9,800 (Jan 4)
- TXN-001-C: $9,400 (Jan 5)
- TXN-001-D: $9,700 (Jan 6)
- TXN-001-E: $9,500 (Jan 7)

**Key red flags:**
- All deposits below $10K CTR reporting threshold (structuring / "smurfing")
- Same branch, consecutive days
- No cash-intensive occupation
- No source of funds documentation for any transaction
- Customer could not explain source when asked

**Ground truth:**
- Decision: `file_sar`
- Typology: `structuring`
- Key entities: `["CUST001"]`
- Key findings: `["multiple_sub_threshold_deposits", "no_cash_intensive_occupation", "same_branch_repeated", "no_source_documentation", "total_exceeds_ctr_threshold"]`

**Optimal path (5 steps):** review_alert → get_customer_profile → query_transactions → check_source_of_funds → file_sar

**Baseline score (gpt-4o-mini):** ~0.95, 100% success rate

---

### Task 2: Layering Through Shell Companies (`task_id="medium"`)

**Customer:** GlobalTrade LLC (CUST002), import/export business, 6 months old, Delaware USA, director Samuel Park  
**Alert:** `ALERT-2024-0187` — $500K inbound from Apex Holdings (BVI) → immediate fan-out to 3 entities within 24 hours. No trade documentation.

**Entity map:**
- **ENT_A** (Apex Holdings) — British Virgin Islands offshore holding, unknown beneficial ownership, source of the inbound $500K
- **ENT_B** (Bright Solutions Ltd) — Singapore shell, received $200K, registered at "45 Marina Bay, Singapore 018982", no public activity
- **ENT_C** (Crescent Ventures) — Singapore shell, received $150K, **same registered address as ENT_B**, no public activity
- **ENT_D** (Delta Resources) — Cyprus, received $150K, director **Viktor Korev** (former Deputy Finance Minister, PEP hit)
- **ENT_E** (EverGreen Supplies) — USA legitimate supplier, monthly $5K payments, **NOT suspicious** (red herring / excluded entity)

**Key red flags:**
- Rapid fan-out of entire inbound wire within 24 hours
- Offshore source (BVI) with unknown beneficial owner
- ENT_B and ENT_C share identical registered address (shell indicator)
- ENT_D director Viktor Korev is a PEP (watchlist hit — must check `check_watchlist("Viktor Korev")`)
- Newly incorporated company (6 months), no audited financials
- No trade documentation to justify any of the transfers

**Ground truth:**
- Decision: `file_sar`
- Typology: `layering`
- Key entities: `["CUST002", "ENT_A", "ENT_B", "ENT_C", "ENT_D"]`
- Excluded entities: `["ENT_E"]` (flagging ENT_E reduces entity F1 score)
- Key findings: `["rapid_fan_out", "pep_connection", "shared_registered_address", "offshore_source", "newly_incorporated", "no_trade_documentation"]`

**Optimal path (8 steps):** review_alert → get_customer_profile (CUST002) → query_transactions → trace_network (CUST002) → check_watchlist (Viktor Korev) → check_watchlist (Apex Holdings) → check_source_of_funds (TXN-002-A1) → file_sar

**Baseline score (gpt-4o-mini):** ~0.82, 85% success rate

---

### Task 3: Trade-Based Money Laundering (`task_id="hard"`)

**Customer:** NovaTech Industries (CUST003), electronics importer, California USA, director Alan Chen  
**Alert:** `ALERT-2024-0391` — 12 payments over 6 months to OceanPrime Exports (Myanmar, FATF-monitored), totalling $600K for "machine parts" at $50K/unit vs. $12K market price. Plus unexplained $200K inbound from Cayman Islands Investment Fund.

**Entity map:**
- **ENT_F** (OceanPrime Exports) — Myanmar (FATF-monitored jurisdiction), director Li Wei, **beneficial owner Marcus Webb** (brother-in-law of Alan Chen / NovaTech's director). Received $600K total.
- **ENT_G** (Cayman Islands Investment Fund) — Sent unexplained $200K to NovaTech, no agreement on file
- **ENT_H** (TechDirect Corp) — USA domestic supplier, legitimate pricing, long-term relationship, **red herring / excluded entity**

**Key red flags:**
- **Over-invoicing**: $50K/unit vs. $12K market price = 317% premium. Total overpayment = $456K across 12 invoices.
- **Beneficial owner family connection**: Marcus Webb (ENT_F owner) = brother-in-law of Alan Chen (CUST003 director). Only discovered via `trace_network("ENT_F", depth=2)`.
- **FATF jurisdiction**: Myanmar is an FATF-monitored (increased monitoring) jurisdiction.
- **Reversed transaction**: TXN-003-F11-REV reversed and re-sent as TXN-003-F11 at $49,750 (amount changed from $50K with no documented reason).
- **Unexplained inbound funds**: $200K from Cayman Islands Investment Fund (TXN-003-G1), no investment agreement on file.
- Invoices lack part numbers, model specs, or shipping manifests.

**Important mechanic (depth-2 network):** The family connection is *only* visible at `depth=2` in the network graph for ENT_F. At depth 1 you see Li Wei (director) and Marcus Webb (beneficial owner). At depth 2 you get the Alan Chen → Marcus Webb brother-in-law link, which is the smoking gun.

**Ground truth:**
- Decision: `file_sar`
- Typology: `trade_based_ml`
- Key entities: `["CUST003", "ENT_F", "Marcus Webb"]`
- Excluded entities: `["ENT_H"]` (TechDirect is legitimate)
- Key findings: `["over_invoicing", "beneficial_owner_connection", "fatf_jurisdiction", "reversed_transaction", "unexplained_funds"]`

**Optimal path (10 steps):** review_alert → get_customer_profile (CUST003) → get_customer_profile (ENT_F) → query_transactions → trace_network (ENT_F, depth=2) → check_watchlist (Marcus Webb) → check_watchlist (OceanPrime Exports) → check_source_of_funds (TXN-003-F01) → assess_risk → file_sar

**Baseline score (gpt-4o-mini):** ~0.65, 50% success rate (agents struggle with the market-price comparison and the depth-2 family connection)

---

## 7. The 9 Investigation Tools

All tools are accessible via HTTP `POST /step` with `{"action": {"tool": "<name>", "parameters": {...}}}`.

| Tool | Key Parameters | What It Returns | Non-Terminal? |
|---|---|---|---|
| `review_alert` | `alert_id` (optional) | Full alert dict | ✅ |
| `get_customer_profile` | `customer_id` (str) | KYC profile dict (exact + fuzzy match by name) | ✅ |
| `query_transactions` | `customer_id`, `date_from`, `date_to`, `min_amount`, `max_amount` | Filtered transaction list | ✅ |
| `check_watchlist` | `entity_name`, `list_type` | Watchlist hit/miss result | ✅ |
| `trace_network` | `entity_id`, `depth` (1 or 2) | Entity connection graph | ✅ |
| `check_source_of_funds` | `transaction_id` | Source documentation status | ✅ |
| `assess_risk` | `customer_id` | Dynamic risk score (20–100) based on state flags | ✅ |
| `file_sar` | `findings` (list), `typology` (str), `entities_involved` (list) | SAR confirmation + final score | ❌ TERMINAL |
| `close_alert` | `reason` (str), `findings` (list) | Alert closure confirmation + final score | ❌ TERMINAL |

**Important:** `file_sar` and `close_alert` immediately end the episode. The `done` flag becomes `True` and the `AMLGrader.grade()` result is returned.

---

## 8. HTTP API Reference

The server runs on port `8000`. All responses follow the `AMLObservation` structure.

### `GET /health`
```json
{"status": "ok", "env": "aml_investigation_env"}
```

### `POST /reset`
```json
// Request
{"task_id": "easy", "seed": null, "episode_id": null}

// Response
{
  "observation": {
    "tool_result": {"alert": {...}},
    "available_tools": ["review_alert", "get_customer_profile", ...],
    "message": "Episode started. Alert ALERT-2024-0042 assigned.",
    "metadata": {"episode_id": "...", "task_id": "easy", "step": 0}
  },
  "reward": null,
  "done": false
}
```

### `POST /step`
```json
// Request
{"action": {"tool": "review_alert", "parameters": {}, "metadata": {}}, "timeout_s": null}

// Response
{
  "observation": {
    "tool_result": {...},
    "available_tools": [...],
    "message": "...",
    "metadata": {"step": 1}
  },
  "reward": 0.05,
  "done": false
}
```

### `GET /state`
Returns `AMLState` as JSON — useful for debugging the current evidence collection status.

### `GET /`
Returns environment metadata: name, version, endpoints, task IDs.

---

## 9. Inference Script (`inference.py`)

The baseline agent script. Implements the ReAct (Reason → Act → Observe) loop.

**Configuration via environment variables:**
```bash
API_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible API
MODEL_NAME="gpt-4o-mini"                    # default model
HF_TOKEN="sk-..."                           # API key
AML_ENV_URL="http://localhost:8000"         # environment server URL
```

**How it works:**
1. Health-checks the environment server
2. Loops through all 3 tasks: `["easy", "medium", "hard"]`
3. For each task: resets environment, then runs the LLM inference loop (max 25 steps)
4. At each step: builds a message history (system prompt + all observations as user messages, LLM responses as assistant messages) and sends to LLM
5. Parses LLM JSON response (`{"tool": "...", "parameters": {...}, "reasoning": "..."}`) with a 3-tier fallback parser
6. POSTs action to environment, receives next observation
7. Logs every step in OpenEnv format: `[START]`, `[STEP]`, `[END]`

**System prompt structure (ReAct framework):**
- Phase 1: Alert Triage (`review_alert`)
- Phase 2: Customer Due Diligence (`get_customer_profile`)
- Phase 3: Transaction Analysis (`query_transactions`)
- Phase 4: Counterparty & Network Analysis (`trace_network`, `check_watchlist`)
- Phase 5: Source & Risk (`check_source_of_funds`, `assess_risk`)
- Phase 6: Determination (`file_sar` or `close_alert`)

**JSON output format enforced on the LLM:**
```json
{"tool": "<tool_name>", "parameters": {...}, "reasoning": "<one sentence>"}
```

---

## 10. Python Client (`client.py`)

`AMLEnvironmentClient` — a thin Python wrapper around the HTTP API for use in scripts and notebooks.

```python
from client import AMLEnvironmentClient

with AMLEnvironmentClient("http://localhost:8000") as client:
    obs = client.reset(task_id="easy")
    obs = client.review_alert()
    obs = client.get_customer_profile("CUST001")
    obs = client.query_transactions("CUST001")
    obs = client.check_source_of_funds("TXN-001-A")
    obs = client.file_sar(
        findings=["multiple_sub_threshold_deposits", "no_source_documentation"],
        typology="structuring",
        entities_involved=["CUST001"]
    )
    print(obs["tool_result"]["final_score"])
```

Also has `client.step(tool, parameters)` for direct tool dispatch.

---

## 11. Baseline Benchmark Results

Measured using `gpt-4o-mini` at temperature=0.0, 3 runs each:

| Task | Difficulty | Baseline Score | Success Rate | Common Failure Mode |
|---|---|---|---|---|
| Task 1: Structuring | Easy | ~0.95 | 100% | None significant |
| Task 2: Layering | Medium | ~0.82 | 85% | Over-flagging ENT_E; missing specific intermediary entity IDs |
| Task 3: Trade-Based ML | Hard | ~0.65 | 50% | Misses the market-price comparison; closes as false positive |

**Average overall baseline:** ~0.80

To reproduce: `python inference.py` (with env server running and env vars set).

---

## 12. Deployment

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
# In another terminal:
python inference.py
```

### Docker
```bash
docker build -t aml-env .
docker run -p 8000:8000 aml-env
```

### HuggingFace Spaces
The `README.md` front-matter declares this as a Docker-based HuggingFace Space:
```yaml
title: MetaHack
sdk: docker
app_port: 8000
```
Push to the HF Space repository to deploy.

### OpenEnv Spec (`openenv.yaml`)
```yaml
spec_version: 1
name: aml_investigation_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
models:
  - name: "Qwen/Qwen2.5-72B-Instruct"
tasks:
  - id: "easy" / "medium" / "hard"
```

---

## 13. Key Design Decisions & Gotchas

1. **Dual-import pattern everywhere**: Every module uses try/except for imports (e.g., `from models import ...` vs `from aml_investigation_env.models import ...`). This makes the package work whether installed as a package or run directly from the repo root.

2. **Stateful single instance**: The server keeps **one** `AMLEnvironment` instance per process (`_env = AMLEnvironment()`). This means concurrent episode resets from multiple clients would clobber each other. This is intentional for the hackathon/benchmark use case (single-agent evaluation).

3. **Score clamping**: Final scores are always clamped to `[0.001, 0.999]` — this avoids degenerate log-probability issues in downstream RL training.

4. **Entity F1 calculation**: The precision formula in the grader is non-standard — it subtracts false positives from the denominator:
   ```python
   precision = true_positives / max(len(flagged_set) - false_positives + true_positives, 1)
   ```
   Flagging an explicitly excluded entity (like ENT_E in medium, ENT_H in hard) directly hurts the precision component.

5. **The `assess_risk` tool is dynamic**: Unlike all other tools that return static scenario data, `_handle_assess_risk` computes a risk score in real-time from the current `AMLState` flags. It rewards investigation progress — more evidence gathered = higher computed risk score. The score tops out at 100.

6. **Findings matching is forgiving**: The 3-tier alias system means agents don't need to produce exact keyword strings. E.g., saying "price_manipulation" matches `over_invoicing`, and "pep" matches `pep_connection`.

7. **The `trace_network` depth-2 mechanic**: Only the Hard scenario uses `depth_2_connections`. The family link (Marcus Webb ↔ Alan Chen) is invisible at depth=1 and only surfaces at depth=2. This is why Hard is hard — agents must think to request deeper network traversal.

8. **Step budget penalty**: Exceeding 25 steps without a terminal action triggers `-0.10` penalty and forces episode end. This is separate from — and additive with — accumulated step rewards.

---

## 14. Typical Agent Failure Patterns (Known Issues to Fix)

- **Premature termination**: Filing SAR before gathering enough evidence (low findings coverage score)
- **Redundant tool calls**: Calling the same watchlist entity twice with identical parameters (-0.02 each)
- **Missing depth-2 trace**: Not calling `trace_network(entity_id, depth=2)` in the Hard scenario — misses the family connection
- **Over-flagging**: Including ENT_E (Medium) or ENT_H (Hard) in entity list — reduces F1 score
- **Wrong typology string**: Using `"trade_based_money_laundering"` instead of `"trade_based_ml"` — exact case-insensitive match required
- **Missing ENT_A in Medium**: Agents often track `CUST002, ENT_B, ENT_C, ENT_D` but forget to include `ENT_A` (Apex Holdings, the inbound source)
- **Skipping `check_watchlist("Viktor Korev")`**: Viktor Korev is the director of ENT_D, but agents need to explicitly look up his name — just checking Delta Resources returns no hit

---

## 15. File-by-File Quick Reference

| File | Primary Role | Key Class/Function |
|---|---|---|
| `server/aml_environment.py` | Core RL environment logic | `AMLEnvironment` |
| `server/app.py` | HTTP server setup + endpoint routing | FastAPI `app`, `_env` singleton |
| `models.py` | Pydantic data contracts | `AMLAction`, `AMLObservation`, `AMLState` |
| `graders/grader.py` | Deterministic scoring | `AMLGrader.grade()`, `AMLGrader.grade_step()` |
| `scenarios/base.py` | Scenario interface definition | `BaseScenario` (ABC) |
| `scenarios/easy.py` | John Doe structuring case | `EasyScenario` |
| `scenarios/medium.py` | GlobalTrade LLC layering case | `MediumScenario` |
| `scenarios/hard.py` | NovaTech TBML case | `HardScenario` |
| `scenarios/__init__.py` | Scenario factory | `get_scenario(task_id: str)` |
| `inference.py` | Baseline LLM agent | `run_task()`, `SYSTEM_PROMPT` |
| `client.py` | Python HTTP client | `AMLEnvironmentClient` |
| `openenv.yaml` | OpenEnv spec declaration | — |
| `Dockerfile` | Container definition | python:3.11-slim, port 8000 |
| `pyproject.toml` | Package build config | `aml_investigation_env` v0.1.0 |
