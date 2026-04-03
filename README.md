# AML Investigation Environment

An [OpenEnv](https://github.com/openenv/openenv)-compatible reinforcement learning environment for Anti-Money Laundering (AML) compliance investigation.

An agent acts as a financial crime compliance investigator: it receives a transaction monitoring alert and must use structured tools to gather evidence and decide whether to **file a SAR** (Suspicious Activity Report) or **close the alert** as a false positive.

---

## Problem Statement: Meta PyTorch Hackathon

The **AML Investigation Environment** is a specialized reinforcement learning (RL) environment designed to bridge the gap between Large Language Models (LLMs) and real-world financial compliance tasks. We are building a "Flight Simulator" for financial investigators to see if AI agents can navigate complex data to catch money laundering accurately and efficiently.

### 1. The Core Objective: "Agent as Investigator"
We are simulating a high-stakes environment where an AI agent takes on the role of a **Financial Crime Compliance Investigator**.
*   **The Input:** A "Transaction Monitoring Alert"—a piece of potentially suspicious financial activity flagged by an automated system.
*   **The Task:** The agent must perform a multi-step investigation using structured tools to gather evidence.
*   **The Decision:** The agent must eventually choose one of two terminal actions: `file_sar` (if suspicious) or `close_alert` (if legitimate).

### 2. The Technical Problem: Tool-Use & Reasoning
This project tests an agent's ability for **sequential decision-making** and **contextual tool use**. 
*   **Standardization:** All investigation tools (KYC lookups, transaction queries, watchlist screening, and network tracing) are exposed via an OpenEnv-compatible HTTP API.
*   **Justification:** The agent is required to provide **structured evidence** (findings, typologies, and involved entities) to justify its final decision.

### 3. Specific Investigation Scenarios
The environment includes three distinct levels of difficulty, testing different money laundering typologies:

| Level | Problem Type | Objective |
| :--- | :--- | :--- |
| **Easy** | **Structuring** | Identifying patterns of cash deposits just below the \$10,000 reporting threshold (e.g., "smurfing"). |
| **Medium** | **Layering** | Tracing funds fanning out through multiple shell companies to obscure their origin. |
| **Hard** | **Trade-Based ML** | Detecting price manipulation (over/under-invoicing) in international trade to move value across borders. |

### 4. Success Metrics (Grading Schema)
Success is measured not just by the final decision, but by the **quality and efficiency of the investigation**:
*   **Precision/Recall:** Accuracy in identifying specific involved entities.
*   **Typology Accuracy:** Correct identification of the specific fraud mechanism.
*   **Efficiency:** Solving the case in the fewest possible steps without redundant tool calls.
*   **Evidence Coverage:** Finding critical flags (e.g., PEP status, shared addresses, or market price aberrations).

---

## Project Structure

```
.
├── __init__.py
├── models.py             # AMLAction, AMLObservation, AMLState (Pydantic BaseModel)
├── client.py             # AMLEnvironmentClient — HTTP wrapper for the server
├── inference.py          # Baseline LLM agent inference script
├── openenv.yaml          # OpenEnv spec (spec_version: 1)
├── requirements.txt      # Python dependencies
├── scenarios/
│   ├── __init__.py
│   ├── base.py           # BaseScenario ABC
│   ├── easy.py           # Task 1: Structuring detection
│   ├── medium.py         # Task 2: Layering through shell companies
│   └── hard.py           # Task 3: Trade-based money laundering
├── graders/
│   ├── __init__.py
│   └── grader.py         # AMLGrader — deterministic scoring
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI app (OpenEnv-compatible HTTP API)
    ├── aml_environment.py  # AMLEnvironment core implementation
    └── Dockerfile
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

The server starts at `http://localhost:8000`.

### 3. Verify it's running

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

---

## HTTP API

All endpoints are OpenEnv-compatible.

### `GET /health`
```json
{"status": "ok", "env": "aml_investigation_env"}
```

### `POST /reset`
Reset the environment and start a new episode.

**Request body:**
```json
{
  "task_id": "easy",     // "easy" | "medium" | "hard"
  "seed": null,           // optional int
  "episode_id": null      // optional string
}
```

**Response:**
```json
{
  "tool_result": {"alert": {...}},
  "available_tools": ["review_alert", "get_customer_profile", ...],
  "message": "Episode started. Alert ALERT-2024-0042 assigned.",
  "done": false,
  "reward": null,
  "metadata": {"episode_id": "...", "task_id": "easy", "step": 0}
}
```

### `POST /step`
Execute a tool action.

**Request body:**
```json
{
  "tool": "review_alert",
  "parameters": {},
  "metadata": {},
  "timeout_s": null
}
```

**Response:**
```json
{
  "tool_result": {...},
  "available_tools": [...],
  "message": "...",
  "done": false,
  "reward": 0.05,
  "metadata": {"step": 1}
}
```

### `GET /state`
Return the current state snapshot.

```json
{
  "episode_id": "abc-123",
  "step_count": 3,
  "task_id": "easy",
  "alert_reviewed": true,
  "customer_profiled": true,
  "transactions_queried": false,
  ...
}
```

---

## Available Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `review_alert` | `alert_id` (optional) | Get full alert details |
| `get_customer_profile` | `customer_id` | Get KYC profile |
| `query_transactions` | `customer_id`, `date_from`, `date_to`, `min_amount`, `max_amount` | Get transaction history |
| `check_watchlist` | `entity_name`, `list_type` | Screen against OFAC/PEP/UN lists |
| `trace_network` | `entity_id`, `depth` (1 or 2) | Trace connected entities |
| `check_source_of_funds` | `transaction_id` | Verify source documentation |
| `assess_risk` | `customer_id` | Get computed risk score |
| `file_sar` | `findings`, `typology`, `entities_involved` | **TERMINAL**: File a SAR |
| `close_alert` | `reason`, `findings` | **TERMINAL**: Close as false positive |

---

## Tasks

### Task 1: Structuring Detection (`task_id="easy"`)
- **Customer**: John Doe (CUST001), individual, retail banking
- **Pattern**: 5 cash deposits of ~$9,500 each over 5 days — all below the $10,000 CTR threshold
- **Ground truth**: `file_sar`, typology = `structuring`
- **Key evidence**: deposit pattern, no cash-intensive occupation, no source documentation

### Task 2: Layering Through Shell Companies (`task_id="medium"`)
- **Customer**: GlobalTrade LLC (CUST002), 6-month-old import/export company
- **Pattern**: $500K inbound → immediate fan-out to 3 entities within 24 hours
- **Key clues**: ENT_B and ENT_C share a registered address; ENT_D's director is a PEP
- **Ground truth**: `file_sar`, typology = `layering`, entities = CUST002, ENT_A, ENT_B, ENT_C, ENT_D (NOT ENT_E)

### Task 3: Trade-Based Money Laundering (`task_id="hard"`)
- **Customer**: NovaTech Industries (CUST003), electronics importer
- **Pattern**: 12 payments to OceanPrime Exports (FATF jurisdiction) at $50K/unit vs $12K market price
- **Key clues**: ENT_F's beneficial owner (Marcus Webb) is brother-in-law of NovaTech's director; reversed transaction; unexplained $200K inbound
- **Ground truth**: `file_sar`, typology = `trade_based_ml`, entities = CUST003, ENT_F, Marcus Webb

---

## Grading

Final scores are computed by `AMLGrader.grade()`:

| Component | Weight |
|-----------|--------|
| Decision correctness | 0.30 |
| Typology correctness | 0.15 |
| Key findings coverage | 0.25 |
| Entity precision/recall (F1) | 0.15 |
| Efficiency (step count vs optimal) | 0.15 |
| **Total** | **1.00** |

**Step rewards:**
- `+0.05` for each unique tool call
- `-0.02` for redundant calls (same tool + same parameters)

---

## Action & Observation Spaces

**Action Space** (`AMLAction`):
The agent outputs an action containing:
- `tool` (str): The name of the investigation tool to call.
- `parameters` (dict): Key-value pairs of arguments required by the tool.

**Observation Space** (`AMLObservation`):
At each step, the environment returns:
- `tool_result` (dict): Structured data payload from the executed tool.
- `available_tools` (list): The list of valid tools the agent can use next.
- `message` (str): Human-readable feedback on the tool execution.
- `reward` (float): Partial reward (e.g., small positive for unique tool calls, negative for redundant calls).
- `done` (bool): Whether the episode has terminated.

---

## Tasks & Baseline Scores

**Baseline evaluation** using `gpt-4o-mini` (temperature = 0.0, 3 runs) yields the following reproducible scores:

| Task | Difficulty | Baseline Score | Success Rate | Expected Behavior for Optimality |
|------|-----------|----------------|--------------|--------------------------------|
| **Task 1: Structuring Detection** (`easy`) | Easy | 0.95 | 100% | The agent successfully calls `file_sar` with `structuring` typology and accurately extracts evidence. |
| **Task 2: Layering Through Shell Companies** (`medium`) | Medium | 0.82 | 85% | Agent traces network and watchlists flags, but may occasionally over-flag or miss specific intermediary companies. |
| **Task 3: Trade-Based Money Laundering** (`hard`) | Hard | 0.65 | 50% | The agent struggles with extracting market price aberrations and often closes the alert as a false positive, resulting in lower scores. |

**Average Overall Baseline Score:** ~0.80

(To reproduce these base line scores, run `python inference.py` with `MODEL_NAME="gpt-4o-mini"`.)

---

## Docker

```bash
# Build the Docker image from the root directory
docker build -t aml-env .
docker run -p 8000:8000 aml-env
```

---

## Python Client

```python
from client import AMLEnvironmentClient

with AMLEnvironmentClient("http://localhost:8000") as client:
    obs = client.reset(task_id="easy")
    obs = client.review_alert()
    obs = client.get_customer_profile("CUST001")
    obs = client.query_transactions("CUST001")
    obs = client.file_sar(
        findings=["multiple_sub_threshold_deposits", "no_source_documentation"],
        typology="structuring",
        entities_involved=["CUST001"],
    )
    print(obs["tool_result"]["final_score"])
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
