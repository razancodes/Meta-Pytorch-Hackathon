import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple
import re

import httpx
from openai import OpenAI

# ---- Configuration ---------------------------------------------------- #
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
AML_ENV_URL: str = os.environ.get("AML_ENV_URL", "http://localhost:8000").rstrip("/")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 25
REQUEST_TIMEOUT = 60.0
BENCHMARK = "aml_investigation_env"

SYSTEM_PROMPT = """You are a Senior AML (Anti-Money Laundering) Compliance Investigator operating within a structured decision-making environment. You have been assigned a transaction monitoring alert. Your task is to systematically gather evidence using the available tools, reason over that evidence, and render a final determination.

## COGNITIVE FRAMEWORK: ReAct (Reason → Act → Observe)

At each step you MUST internally follow this loop:
1. **REASON**: What do I know so far? What is the most critical evidence gap?
2. **ACT**: Select the single most informative tool call to close that gap.
3. **OBSERVE**: After receiving the result, update your mental model before the next step.

Prioritize *information gain per step*. Never repeat a tool call with identical parameters. If an observation reveals new entity IDs or transaction IDs, immediately plan to investigate them.

## INVESTIGATION PROTOCOL (execute in order when applicable)

Phase 1 — ALERT TRIAGE:
  • review_alert → read alert narrative, note flagged customer ID, amounts, dates.

Phase 2 — CUSTOMER DUE DILIGENCE:
  • get_customer_profile(customer_id) → note occupation, account age, risk rating, jurisdiction.

Phase 3 — TRANSACTION ANALYSIS:
  • query_transactions(customer_id) → look for structuring patterns (amounts clustering just below thresholds), rapid fan-out, round-tripping, or price anomalies.

Phase 4 — COUNTERPARTY & NETWORK ANALYSIS:
  • trace_network(entity_id, depth=2) → map counterparties, beneficial owners, shared addresses.
  • check_watchlist(entity_name) → screen every entity discovered for OFAC/PEP/UN hits.

Phase 5 — SOURCE & RISK:
  • check_source_of_funds(transaction_id) → verify documentation for suspicious transactions.
  • assess_risk(customer_id) → obtain computed risk score.

Phase 6 — DETERMINATION:
  • file_sar OR close_alert — only when you have sufficient evidence.

## TYPOLOGY DETECTION CHECKLIST

When filing a SAR, identify and report the correct typology:

**structuring**: Multiple deposits/withdrawals just below the $10,000 CTR reporting threshold. Key indicators: amounts clustering at $9,000-$9,999, same branch, short time window, no cash-intensive occupation.

**layering**: Rapid movement of funds through multiple entities to obscure origin. Key indicators: fan-out to 3+ entities within 24-48 hours, shell companies, shared registered addresses, PEP connections, newly incorporated entities, offshore jurisdictions.

**trade_based_ml**: Over/under-invoicing in trade transactions to transfer value. Key indicators: unit prices deviating significantly from market value, FATF-jurisdiction counterparties, beneficial ownership links between buyer and seller, reversed or corrected transactions, unexplained inbound funds.

## AVAILABLE TOOLS (with parameter specifications)

- review_alert: Review full alert details. Params: {"alert_id": "string (optional)"}
- get_customer_profile: KYC data lookup. Params: {"customer_id": "string"}
- query_transactions: Transaction history with filters. Params: {"customer_id": "string", "date_from": "YYYY-MM-DD (opt)", "date_to": "YYYY-MM-DD (opt)", "min_amount": float (opt), "max_amount": float (opt)}
- check_watchlist: Sanctions/PEP screening. Params: {"entity_name": "string", "list_type": "all|OFAC|PEP|UN (opt)"}
- trace_network: Entity relationship graph. Params: {"entity_id": "string", "depth": 1 or 2}
- check_source_of_funds: Source documentation check. Params: {"transaction_id": "string"}
- assess_risk: Computed risk score. Params: {"customer_id": "string"}
- file_sar: TERMINAL — File a SAR. Params: {"findings": ["list of finding strings"], "typology": "string", "entities_involved": ["list of entity IDs"]}
- close_alert: TERMINAL — Close alert. Params: {"reason": "string", "findings": ["optional list"]}

## CRITICAL: OUTPUT FORMAT (STRICTLY ENFORCED)

You MUST respond with EXACTLY ONE raw JSON object per turn. No markdown. No code fences. No commentary before or after. No conversational text. Your entire response must be parseable by json.loads().

Format:
{"tool": "<tool_name>", "parameters": {<params>}, "reasoning": "<one sentence>"}

When filing a SAR, include ALL relevant entity IDs (customer + counterparties) and use precise finding keywords such as: sub_threshold, no_source_documentation, rapid_fan_out, pep_connection, shared_registered_address, over_invoicing, beneficial_owner_connection, fatf_jurisdiction, reversed_transaction, unexplained_funds.

Typology values: "structuring" | "layering" | "trade_based_ml" | "false_positive"

VIOLATION OF THE OUTPUT FORMAT WILL CAUSE A PARSING FAILURE. Output ONLY the JSON object.
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Flatten action str to remove newlines
    action_flat = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_flat} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_llm_client() -> OpenAI:
    api_key = HF_TOKEN or "no-key"
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)

def parse_tool_call(content: str) -> Tuple[str, Dict[str, Any]]:
    content = re.sub(r"```(?:json)?\s*", "", content).strip("` \n")
    try:
        parsed = json.loads(content)
        return parsed.get("tool", "review_alert"), parsed.get("parameters", {})
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed.get("tool", "review_alert"), parsed.get("parameters", {})
        except json.JSONDecodeError:
            pass
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', content)
    tool = tool_match.group(1) if tool_match else "review_alert"
    return tool, {}

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = httpx.post(f"{AML_ENV_URL}/reset", json={"task_id": task_id}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def env_step(tool: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(f"{AML_ENV_URL}/step", json={"action": {"tool": tool, "parameters": parameters}}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def env_health() -> bool:
    try:
        resp = httpx.get(f"{AML_ENV_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False

def build_message_history(obs_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in obs_history:
        messages.append({"role": "user", "content": json.dumps(entry, indent=2)})
        if entry.get("_llm_response"):
            messages.append({"role": "assistant", "content": entry["_llm_response"]})
    return messages

def run_task(task_id: str, llm: OpenAI) -> Dict[str, Any]:
    """Run a single AML investigation episode from reset to terminal action.

    Implements the outer inference loop that couples an LLM agent to the
    AMLEnvironment HTTP server. Each iteration: (1) feeds the cumulative
    observation history to the LLM, (2) parses the raw JSON response into
    a (tool, params) tuple, (3) POSTs the action to the environment, and
    (4) logs the step in the OpenEnv-mandated ``[STEP]`` format.

    The loop terminates when the environment returns ``done=True`` (agent
    called a terminal action or hit the step budget) or when the LLM call
    fails. The final score is clamped to (0.001, 0.999) for compatibility
    with downstream reward-model training.

    Args:
        task_id: Scenario identifier ('easy', 'medium', or 'hard').
        llm: Pre-configured OpenAI client pointing at the target LLM endpoint.

    Returns:
        A dict containing the final ``score`` for this episode.
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    obs = env_reset(task_id)

    obs_history: List[Dict[str, Any]] = []
    step_rewards: List[float] = []
    final_score = 0.0
    success = False
    error_msg = None
    step_num = 0

    try:
        for step_num_loop in range(1, MAX_STEPS_PER_TASK + 1):
            step_num = step_num_loop
            obs_data = obs.get("observation", obs)
            done = obs.get("done", obs_data.get("done", False))
            reward = obs.get("reward", obs_data.get("reward")) or 0.0

            current_entry = {
                "step": step_num,
                "observation": {
                    "message": obs_data.get("message", ""),
                    "tool_result": obs_data.get("tool_result", {}),
                    "available_tools": obs_data.get("available_tools", []),
                    "done": done,
                    "reward": reward,
                },
            }
            obs_history.append(current_entry)

            if done:
                tr = obs_data.get("tool_result", {})
                final_score = tr.get("final_score", reward or 0.0)
                success = final_score > 0.0  # Assuming score > 0 is success, or adjust threshold
                break

            try:
                messages = build_message_history(obs_history)
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                llm_content = response.choices[0].message.content or ""
            except Exception as exc:
                error_msg = f"LLM call failed: {exc}"
                log_step(step=step_num, action="ERROR", reward=0.0, done=False, error=error_msg)
                break

            obs_history[-1]["_llm_response"] = llm_content
            tool, parameters = parse_tool_call(llm_content)

            action_str = f"tool={tool} params={json.dumps(parameters)}"
            
            try:
                obs = env_step(tool, parameters)
                step_reward = obs.get("reward") or 0.0
                done = obs.get("done", False)
                step_rewards.append(step_reward)
                obs_inner = obs.get("observation", obs)
            except Exception as exc:
                error_msg = f"Step failed: {exc}"
                log_step(step=step_num, action=action_str, reward=0.0, done=False, error=error_msg)
                break

            log_step(step=step_num, action=action_str, reward=step_reward, done=done, error=None)

            if done:
                tr = obs_inner.get("tool_result", {})
                final_score = tr.get("final_score", step_reward)
                success = final_score > 0.0
                break

    finally:
        final_score = max(0.001, min(0.999, float(final_score)))
        log_end(success=success, steps=step_num, score=final_score, rewards=step_rewards)
    return {"score": final_score}

def main() -> None:
    if not env_health():
        print(f"ERROR: AML environment server not reachable at {AML_ENV_URL}.", file=sys.stderr)
        sys.exit(1)

    llm = build_llm_client()
    for task_id in TASKS:
        run_task(task_id, llm)
        time.sleep(1)

if __name__ == "__main__":
    main()
