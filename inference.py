import asyncio
import json
import os
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

SYSTEM_PROMPT = """You are an AML (Anti-Money Laundering) compliance investigator at a financial institution.
You have been assigned a transaction monitoring alert to investigate.

Your goal is to gather enough evidence to make a well-reasoned decision:
either FILE A SAR (Suspicious Activity Report) or CLOSE the alert as a false positive.

Use the available tools to investigate. Be thorough but efficient — unnecessary repeated queries waste time.

Available tools:
- review_alert: Review the full alert details. Parameters: {"alert_id": "string (optional)"}
- get_customer_profile: Get KYC data for a customer. Parameters: {"customer_id": "string"}
- query_transactions: Get transaction history. Parameters: {"customer_id": "string", "date_from": "YYYY-MM-DD (optional)", "date_to": "YYYY-MM-DD (optional)", "min_amount": float (optional), "max_amount": float (optional)}
- check_watchlist: Screen an entity against sanctions/PEP lists. Parameters: {"entity_name": "string", "list_type": "all|OFAC|PEP|UN (optional)"}
- trace_network: Get connected entities (counterparties, directors, beneficial owners). Parameters: {"entity_id": "string", "depth": 1 or 2}
- check_source_of_funds: Verify source documentation for a transaction. Parameters: {"transaction_id": "string"}
- assess_risk: Get a computed risk score based on all gathered evidence. Parameters: {"customer_id": "string"}
- file_sar: TERMINAL — File a SAR. Parameters: {"findings": ["list of finding strings"], "typology": "string", "entities_involved": ["list of entity IDs"]}
- close_alert: TERMINAL — Close the alert as false positive. Parameters: {"reason": "string", "findings": ["optional list"]}

Always respond with a single JSON object in this exact format (no other text):
{"tool": "<tool_name>", "parameters": {<parameter_dict>}, "reasoning": "<one-sentence explanation>"}

When you have gathered enough evidence, call file_sar or close_alert.
Typology options: "structuring", "layering", "trade_based_ml", "false_positive"
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
