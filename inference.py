import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

# System Prompt — dynamically extended with kernel directives each turn

BASE_SYSTEM_PROMPT = """You are a Senior AML (Anti-Money Laundering) Compliance Investigator operating within a **Memex OS-Agent Environment**. You manage a limited context window (RAM) and must use OS-level tools to persist memory and acquire intelligence.

## COGNITIVE FRAMEWORK: ReAct (Reason → Act → Observe)

At each step you MUST internally follow this loop:
1. **REASON**: What do I know so far? What is the most critical evidence gap?
2. **ACT**: Select the single most informative tool call to close that gap.
3. **OBSERVE**: After receiving the result, update your mental model before the next step.

Prioritize *information gain per step*. Never repeat a tool call with identical parameters.

## OS MECHANICS (CRITICAL)

You operate under three OS constraints:

### I. Virtual Memory (RAM Eviction)
- Your context window only holds the **last 2 observations**. Older data is PERMANENTLY LOST.
- Use `write_to_case_file(content="...")` to page important findings to your persistent disk.
- If you reference an entity ID that was evicted from RAM and NOT saved to disk, you incur a **Page Fault penalty (-0.05)**.

### II. Interrupts (Async Background Tasks)
- `request_wire_trace(entity_id/transaction_id)` returns a Job ID + ETA (2-4 steps). The data is NOT available immediately.
- **Do NOT wait idle.** Pivot to other investigation tasks while the async job completes.
- Use `retrieve_async_result(job_id="REQ-XXX")` ONLY when the ETA has reached 0.
- Premature retrieval incurs an **Async Timeout penalty (-0.10)**.

### III. Kernel Updates (Self-Improvement)
- You start with basic directives. Use `search_compliance_manual(query="...")` to find AML rules.
- Then call `update_system_prompt(rule="...")` to inject the rule into your active directives.
- This earns a **Meta-Injection reward (+0.15)** and improves your decision-making.

## INVESTIGATION PROTOCOL

Phase 1 — ALERT TRIAGE: `review_alert`
Phase 2 — CUSTOMER DUE DILIGENCE: `get_customer_profile(customer_id)`
Phase 3 — TRANSACTION ANALYSIS: `query_transactions(customer_id)`
Phase 4 — SAVE TO DISK: `write_to_case_file(content="key findings so far...")`
Phase 5 — ASYNC INTELLIGENCE: `request_wire_trace(entity_id)` → note the Job ID
Phase 6 — COMPLIANCE RULES: `search_compliance_manual(query)` → `update_system_prompt(rule)`
Phase 7 — NETWORK & WATCHLIST: `trace_network(entity_id, depth=2)`, `check_watchlist(entity_name)`
Phase 8 — RETRIEVE ASYNC: `retrieve_async_result(job_id)` (when ETA=0)
Phase 9 — DETERMINATION: `file_sar` or `close_alert`

## AVAILABLE TOOLS (15 total)

### Domain Tools
- review_alert: {alert_id: optional}
- get_customer_profile: {customer_id: string}
- query_transactions: {customer_id, date_from?, date_to?, min_amount?, max_amount?}
- check_watchlist: {entity_name, list_type?: all|OFAC|PEP|UN}
- trace_network: {entity_id, depth?: 1|2}
- check_source_of_funds: {transaction_id}
- check_market_price: {commodity} — compare invoiced vs market prices
- assess_risk: {customer_id}
- file_sar: {findings: [], typology: string, entities_involved: []} — TERMINAL
- close_alert: {reason, findings?: []} — TERMINAL

### OS-Mechanic Tools
- write_to_case_file: {content: string} — page data to persistent disk (+0.10)
- request_wire_trace: {entity_id?, transaction_id?} — async job, returns job_id + ETA
- retrieve_async_result: {job_id} — get completed job result
- search_compliance_manual: {query, category?, max_results?} — find AML rules
- update_system_prompt: {rule: string} — inject rule into kernel (+0.15)

## TYPOLOGY VALUES
"structuring" | "layering" | "trade_based_ml" | "false_positive"

## OUTPUT FORMAT (STRICTLY ENFORCED)

Respond with EXACTLY ONE raw JSON object per turn. No markdown. No code fences.
{"tool": "<tool_name>", "parameters": {<params>}, "reasoning": "<one sentence>"}

VIOLATION OF THE OUTPUT FORMAT WILL CAUSE A PARSING FAILURE.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_flat = action.replace("\n", " ").replace("\r", " ")
    print(f"[STEP] step={step} action={action_flat} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

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
    resp = httpx.post(
        f"{AML_ENV_URL}/step",
        json={"action": {"tool": tool, "parameters": parameters}},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()

def env_health() -> bool:
    try:
        resp = httpx.get(f"{AML_ENV_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def build_message_history(
    ram_contents: List[str],
    disk_contents: List[str],
    kernel_directives: List[str],
    current_obs: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build LLM message history enforcing the Virtual Memory constraint.

    Only includes:
    - System prompt + kernel directives (dynamic)
    - Disk contents (persistent scratchpad)
    - RAM contents (last 2 observations)
    - Current observation
    """
    # Build dynamic system prompt with kernel directives
    system_parts = [BASE_SYSTEM_PROMPT]
    if len(kernel_directives) > 1:
        system_parts.append("\n## ACTIVE KERNEL DIRECTIVES (injected by you)")
        for d in kernel_directives:
            system_parts.append(f"- {d}")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "\n".join(system_parts)}
    ]

    # Add disk contents as persistent context
    if disk_contents:
        disk_text = "## YOUR CASE FILE (Disk — persistent across evictions)\n"
        for i, entry in enumerate(disk_contents, 1):
            disk_text += f"{i}. {entry}\n"
        messages.append({"role": "user", "content": disk_text})

    # Add RAM contents (only last 2 observations)
    for obs_text in ram_contents:
        messages.append({"role": "user", "content": f"[RAM] {obs_text}"})

    # Add the current observation
    messages.append({"role": "user", "content": json.dumps(current_obs, indent=2)})

    return messages


def run_task(task_id: str, llm: OpenAI) -> Dict[str, Any]:
    """Run a single AML investigation episode with OS mechanics."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    obs = env_reset(task_id)

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

            if done:
                tr = obs_data.get("tool_result", {})
                final_score = tr.get("final_score", reward or 0.0)
                success = final_score > 0.0
                break

            # Extract OS mechanic state from AGUI payload
            agui = obs_data.get("metadata", {}).get("agui_state", {})
            ram_contents = agui.get("ram_usage", {}).get("active_context", [])
            disk_contents = agui.get("disk_storage", [])
            kernel_directives = agui.get("kernel_directives", [])

            current_entry = {
                "step": step_num,
                "message": obs_data.get("message", ""),
                "tool_result": obs_data.get("tool_result", {}),
                "available_tools": obs_data.get("available_tools", []),
                "reward": reward,
                "async_jobs": agui.get("async_jobs", []),
            }

            try:
                messages = build_message_history(
                    ram_contents, disk_contents, kernel_directives, current_entry
                )
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
        final_score = max(-1.0, min(1.0, float(final_score)))
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
