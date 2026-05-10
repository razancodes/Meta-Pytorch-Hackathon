"""
AgentOS-Kernel — Full Integration Test Suite (Mock Mode).

Tests the complete AgentOS pipeline using mock backends:
  - Mock LLM (keyword-based tool selection)
  - Mock compactor (heuristic entity extraction)
  - Mock embeddings (deterministic word hashing)
  - Real Rust ToolRuntime (Tokio, mock handlers)

All 6 tests validate the architecture without requiring GPU.
"""

import asyncio
import json
import sys

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from agent_os import AgentOS

WIDTH = 70
PASS = 0
FAIL = 0


def header(text: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"▸ {text}")
    print(f"{'─' * WIDTH}")


def check(condition: bool, label: str) -> bool:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"    ✓ {label}")
    else:
        FAIL += 1
        print(f"    ✗ {label}")
    return condition


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def run_tests():
    print("=" * WIDTH)
    print("  AgentOS-Kernel — Full Integration Test (Mock Mode)")
    print("=" * WIDTH)

    # Instantiate with ALL mocks enabled
    agent = AgentOS(
        max_l1_tokens=600,
        max_l2_tokens=400,
        l3_db_path="/tmp/agent_os_integration_test",
        use_mock_llm=True,
        use_mock_compactor=True,
        use_mock_embeddings=True,
    )

    try:
        from agent_os_core import ToolRuntime
        rust = "✓ Available"
    except ImportError:
        rust = "✗ Not available"

    print(f"\n  Rust runtime: {rust}")
    print(f"  L1 budget:    {agent._memory.max_l1_tokens} tokens")
    print(f"  L2 budget:    {agent._memory.max_l2_tokens} tokens")
    print(f"  Mode:         all-mock")

    # ── Turn 1: Investigation ──
    header("Turn 1: 'Investigate alert ALT-7729 for customer CUST-A'")
    r1 = await agent.step("Investigate alert ALT-7729 for customer CUST-A")
    reasoning = r1.llm_response.get("reasoning", "")
    tool = r1.llm_response.get("tool", "")
    status = r1.tool_results.get("status", "") if r1.tool_results else ""
    print(f"  LLM reasoning:  {reasoning[:80]}")
    print(f"  Tool called:    {tool}")
    print(f"  Tool status:    {status}")
    print(f"  L1 compacted:   {r1.l1_compacted}")
    print(f"  L3 retrieved:   {r1.l3_facts_retrieved} facts")
    print(f"  Prompt tokens:  {r1.prompt_token_count}")
    print(f"  Elapsed:        {r1.elapsed_ms / 1000:.2f}s")
    print(f"  State:          {agent}")
    all_pass = check(tool == "query_transactions", "Turn 1: correct tool selected")

    # ── Turn 2: Network Trace ──
    header("Turn 2: 'Who is the beneficial owner of CUST-A?'")
    r2 = await agent.step("Who is the beneficial owner of CUST-A?")
    tool2 = r2.llm_response.get("tool", "")
    print(f"  LLM reasoning:  {r2.llm_response.get('reasoning', '')[:80]}")
    print(f"  Tool called:    {tool2}")
    print(f"  Tool status:    {r2.tool_results.get('status', '') if r2.tool_results else ''}")
    print(f"  L1 compacted:   {r2.l1_compacted}")
    print(f"  L2→L3 archived: {r2.l2_archived_to_l3}")
    print(f"  L3 retrieved:   {r2.l3_facts_retrieved} facts")
    print(f"  Prompt tokens:  {r2.prompt_token_count}")
    print(f"  State:          {agent}")
    all_pass = check(tool2 == "trace_network", "Turn 2: correct tool selected") and all_pass

    # ── Turn 3: Risk Assessment ──
    header("Turn 3: 'Run a risk assessment on this entire cluster'")
    r3 = await agent.step("Run a risk assessment on this entire cluster")
    tool3 = r3.llm_response.get("tool", "")
    print(f"  LLM reasoning:  {r3.llm_response.get('reasoning', '')[:80]}")
    print(f"  Tool called:    {tool3}")
    print(f"  Tool status:    {r3.tool_results.get('status', '') if r3.tool_results else ''}")
    print(f"  L1 compacted:   {r3.l1_compacted}")
    print(f"  L2→L3 archived: {r3.l2_archived_to_l3}")
    print(f"  L3 retrieved:   {r3.l3_facts_retrieved} facts")
    print(f"  Prompt tokens:  {r3.prompt_token_count}")
    print(f"  State:          {agent}")
    all_pass = check(tool3 == "assess_risk", "Turn 3: correct tool selected") and all_pass

    # ── Test 4: State Integrity ──
    header("Test 4: State Integrity After 3 Turns")
    state = agent.get_full_state()
    mem = state["memory"]
    l3 = state["l3_index"]
    print(f"  Turns completed:   {state['turn_number']}")
    print(f"  Total tool calls:  {state['total_tool_calls']}")
    print(f"  L1 tokens:         {mem['l1_tokens']}/{mem['l1_budget']}")
    print(f"  L1 turns:          {mem['l1_turns']}")
    print(f"  L2 entities:       {mem['l2_entities']}")
    print(f"  L2 facts:          {mem['l2_facts']}")
    print(f"  L3 rows:           {l3['rows_in_table']}")
    print(f"  Compactions run:   {mem['compactions_run']}")
    check(state["turn_number"] == 3, "3 turns completed")
    check(state["total_tool_calls"] == 3, "3 tool calls made")
    check(mem["l1_tokens"] <= mem["l1_budget"], "L1 under budget")
    all_pass = check(state["rust_runtime_available"], "Rust runtime used") and all_pass

    # ── Test 5: Prompt Assembly ──
    header("Test 5: Prompt Assembly (full pipeline proof)")
    messages = agent._memory.get_injected_prompt()
    print(f"  System message:  {'present' if messages[0]['role'] == 'system' else 'MISSING'}")
    has_l2 = "SCRATCHPAD" in messages[0]["content"] or "Scratchpad" in messages[0]["content"]
    print(f"  Has L2 preamble: {has_l2}")
    print(f"  History turns:   {len(messages) - 1}")
    print(f"  Total messages:  {len(messages)}")
    print(f"\n  System prompt preview:")
    sys_content = messages[0]["content"][:200]
    print(f"    {sys_content}")
    all_pass = check(
        messages[0]["role"] == "system" and len(messages) >= 4,
        "Prompt assembly produces valid message list"
    ) and all_pass

    # ── Test 6: Data Flow ──
    header("Test 6: End-to-End Data Flow Verification")
    all_content = " ".join(m["content"] for m in messages[1:])
    has_tool = any(m.role == "tool" for m in agent._memory._l1_messages)
    has_user = any(m.role == "user" for m in agent._memory._l1_messages)
    has_assistant = any(m.role == "assistant" for m in agent._memory._l1_messages)
    print(f"  Tool results in L1: {has_tool}")
    print(f"  User messages in L1: {has_user}")
    print(f"  LLM responses in L1: {has_assistant}")
    all_pass = check(
        has_tool and has_user and has_assistant,
        "User→LLM→Rust→L1 data flow verified"
    ) and all_pass

    # ── Cleanup ──
    agent.cleanup()

    # ── Summary ──
    print(f"\n{'=' * WIDTH}")
    if FAIL == 0:
        print(f"  ALL {PASS} TESTS PASSED ✓")
    else:
        print(f"  {PASS} PASSED, {FAIL} FAILED ✗")
    print(f"{'=' * WIDTH}\n")

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
