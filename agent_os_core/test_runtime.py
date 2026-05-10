#!/usr/bin/env python3
"""
AgentOS-Kernel — Async Tool Runtime Test Suite.

Proves:
  1. Batch concurrency  — 5 tools (max latency 2s) complete in ~2s, not 5.5s+
  2. Timeout handling   — A 2s tool with 500ms timeout returns status:"timeout"
  3. Single-call API    — execute_one works correctly
  4. Error resilience   — Unknown tools return a valid result (mock echo)
"""

import json
import time
import sys

def main():
    # ── Import ──────────────────────────────────────────────────────────
    try:
        from agent_os_core import ToolRuntime
    except ImportError:
        print("ERROR: agent_os_core not found.")
        print("Build first:  cd agent_os_core && pip install maturin && maturin develop --release")
        sys.exit(1)

    rt = ToolRuntime()
    print("=" * 70)
    print("  AgentOS-Kernel — Async Tool Runtime Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    # ── Test 1: Batch concurrency ───────────────────────────────────────
    print("\n▸ Test 1: Batch Concurrency (5 tools, expect ~2s wall-clock)")

    batch_calls = [
        ("trace_network",      '{"entity_id": "ENT_A", "depth": 2}'),     # 2000ms
        ("check_watchlist",    '{"entity_name": "Viktor Korev"}'),         #  500ms
        ("query_transactions", '{"customer_id": "CUST-123"}'),             # 1000ms
        ("request_wire_trace", '{"entity_id": "ENT_A"}'),                  # 1800ms
        ("check_device_overlap", '{"entity_id": "ENT_B"}'),               #  300ms
    ]

    t0 = time.perf_counter()
    result_json = rt.execute_batch(batch_calls, 10000)
    wall_clock = time.perf_counter() - t0
    batch = json.loads(result_json)

    print(f"  Wall-clock:      {wall_clock:.2f}s")
    print(f"  Tools succeeded: {batch['tools_succeeded']}")
    print(f"  Tools timed out: {batch['tools_timed_out']}")
    print(f"  Tools errored:   {batch['tools_errored']}")

    sequential_sum = 2.0 + 0.5 + 1.0 + 1.8 + 0.3  # = 5.6s
    if wall_clock < sequential_sum * 0.6 and batch["tools_succeeded"] == 5:
        print(f"  ✓ PASS — {wall_clock:.2f}s << {sequential_sum:.1f}s sequential (concurrency proven)")
        passed += 1
    else:
        print(f"  ✗ FAIL — wall_clock={wall_clock:.2f}s, succeeded={batch['tools_succeeded']}")
        failed += 1

    for r in batch["results"]:
        print(f"    [{r['status']:>7}] {r['tool']:25} {r['elapsed_ms']}ms")

    # ── Test 2: Timeout handling ────────────────────────────────────────
    print("\n▸ Test 2: Timeout Handling (2s tool with 500ms timeout)")

    t0 = time.perf_counter()
    result_json = rt.execute_one("trace_network", '{"entity_id": "ENT_X"}', 500)
    wall_clock = time.perf_counter() - t0
    result = json.loads(result_json)

    print(f"  Status:     {result['status']}")
    print(f"  Wall-clock: {wall_clock:.2f}s")
    print(f"  Error:      {result.get('error', 'none')}")

    if result["status"] == "timeout" and wall_clock < 1.0:
        print(f"  ✓ PASS — Tool timed out correctly in {wall_clock:.2f}s")
        passed += 1
    else:
        print(f"  ✗ FAIL — status={result['status']}, wall_clock={wall_clock:.2f}s")
        failed += 1

    # ── Test 3: Single-call API ─────────────────────────────────────────
    print("\n▸ Test 3: Single-Call API (assess_risk, ~200ms)")

    t0 = time.perf_counter()
    result_json = rt.execute_one("assess_risk", '{"customer_id": "CUST-456"}', 5000)
    wall_clock = time.perf_counter() - t0
    result = json.loads(result_json)

    print(f"  Status:     {result['status']}")
    print(f"  Risk score: {result['data'].get('risk_score', 'N/A')}")
    print(f"  Wall-clock: {wall_clock:.2f}s")

    if result["status"] == "success" and result["data"].get("risk_score") == 78:
        print(f"  ✓ PASS — Correct result in {wall_clock:.2f}s")
        passed += 1
    else:
        print(f"  ✗ FAIL — status={result['status']}, data={result['data']}")
        failed += 1

    # ── Test 4: Unknown tool (echo fallback) ────────────────────────────
    print("\n▸ Test 4: Unknown Tool (echo fallback)")

    result_json = rt.execute_one("nonexistent_tool", '{"foo": "bar"}', 5000)
    result = json.loads(result_json)

    print(f"  Status: {result['status']}")
    print(f"  Source: {result['data'].get('source', 'N/A')}")

    if result["status"] == "success" and result["data"].get("source") == "mock_default":
        print(f"  ✓ PASS — Unknown tool echoed gracefully")
        passed += 1
    else:
        print(f"  ✗ FAIL — status={result['status']}")
        failed += 1

    # ── Test 5: Batch with mixed timeouts ───────────────────────────────
    print("\n▸ Test 5: Batch with Tight Timeout (some succeed, some timeout)")

    mixed_calls = [
        ("assess_risk",        '{"customer_id": "C-1"}'),      #  200ms — should succeed
        ("check_device_overlap", '{"entity_id": "E-1"}'),      #  300ms — should succeed
        ("trace_network",      '{"entity_id": "ENT_SLOW"}'),   # 2000ms — should timeout
        ("request_wire_trace", '{"entity_id": "ENT_SLOW2"}'),  # 1800ms — should timeout
    ]

    result_json = rt.execute_batch(mixed_calls, 700)
    batch = json.loads(result_json)

    print(f"  Succeeded:  {batch['tools_succeeded']}")
    print(f"  Timed out:  {batch['tools_timed_out']}")

    for r in batch["results"]:
        print(f"    [{r['status']:>7}] {r['tool']:25} {r['elapsed_ms']}ms")

    if batch["tools_succeeded"] == 2 and batch["tools_timed_out"] == 2:
        print(f"  ✓ PASS — 2 succeeded, 2 timed out (correct)")
        passed += 1
    else:
        print(f"  ✗ FAIL — expected 2+2, got {batch['tools_succeeded']}+{batch['tools_timed_out']}")
        failed += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED ✓")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED ✗")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
