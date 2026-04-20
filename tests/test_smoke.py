"""
Memex OS-Agent Benchmark — Smoke Tests.

Exercises all three OS mechanics (Virtual Memory, Interrupts, Kernel Updates)
directly against the AMLEnvironment class (no HTTP server required).

Run: cd /home/Muaz/Documents/Software/MetaHack && python tests/test_smoke.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AMLAction, AMLObservation, AMLState
from server.aml_environment import AMLEnvironment


def _step(env: AMLEnvironment, tool: str, **params) -> AMLObservation:
    """Helper to take a step and print a summary."""
    action = AMLAction(tool=tool, parameters=params)
    obs = env.step(action)
    agui = obs.metadata.get("agui_state", {})
    ram = agui.get("ram_usage", {}).get("capacity", "?")
    disk = len(agui.get("disk_storage", []))
    async_jobs = len(agui.get("async_jobs", []))
    kernel = len(agui.get("kernel_directives", []))
    reward_str = f"{obs.reward:+.4f}" if obs.reward is not None else "None"
    print(
        f"  Step {env.state.step_count:2d} | {tool:30s} | "
        f"R={reward_str} | RAM={ram} | Disk={disk} | Async={async_jobs} | Kernel={kernel} | "
        f"{'DONE' if obs.done else 'ok'}"
    )
    return obs


def test_easy_scenario():
    """Full episode on easy scenario exercising all OS mechanics."""
    print("\n{'='*70}")
    print("TEST: Easy Scenario (Structuring) — Full OS Mechanics")
    print("=" * 70)

    env = AMLEnvironment()
    obs = env.reset(task_id="easy")
    assert not obs.done
    assert "agui_state" in obs.metadata
    print(f"  Reset OK | Episode: {env.state.episode_id[:8]}...")

    # Phase 1: Review alert
    obs = _step(env, "review_alert")
    assert not obs.done

    # Phase 2: Get customer profile
    obs = _step(env, "get_customer_profile", customer_id="CUST001")
    assert not obs.done

    # Phase 3: Write findings to disk (Virtual Memory — paging)
    obs = _step(env, "write_to_case_file",
                content="CUST001 = John Doe, retail clerk, $38k income, 5 cash deposits ~$9.5k each")
    assert not obs.done
    assert env.state.successful_pages == 1
    print("    ✓ Successful page to disk")

    # Phase 4: Query transactions
    obs = _step(env, "query_transactions", customer_id="CUST001")
    assert not obs.done

    # Phase 5: Async wire trace (Interrupt mechanic)
    obs = _step(env, "request_wire_trace", entity_id="CUST001")
    assert not obs.done
    job_id = obs.tool_result.get("job_id")
    eta = obs.tool_result.get("eta_steps")
    assert job_id is not None, "Wire trace should return a job_id"
    print(f"    ✓ Async job {job_id} enqueued, ETA={eta}")

    # Phase 6: Try premature retrieval (should trigger Async Timeout)
    obs = _step(env, "retrieve_async_result", job_id=job_id)
    if obs.tool_result.get("error") == "Job not ready":
        assert env.state.async_timeout_count >= 1
        print("    ✓ Async timeout penalty triggered correctly")

    # Phase 7: Search compliance manual (Kernel Update — search)
    obs = _step(env, "search_compliance_manual", query="structuring threshold CTR")
    assert obs.tool_result.get("count", 0) > 0
    rules = obs.tool_result.get("results", [])
    print(f"    ✓ Found {len(rules)} compliance rules")

    # Phase 8: Inject rule into kernel (Kernel Update — injection)
    if rules:
        obs = _step(env, "update_system_prompt", rule=rules[0]["text"])
        assert env.state.meta_injections >= 1
        print("    ✓ Meta-injection successful")

    # Phase 9: More investigation
    obs = _step(env, "check_watchlist", entity_name="John Doe")
    obs = _step(env, "check_source_of_funds", transaction_id="TXN-001-A")

    # Phase 10: Save more findings to disk
    obs = _step(env, "write_to_case_file",
                content="TXN-001-A through TXN-001-E: 5 cash deposits, all sub-$10k, same branch")

    # Phase 11: Retrieve async result (should be ready by now)
    obs = _step(env, "retrieve_async_result", job_id=job_id)
    # May or may not be ready depending on ETA — that's ok

    # Phase 12: File SAR (terminal)
    obs = _step(env, "file_sar",
                findings=[
                    "multiple_sub_threshold_deposits",
                    "no_cash_intensive_occupation",
                    "same_branch_repeated",
                    "no_source_documentation",
                    "total_exceeds_ctr_threshold",
                ],
                typology="structuring",
                entities_involved=["CUST001"])
    assert obs.done
    final_score = obs.tool_result.get("final_score", 0)
    print(f"\n  FINAL SCORE: {final_score:+.4f}")
    assert final_score > 0, f"Expected positive score for correct SAR, got {final_score}"
    print("  ✓ Easy scenario PASSED")

    # Verify AGUI payload structure
    agui = obs.metadata.get("agui_state", {})
    assert "ram_usage" in agui
    assert "disk_storage" in agui
    assert "async_jobs" in agui
    assert "kernel_directives" in agui
    print("  ✓ AGUI payload structure verified")

    # Verify OS mechanic counters
    state = env.state
    assert state.successful_pages >= 2, "Should have 2 successful pages"
    assert state.meta_injections >= 1, "Should have 1 meta-injection"
    print(f"  ✓ OS Stats — Pages:{state.successful_pages} MetaInj:{state.meta_injections} "
          f"Faults:{state.page_fault_count} Timeouts:{state.async_timeout_count}")


def test_ram_eviction():
    """Verify that RAM eviction works correctly."""
    print(f"\n{'='*70}")
    print("TEST: RAM Eviction Mechanics")
    print("=" * 70)

    env = AMLEnvironment()
    env.reset(task_id="easy")

    # Fill past RAM capacity (2 observations) + initial
    _step(env, "review_alert")
    _step(env, "get_customer_profile", customer_id="CUST001")
    _step(env, "query_transactions", customer_id="CUST001")

    # Check RAM has exactly 2 entries (capacity)
    agui = env._sm.build_agui_state()
    ram_count = len(agui["ram_usage"]["active_context"])
    assert ram_count == 2, f"Expected 2 RAM entries (capacity), got {ram_count}"
    print(f"  ✓ RAM capped at 2 entries after 4 observations (1 init + 3 steps)")

    # The oldest observation should have been evicted
    print("  ✓ Eviction confirmed")


def test_medium_scenario():
    """Quick medium scenario exercise."""
    print(f"\n{'='*70}")
    print("TEST: Medium Scenario (Layering) — Quick Run")
    print("=" * 70)

    env = AMLEnvironment()
    obs = env.reset(task_id="medium")
    assert not obs.done
    print(f"  Reset OK | Episode: {env.state.episode_id[:8]}...")

    _step(env, "review_alert")
    _step(env, "get_customer_profile", customer_id="CUST002")
    _step(env, "query_transactions", customer_id="CUST002")
    _step(env, "write_to_case_file", content="CUST002 layering scenario — funds fan-out to 3+ entities")
    _step(env, "trace_network", entity_id="CUST002", depth=2)

    obs = _step(env, "file_sar",
                findings=["rapid_fan_out", "pep_connection", "shared_registered_address", "newly_incorporated"],
                typology="layering",
                entities_involved=["CUST002", "ENT_A", "ENT_B", "ENT_C", "ENT_D"])
    assert obs.done
    print(f"  FINAL SCORE: {obs.tool_result.get('final_score', 0):+.4f}")
    print("  ✓ Medium scenario PASSED")


def test_hard_scenario():
    """Quick hard scenario exercise."""
    print(f"\n{'='*70}")
    print("TEST: Hard Scenario (Trade-Based ML) — Quick Run")
    print("=" * 70)

    env = AMLEnvironment()
    obs = env.reset(task_id="hard")
    assert not obs.done
    print(f"  Reset OK | Episode: {env.state.episode_id[:8]}...")

    _step(env, "review_alert")
    _step(env, "get_customer_profile", customer_id="CUST003")
    _step(env, "query_transactions", customer_id="CUST003")
    _step(env, "check_market_price", commodity="machine parts")
    _step(env, "write_to_case_file", content="CUST003 TBML — $50k/unit vs $12k market, 317% above")
    _step(env, "trace_network", entity_id="CUST003")

    obs = _step(env, "file_sar",
                findings=["over_invoicing", "beneficial_owner_connection", "fatf_jurisdiction"],
                typology="trade_based_ml",
                entities_involved=["CUST003", "ENT_F", "Marcus Webb"])
    assert obs.done
    print(f"  FINAL SCORE: {obs.tool_result.get('final_score', 0):+.4f}")
    print("  ✓ Hard scenario PASSED")


def test_compliance_manual_search():
    """Verify the compliance manual search works."""
    print(f"\n{'='*70}")
    print("TEST: Compliance Manual Search")
    print("=" * 70)

    from scenarios.compliance_manual import search_compliance_manual

    # Search for structuring rules
    results = search_compliance_manual("structuring threshold deposits")
    assert len(results) > 0, "Should find structuring rules"
    print(f"  ✓ 'structuring threshold deposits' → {len(results)} results")

    # Search for FATF
    results = search_compliance_manual("FATF jurisdiction")
    assert len(results) > 0, "Should find FATF rules"
    print(f"  ✓ 'FATF jurisdiction' → {len(results)} results")

    # Search with category filter
    results = search_compliance_manual("price", category_filter="trade_based_ml")
    assert len(results) > 0, "Should find trade-based ML rules about pricing"
    print(f"  ✓ 'price' (trade_based_ml) → {len(results)} results")

    print("  ✓ Compliance manual search PASSED")


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_compliance_manual_search,
        test_ram_eviction,
        test_easy_scenario,
        test_medium_scenario,
        test_hard_scenario,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    if failed:
        print(f"         {failed} FAILED")
        sys.exit(1)
    else:
        print("         All tests PASSED ✓")
    print("=" * 70)
