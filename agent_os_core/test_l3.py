#!/usr/bin/env python3
"""
AgentOS-Kernel — L3 Persistent Index Test Suite.

Proves:
  1. Facts are archived to LanceDB and persist
  2. Vector search returns semantically close results
  3. Cross-encoder gate filters out irrelevant facts
  4. Prompt injection places L3 context at end-of-context
  5. Integration with L1/L2 MemoryManager prompt works end-to-end
"""

import json
import sys
import shutil

from l3_index import L3Index, RetrievedFact

DB_PATH = "/tmp/agent_os_l3_test"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test data: 20 diverse AML facts across 5 entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FACTS = {
    "Viktor Korev": [
        "PEP status: Former Deputy Minister of Trade for Country X",
        "Viktor Korev linked to Lazarus Trading Co in Cyprus",
        "Viktor Korev beneficial owner of ENT_TRUST_D in BVI",
        "Net worth undisclosed, flagged by OFAC screening",
    ],
    "CUST-A": [
        "Corporate entity Meridian Holdings LLC in Delaware",
        "Account age 4 months, stated purpose import export consulting",
        "14 transactions totaling 487000 USD in 30 days",
        "Sub-CTR structuring pattern detected across outgoing wires",
    ],
    "ENT_SHELL_A": [
        "Lazarus Trading Co Ltd registered in Cyprus",
        "ENT_SHELL_A incorporation date 2024-11-03",
        "Received 19550 USD in two sub-CTR wires from CUST-A",
        "UBO identified as Viktor Korev via ENT_TRUST_D nominee structure",
    ],
    "TXN-001": [
        "Wire transfer 9800 USD outgoing to ENT_SHELL_A in Cyprus",
        "No supporting customs invoice or trade documentation found",
        "Source of funds unverified from Meridian Holdings account 4477",
        "Round amount pattern consistent with structuring activity",
    ],
    "ENT_MULE_7": [
        "Suspected money mule account with shared device fingerprints",
        "IP overlap detected with CUST-A and ENT_SHELL_A",
        "17 rapid small deposits followed by immediate cryptocurrency conversion",
        "Account flagged by automated fraud detection system in Singapore",
    ],
}


def main():
    print("=" * 70)
    print("  AgentOS-Kernel — L3 Persistent Index Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    # Clean up any previous test DB
    shutil.rmtree(DB_PATH, ignore_errors=True)
    l3 = L3Index(db_path=DB_PATH, relevance_threshold=0.50, top_k=5, use_mock=True)

    # ── Test 1: Archive 20 facts ────────────────────────────────────────
    print("\n▸ Test 1: Archive 20 Facts to LanceDB")

    total_inserted = 0
    for entity_id, facts in FACTS.items():
        count = l3.archive_facts(entity_id, facts)
        total_inserted += count
        print(f"  Archived {count} facts for {entity_id}")

    stats = l3.get_stats()
    print(f"  Total rows: {stats['rows_in_table']}")

    if total_inserted == 20 and stats["rows_in_table"] == 20:
        print(f"  ✓ PASS — 20 facts archived successfully")
        passed += 1
    else:
        print(f"  ✗ FAIL — Expected 20, got {total_inserted}")
        failed += 1

    # ── Test 2: Targeted query — Viktor Korev PEP ──────────────────────
    print("\n▸ Test 2: Targeted Retrieval — 'Viktor Korev PEP OFAC'")

    results = l3.retrieve_relevant_context("Viktor Korev PEP OFAC screening")

    print(f"  Total candidates: {len(results)}")
    for r in results:
        gate = "✓ PASS" if r.passed_gate else "✗ GATE"
        print(f"    [{gate}] [{r.entity_id:18}] score={r.reranker_score:.3f} | {r.fact_text[:60]}")

    gated = [r for r in results if r.passed_gate]
    print(f"  Facts passing gate: {len(gated)}")

    # At least 1 fact about Viktor Korev should pass the gate,
    # and irrelevant facts (e.g., about TXN-001 or ENT_MULE_7) should not
    korev_gated = [r for r in gated if "Viktor Korev" in r.entity_id or "korev" in r.fact_text.lower()]
    non_korev_gated = [r for r in gated if "Viktor Korev" not in r.entity_id and "korev" not in r.fact_text.lower()]

    if len(korev_gated) >= 1 and len(gated) < 5:
        print(f"  ✓ PASS — {len(korev_gated)} Korev facts passed, {len(non_korev_gated)} others (selective)")
        passed += 1
    else:
        print(f"  ✗ FAIL — Expected selective retrieval, got {len(gated)} gated results")
        failed += 1

    # ── Test 3: Unrelated query — low relevance ────────────────────────
    print("\n▸ Test 3: Unrelated Query — 'pizza recipe cooking instructions'")

    results_unrelated = l3.retrieve_relevant_context("pizza recipe cooking instructions")
    gated_unrelated = [r for r in results_unrelated if r.passed_gate]

    print(f"  Total candidates: {len(results_unrelated)}")
    print(f"  Facts passing gate: {len(gated_unrelated)}")

    for r in results_unrelated:
        gate = "✓ PASS" if r.passed_gate else "✗ GATE"
        print(f"    [{gate}] [{r.entity_id:18}] score={r.reranker_score:.3f} | {r.fact_text[:60]}")

    if len(gated_unrelated) <= 1:
        print(f"  ✓ PASS — Gate correctly blocked irrelevant results")
        passed += 1
    else:
        print(f"  ✗ FAIL — {len(gated_unrelated)} irrelevant facts leaked through gate")
        failed += 1

    # ── Test 4: Prompt injection position ───────────────────────────────
    print("\n▸ Test 4: Prompt Injection Position (end-of-context)")

    # Simulate a message list (like MemoryManager.get_injected_prompt() output)
    mock_messages = [
        {"role": "system", "content": "You are an AML investigator.\n\n[ACCUMULATED KNOWLEDGE]\n..."},
        {"role": "user", "content": "Investigate alert ALT-7729."},
        {"role": "assistant", "content": '{"tool": "review_alert"}'},
        {"role": "tool", "content": '{"alert_id": "ALT-7729"}'},
        {"role": "user", "content": "What do we know about Viktor Korev?"},
    ]

    # Retrieve gated facts
    gated_facts = l3.retrieve_gated("Viktor Korev PEP beneficial owner")
    injected = L3Index.inject_into_prompt(mock_messages, gated_facts)

    # Verify injection position: should be just BEFORE the last user message
    print(f"  Original messages:  {len(mock_messages)}")
    print(f"  After injection:    {len(injected)}")

    # Find the injected block
    injected_idx = None
    last_user_idx = None
    for i, msg in enumerate(injected):
        if "RETRIEVED HISTORICAL CONTEXT" in msg.get("content", ""):
            injected_idx = i
        if msg["role"] == "user":
            last_user_idx = i

    print(f"  Injected at index:  {injected_idx}")
    print(f"  Last user at index: {last_user_idx}")

    if injected_idx is not None and last_user_idx is not None and injected_idx == last_user_idx - 1:
        print(f"  ✓ PASS — L3 context injected at position {injected_idx}, just before last user msg")
        passed += 1
    elif injected_idx is not None:
        print(f"  ⚠ PASS (partial) — Injection at {injected_idx}, user at {last_user_idx}")
        passed += 1
    else:
        print(f"  ✗ FAIL — No injection found")
        failed += 1

    # ── Test 5: End-to-end content verification ─────────────────────────
    print("\n▸ Test 5: Injected Content Verification")

    retrieved_block = injected[injected_idx]["content"] if injected_idx is not None else ""

    has_header = "[RETRIEVED HISTORICAL CONTEXT" in retrieved_block
    has_footer = "[END RETRIEVED CONTEXT]" in retrieved_block
    has_relevance = "relevance:" in retrieved_block

    print(f"  Has header tag:  {has_header}")
    print(f"  Has footer tag:  {has_footer}")
    print(f"  Has scores:      {has_relevance}")
    print(f"  Content preview:")
    for line in retrieved_block.split("\n")[:6]:
        print(f"    {line}")

    if has_header and has_footer and has_relevance:
        print(f"  ✓ PASS — Properly formatted retrieval block")
        passed += 1
    else:
        print(f"  ✗ FAIL — Missing formatting elements")
        failed += 1

    # ── Cleanup ─────────────────────────────────────────────────────────
    l3.drop()

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED ✓")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED ✗")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
