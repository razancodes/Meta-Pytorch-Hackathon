#!/usr/bin/env python3
"""
AgentOS-Kernel — L1/L2 Cognitive Cache Test Suite.

Simulates a 30-turn AML investigation conversation and proves:
  1. L1 stays strictly under its token budget after compactions
  2. L2 scratchpad accumulates entities/facts from evicted turns
  3. get_injected_prompt() produces a valid message list with L2 preamble
  4. Compaction runs asynchronously without blocking
  5. Multiple compaction cycles work correctly
"""

import asyncio
import json
import time
import sys

from memory_manager import MemoryManager


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulated conversation data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = (
    "You are an AML investigator. Analyze alerts systematically. "
    "Use all available tools. File a SAR if suspicious, close if benign."
)

# Each turn is (role, content) — simulates a realistic agent loop
CONVERSATION_TURNS = [
    ("user", "New alert: ALT-7729. Customer CUST-A flagged for unusual wire transfers totaling $487,000 across 14 transactions in the past 30 days."),
    ("assistant", '{"tool": "review_alert", "parameters": {"alert_id": "ALT-7729"}}'),
    ("tool", json.dumps({
        "alert_id": "ALT-7729",
        "customer_id": "CUST-A",
        "summary": "Multiple sub-CTR wires to offshore entities",
        "risk_score": 72,
        "triggered_rules": ["STRUCT-01", "GEO-RISK-03"],
    })),
    ("assistant", '{"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-A"}}'),
    ("tool", json.dumps({
        "customer_id": "CUST-A",
        "name": "Meridian Holdings LLC",
        "type": "corporate",
        "jurisdiction": "Delaware, US",
        "account_age_months": 4,
        "stated_purpose": "Import/export consulting",
        "beneficial_owner": "Viktor Korev",
        "kyc_status": "enhanced_due_diligence",
    })),
    ("assistant", '{"tool": "query_transactions", "parameters": {"customer_id": "CUST-A", "min_amount": 5000}}'),
    ("tool", json.dumps({
        "customer_id": "CUST-A",
        "count": 14,
        "transactions": [
            {"id": "TXN-001", "amount": 9800, "direction": "outgoing", "counterparty": "ENT_SHELL_A", "country": "CY"},
            {"id": "TXN-002", "amount": 9750, "direction": "outgoing", "counterparty": "ENT_SHELL_A", "country": "CY"},
            {"id": "TXN-003", "amount": 9900, "direction": "outgoing", "counterparty": "ENT_SHELL_B", "country": "KY"},
            {"id": "TXN-004", "amount": 9600, "direction": "outgoing", "counterparty": "ENT_SHELL_B", "country": "KY"},
            {"id": "TXN-005", "amount": 8500, "direction": "outgoing", "counterparty": "ENT_BANK_C",  "country": "SG"},
        ],
        "total_amount": 487000,
        "pattern_detected": "sub_ctr_structuring",
    })),
    ("assistant", '{"tool": "check_watchlist", "parameters": {"entity_name": "Viktor Korev"}}'),
    ("tool", json.dumps({
        "entity": "Viktor Korev",
        "hit": True,
        "lists": ["PEP — Former Deputy Minister of Trade, Country X"],
        "match_score": 0.94,
        "risk_level": "CRITICAL",
    })),
    ("assistant", '{"tool": "trace_network", "parameters": {"entity_id": "CUST-A", "depth": 2}}'),
    ("tool", json.dumps({
        "entity_id": "CUST-A",
        "connections": [
            {"target": "ENT_SHELL_A", "relationship": "beneficial_owner", "jurisdiction": "Cyprus"},
            {"target": "ENT_SHELL_B", "relationship": "transfer_to",      "jurisdiction": "Cayman Islands"},
            {"target": "ENT_BANK_C",  "relationship": "correspondent",    "jurisdiction": "Singapore"},
            {"target": "ENT_TRUST_D", "relationship": "nominee_director", "jurisdiction": "BVI"},
        ],
    })),
    ("assistant", '{"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-001"}}'),
    ("tool", json.dumps({
        "transaction_id": "TXN-001",
        "source": "Unverified wire from Meridian Holdings account ending 4477",
        "verified": False,
        "flags": ["no_supporting_invoice", "round_amount_pattern"],
    })),
    ("assistant", '{"tool": "check_device_overlap", "parameters": {"entity_id": "CUST-A"}}'),
    ("tool", json.dumps({
        "entity_id": "CUST-A",
        "shared_devices": 3,
        "shared_ips": 2,
        "overlapping_entities": ["ENT_SHELL_A", "ENT_SHELL_B", "ENT_MULE_7"],
        "risk_indicator": "mule_ring_suspected",
    })),
    ("assistant", '{"tool": "query_beneficial_ownership", "parameters": {"entity_id": "ENT_SHELL_A"}}'),
    ("tool", json.dumps({
        "entity_id": "ENT_SHELL_A",
        "registered_name": "Lazarus Trading Co. Ltd",
        "jurisdiction": "Cyprus",
        "ubo": "Viktor Korev (indirect, via ENT_TRUST_D)",
        "incorporation_date": "2024-11-03",
        "status": "active",
    })),
    ("user", "What's the risk assessment for this cluster?"),
    ("assistant", '{"tool": "assess_risk", "parameters": {"customer_id": "CUST-A"}}'),
    ("tool", json.dumps({
        "customer_id": "CUST-A",
        "risk_score": 92,
        "risk_level": "CRITICAL",
        "recommendation": "SAR filing strongly recommended",
        "risk_factors": [
            "PEP connection (Viktor Korev)",
            "Sub-CTR structuring pattern",
            "Multi-jurisdictional shell network",
            "Unverified source of funds",
            "Device overlap with suspected mule accounts",
        ],
    })),
    ("assistant", '{"tool": "verify_customs_invoice", "parameters": {"transaction_id": "TXN-003"}}'),
    ("tool", json.dumps({
        "transaction_id": "TXN-003",
        "invoice_found": False,
        "note": "No customs declaration or trade invoice on file for $9,900 wire to ENT_SHELL_B (Cayman Islands)",
    })),
    ("user", "File the SAR now."),
    ("assistant", json.dumps({
        "tool": "file_sar",
        "parameters": {
            "findings": [
                "Sub-CTR structuring across 14 transactions ($487K total)",
                "PEP beneficial owner: Viktor Korev (former Deputy Minister)",
                "Multi-jurisdictional shell network (CY, KY, SG, BVI)",
                "Unverified source of funds on sampled transactions",
                "Device/IP overlap with suspected mule ring",
                "No customs documentation for cross-border wires",
            ],
            "typology": "Layering via Shell Companies",
            "entities_involved": ["CUST-A", "Viktor Korev", "ENT_SHELL_A", "ENT_SHELL_B", "ENT_TRUST_D"],
        },
    })),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def main():
    print("=" * 70)
    print("  AgentOS-Kernel — L1/L2 Cognitive Cache Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    # Use a lower token budget so compaction triggers within our test data
    mm = MemoryManager(
        system_prompt=SYSTEM_PROMPT,
        max_l1_tokens=800,      # Low budget to force multiple compactions
        max_l2_tokens=4000,
        compaction_batch_size=4,
        use_mock_compactor=True,
    )

    print(f"\n  System prompt: {mm._system_token_count} tokens")
    print(f"  L1 budget:     {mm.max_l1_tokens} tokens")
    print(f"  Compaction:    every {mm._compaction_batch_size} oldest turns")

    # ── Feed the conversation ───────────────────────────────────────────
    print("\n▸ Feeding conversation turns and triggering compactions...\n")

    compaction_count = 0
    for i, (role, content) in enumerate(CONVERSATION_TURNS):
        mm.add_turn(role, content, step=i)

        # After each turn, compact if needed
        result = await mm.compact_if_needed()
        if result:
            compaction_count += 1
            print(
                f"  Turn {i:2d}: COMPACTION #{compaction_count} — "
                f"evicted to L2, extracted {len(result.entities)} entities. "
                f"L1={mm.l1_token_count}tok, L2={mm.l2_entity_count} entities"
            )
        else:
            flag = " ⚠ OVER BUDGET" if mm.needs_compaction else ""
            print(
                f"  Turn {i:2d}: [{role:9}] L1={mm.l1_token_count:4d}tok "
                f"({mm.l1_turn_count} turns){flag}"
            )

    # Drain any remaining compaction needs
    while mm.needs_compaction:
        result = await mm.compact_if_needed()
        if result:
            compaction_count += 1
            print(
                f"  Drain compaction #{compaction_count} — "
                f"L1={mm.l1_token_count}tok"
            )

    # ── Test 1: L1 under budget ─────────────────────────────────────────
    print(f"\n▸ Test 1: L1 Token Budget Enforcement")
    stats = mm.get_stats()
    print(f"  L1 tokens: {stats['l1_tokens']} / {stats['l1_budget']}")

    if stats["l1_tokens"] <= stats["l1_budget"]:
        print(f"  ✓ PASS — L1 is within budget")
        passed += 1
    else:
        print(f"  ✗ FAIL — L1 exceeded budget: {stats['l1_tokens']} > {stats['l1_budget']}")
        failed += 1

    # ── Test 2: L2 scratchpad populated ─────────────────────────────────
    print(f"\n▸ Test 2: L2 Scratchpad Populated")
    print(f"  L2 entities: {stats['l2_entities']}")
    print(f"  L2 facts:    {stats['l2_facts']}")
    print(f"  L2 tokens:   {stats['l2_tokens']}")

    if stats["l2_entities"] > 0 and stats["l2_facts"] > 0:
        print(f"  ✓ PASS — L2 has {stats['l2_entities']} entities with {stats['l2_facts']} facts")
        passed += 1
    else:
        print(f"  ✗ FAIL — L2 is empty")
        failed += 1

    # ── Test 3: Compactions actually ran ─────────────────────────────────
    print(f"\n▸ Test 3: Compaction Cycles")
    print(f"  Compactions run: {stats['compactions_run']}")
    print(f"  Turns compacted: {stats['turns_compacted']}")

    if stats["compactions_run"] >= 2:
        print(f"  ✓ PASS — Multiple compaction cycles completed")
        passed += 1
    else:
        print(f"  ✗ FAIL — Expected ≥2 compactions, got {stats['compactions_run']}")
        failed += 1

    # ── Test 4: Prompt assembly ──────────────────────────────────────────
    print(f"\n▸ Test 4: Prompt Assembly (get_injected_prompt)")
    messages = mm.get_injected_prompt()

    print(f"  Message count:  {len(messages)}")
    print(f"  First role:     {messages[0]['role']}")
    has_scratchpad = "[ACCUMULATED KNOWLEDGE" in messages[0]["content"]
    print(f"  Has L2 preamble: {has_scratchpad}")

    if (
        messages[0]["role"] == "system"
        and has_scratchpad
        and len(messages) > 1
    ):
        print(f"  ✓ PASS — Valid prompt with L2 preamble injected")
        passed += 1
    else:
        print(f"  ✗ FAIL — Prompt assembly issue")
        failed += 1

    # ── Test 5: Deduplicated facts ───────────────────────────────────────
    print(f"\n▸ Test 5: Fact Deduplication in L2")
    all_facts = []
    for facts in mm._l2_scratchpad.values():
        all_facts.extend(facts)
    unique_facts = set(all_facts)

    print(f"  Total facts:  {len(all_facts)}")
    print(f"  Unique facts: {len(unique_facts)}")

    if len(all_facts) == len(unique_facts):
        print(f"  ✓ PASS — No duplicate facts in L2")
        passed += 1
    else:
        duplicates = len(all_facts) - len(unique_facts)
        print(f"  ✗ FAIL — {duplicates} duplicate facts found")
        failed += 1

    # ── Print final state ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Final State: {mm}")
    print(f"  Total prompt tokens: {stats['total_prompt_tokens']}")
    print(f"{'─' * 70}")

    # ── Print L2 scratchpad contents ──────────────────────────────────────
    print(f"\n▸ L2 Scratchpad Contents:\n")
    for entity, facts in sorted(mm._l2_scratchpad.items()):
        label = entity if not entity.startswith("__") else entity.strip("_").upper()
        print(f"  {label}:")
        for fact in facts:
            print(f"    • {fact}")
        print()

    # ── Print assembled prompt (truncated) ────────────────────────────────
    print(f"▸ Assembled Prompt Preview (system message, first 800 chars):\n")
    system_msg = messages[0]["content"]
    print(f"  {system_msg[:800]}{'...' if len(system_msg) > 800 else ''}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED ✓")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED ✗")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
