#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — Smoke Tests (Procedural Generator Edition).

Exercises:
  1. Procedural Generator: unique IDs per episode, all 3 typologies × 3 difficulties.
  2. Anti-Memorization: two successive resets produce different entity IDs.
  3. Compliance Manual: keyword search.
  4. Noise Injection: scales with difficulty.
  5. Full Easy Episode: structuring with all OS mechanics.
  6. Full Medium Episode: layering with all OS mechanics.
  7. Full Hard Episode: trade-based ML with all OS mechanics.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AMLAction
from scenarios.procedural_generator import ScenarioGenerator, GeneratedScenario
from scenarios.compliance_manual import search_compliance_manual
from server.aml_environment import AMLEnvironment

PASS = 0
TOTAL = 0

def test(name: str):
    global TOTAL
    TOTAL += 1
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

def ok(msg: str):
    global PASS
    PASS += 1
    print(f"  ✓ {msg}")

def fail(msg: str):
    print(f"  ✗ FAIL: {msg}")
    sys.exit(1)


# Helper: run a sequence of tools through the environment
def run_tools(env, tools):
    """Execute a list of (tool_name, params) through env.step(). Returns last obs."""
    obs = None
    for tool_name, params in tools:
        action = AMLAction(tool=tool_name, parameters=params)
        obs = env.step(action)
        r = obs.reward or 0.0
        ram = obs.metadata.get("agui_state", {}).get("ram_usage", {}).get("capacity", "?")
        done_lbl = "DONE" if obs.done else "ok"
        print(f"  Step | {tool_name:<30} | R={r:+.4f} | RAM={ram} | {done_lbl}")
        if obs.done:
            break
    return obs


# ===================================================================== #
# 1. Procedural Generator — All Typologies                              #
# ===================================================================== #
test("Procedural Generator — All 9 Combos")
gen = ScenarioGenerator(seed=42)

for typo in ["structuring", "layering", "trade_based_ml"]:
    for diff in ["easy", "medium", "hard"]:
        sc = gen.generate(difficulty=diff, typology=typo)
        assert isinstance(sc, GeneratedScenario)
        assert sc.initial_alert
        assert sc.customer_profiles
        assert sc.transactions
        assert sc.watchlist_results
        assert sc.network_graph
        assert sc.source_of_funds
        assert sc.ground_truth
        assert sc.ground_truth["typology"] == typo
        assert sc.ground_truth["correct_decision"] == "file_sar"
        assert len(sc.ground_truth["key_entities"]) >= 1
        assert len(sc.ground_truth["key_findings"]) >= 3

        print(f"  ✓ {diff}/{typo}: alert={sc.initial_alert['alert_id']}, "
              f"entities={len(sc.customer_profiles)}, txns={len(sc.transactions)}, "
              f"GT_keys={sc.ground_truth['key_entities'][:2]}...")

ok("All 9 typology/difficulty combos generate valid scenarios")


# ===================================================================== #
# 2. Anti-Memorization                                                   #
# ===================================================================== #
test("Anti-Memorization — Unique IDs per Episode")

gen_a = ScenarioGenerator()
gen_b = ScenarioGenerator()

sc_a = gen_a.generate(difficulty="easy", typology="structuring")
sc_b = gen_b.generate(difficulty="easy", typology="structuring")

id_a = sc_a.initial_alert["customer_id"]
id_b = sc_b.initial_alert["customer_id"]
txn_ids_a = set(t["transaction_id"] for t in sc_a.transactions)
txn_ids_b = set(t["transaction_id"] for t in sc_b.transactions)

print(f"  Episode A: customer={id_a}, txns={len(txn_ids_a)}")
print(f"  Episode B: customer={id_b}, txns={len(txn_ids_b)}")
assert txn_ids_a != txn_ids_b, "Transaction IDs should differ!"
ok(f"Unique IDs confirmed across episodes")


# ===================================================================== #
# 3. Compliance Manual                                                   #
# ===================================================================== #
test("Compliance Manual Search")

for query, min_hits in [
    ("structuring threshold deposits", 2),
    ("FATF jurisdiction", 1),
    ("price trade_based_ml", 1),
]:
    results = search_compliance_manual(query)
    assert len(results) >= min_hits
    print(f"  ✓ '{query}' → {len(results)} results")
ok("Compliance manual PASSED")


# ===================================================================== #
# 4. Noise Injection                                                     #
# ===================================================================== #
test("Noise Injection — Scales with Difficulty")
gen = ScenarioGenerator(seed=99)

for diff in ["easy", "medium", "hard"]:
    sc = gen.generate(difficulty=diff, typology="structuring")
    print(f"  {diff}: profiles={len(sc.customer_profiles)}, txns={len(sc.transactions)}")

sc_e = gen.generate(difficulty="easy", typology="layering")
sc_h = gen.generate(difficulty="hard", typology="layering")
assert len(sc_h.customer_profiles) >= len(sc_e.customer_profiles)
assert len(sc_h.transactions) >= len(sc_e.transactions)
ok("Noise scales with difficulty")


# ===================================================================== #
# 5. Full Episode — Structuring                                         #
# ===================================================================== #
test("Full Episode — Procedural Structuring (Easy)")

env = AMLEnvironment()
init_obs = env.reset(task_id="easy", seed=42)
subject_id = init_obs.tool_result["alert"]["customer_id"]
print(f"  Reset OK | Subject: {subject_id}")

obs = run_tools(env, [
    ("review_alert", {}),
    ("get_customer_profile", {"customer_id": subject_id}),
    ("write_to_case_file", {"note": "Structuring suspect observed"}),
    ("query_transactions", {"customer_id": subject_id}),
    ("search_compliance_manual", {"query": "structuring"}),
    ("update_system_prompt", {"directive": "Apply structuring detection rules"}),
    ("check_watchlist", {"entity": subject_id}),
])

# File SAR with ground truth
gt = env._current_scenario.ground_truth
obs = env.step(AMLAction(
    tool="file_sar",
    parameters={
        "typology": gt["typology"],
        "entities_involved": gt["key_entities"],
        "findings": gt["key_findings"],
    },
))
print(f"\n  FINAL SCORE: {obs.reward:+.4f}")
assert obs.done
assert obs.reward > 0.5, f"Expected >0.5, got {obs.reward}"
ok(f"Procedural structuring PASSED (score={obs.reward:+.4f})")


# ===================================================================== #
# 6. Full Episode — Layering                                            #
# ===================================================================== #
test("Full Episode — Procedural Layering (Medium)")

env2 = AMLEnvironment()
init_obs2 = env2.reset(task_id="medium", seed=77)
subject_id2 = init_obs2.tool_result["alert"]["customer_id"]
print(f"  Reset OK | Subject: {subject_id2}")

obs2 = run_tools(env2, [
    ("review_alert", {}),
    ("get_customer_profile", {"customer_id": subject_id2}),
    ("query_transactions", {"customer_id": subject_id2}),
    ("trace_network", {"customer_id": subject_id2}),
    ("write_to_case_file", {"note": "Fan-out pattern detected"}),
])

gt2 = env2._current_scenario.ground_truth
obs2 = env2.step(AMLAction(
    tool="file_sar",
    parameters={
        "typology": gt2["typology"],
        "entities_involved": gt2["key_entities"],
        "findings": gt2["key_findings"],
    },
))
print(f"\n  FINAL SCORE: {obs2.reward:+.4f}")
assert obs2.done and obs2.reward > 0.5
ok(f"Procedural layering PASSED (score={obs2.reward:+.4f})")


# ===================================================================== #
# 7. Full Episode — Trade-Based ML                                      #
# ===================================================================== #
test("Full Episode — Procedural Trade-Based ML (Hard)")

env3 = AMLEnvironment()
init_obs3 = env3.reset(task_id="hard", seed=123)
subject_id3 = init_obs3.tool_result["alert"]["customer_id"]
print(f"  Reset OK | Subject: {subject_id3}")

# Get commodity from procedural market_data
mdata = env3._current_scenario.market_data
commodity_key = list(mdata.keys())[0] if mdata else None

tools3 = [
    ("review_alert", {}),
    ("get_customer_profile", {"customer_id": subject_id3}),
    ("query_transactions", {"customer_id": subject_id3}),
    ("trace_network", {"customer_id": subject_id3}),
    ("write_to_case_file", {"note": "Over-invoicing pattern detected"}),
]
if commodity_key:
    tools3.append(("check_market_price", {"commodity": commodity_key}))

obs3 = run_tools(env3, tools3)

gt3 = env3._current_scenario.ground_truth
obs3 = env3.step(AMLAction(
    tool="file_sar",
    parameters={
        "typology": gt3["typology"],
        "entities_involved": gt3["key_entities"],
        "findings": gt3["key_findings"],
    },
))
print(f"\n  FINAL SCORE: {obs3.reward:+.4f}")
assert obs3.done and obs3.reward > 0.3
ok(f"Procedural TBML PASSED (score={obs3.reward:+.4f})")


# ===================================================================== #
# 8. PLR Curriculum Engine                                               #
# ===================================================================== #
test("PLR Curriculum Engine — Buffer, Sampling, Metrics")

from curriculum.plr_engine import PLREngine, ScenarioRecord
from curriculum.oracle import proxy_regret

plr = PLREngine(buffer_size=50, temperature=0.1, staleness_threshold=10)

# Verify proxy regret
assert proxy_regret(0.3) == 0.71, f"Expected 0.71, got {proxy_regret(0.3)}"
assert proxy_regret(1.01) == 0.0, f"Expected 0.0, got {proxy_regret(1.01)}"
assert proxy_regret(1.5) == 0.0, f"Expected 0.0, got {proxy_regret(1.5)}"
print(f"  ✓ proxy_regret(0.3)={proxy_regret(0.3):.2f}, proxy_regret(1.01)={proxy_regret(1.01):.2f}")

# Fill buffer with scenarios of varying scores
difficulties = ["easy", "medium", "hard"]
typologies = ["structuring", "layering", "trade_based_ml"]

for i, (d, t) in enumerate([
    ("easy", "structuring"), ("medium", "layering"), ("hard", "trade_based_ml"),
    ("easy", "layering"), ("medium", "structuring"),
]):
    # Simulate decreasing scores (harder = lower score)
    score = 0.8 - (i * 0.15)
    plr.update(f"test_ep_{i}", d, t, score)
    print(f"  ✓ Added {d}/{t} score={score:+.2f} regret={proxy_regret(score):.2f}")

assert len(plr.buffer) == 5, f"Expected 5 buffer entries, got {len(plr.buffer)}"
print(f"  ✓ Buffer size: {len(plr.buffer)}")

# Verify sampling returns valid difficulty/typology pairs
for _ in range(20):
    d, t = plr.sample_scenario(difficulties, typologies)
    assert d in difficulties, f"Invalid difficulty: {d}"
    assert t in typologies, f"Invalid typology: {t}"
print(f"  ✓ 20 samples all returned valid (difficulty, typology) pairs")

# Verify WandB metrics
metrics = plr.get_wandb_metrics()
assert "curriculum/mean_regret" in metrics
assert "curriculum/buffer_diversity" in metrics
assert metrics["curriculum/buffer_size"] == 5
assert metrics["curriculum/mean_regret"] > 0
print(f"  ✓ WandB metrics: mean_regret={metrics['curriculum/mean_regret']:.3f}, "
      f"diversity={metrics['curriculum/buffer_diversity']:.2f}")

# Verify AGUI state output
state = plr.get_current_state()
assert state["enabled"] is True
assert state["buffer_size"] == 5
assert state["mean_regret"] > 0
print(f"  ✓ AGUI state: enabled={state['enabled']}, difficulty={state['difficulty_label']}")

ok("PLR Curriculum Engine PASSED")


# ===================================================================== #
# Results                                                                #
# ===================================================================== #
print(f"\n{'='*70}")
print(f"RESULTS: {PASS}/{TOTAL} tests passed")
print(f"         {'All tests PASSED ✓' if PASS == TOTAL else 'SOME TESTS FAILED ✗'}")
print(f"{'='*70}\n")
sys.exit(0 if PASS == TOTAL else 1)

