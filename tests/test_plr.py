#!/usr/bin/env python3
"""Quick standalone test for PLR curriculum engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum.plr_engine import PLREngine, ScenarioRecord
from curriculum.oracle import proxy_regret


def main():
    print("=== PLR Curriculum Engine Smoke Test ===\n")

    # Test proxy regret
    assert proxy_regret(0.3) == 0.71
    assert proxy_regret(1.01) == 0.0
    assert proxy_regret(1.5) == 0.0
    print("  PASS: proxy_regret()")

    plr = PLREngine(buffer_size=50, temperature=0.1, staleness_threshold=10)

    # Fill buffer
    for i, (d, t) in enumerate([
        ("easy", "structuring"), ("medium", "layering"), ("hard", "trade_based_ml"),
        ("easy", "layering"), ("medium", "structuring"),
    ]):
        score = 0.8 - (i * 0.15)
        plr.update(f"test_ep_{i}", d, t, score)
        regret = proxy_regret(score)
        print(f"  Added {d}/{t} score={score:+.2f} regret={regret:.2f}")

    assert len(plr.buffer) == 5
    print(f"  PASS: Buffer size = {len(plr.buffer)}")

    # Test sampling
    difficulties = ["easy", "medium", "hard"]
    typologies = ["structuring", "layering", "trade_based_ml"]
    for _ in range(20):
        d, t = plr.sample_scenario(difficulties, typologies)
        assert d in difficulties, f"Invalid difficulty: {d}"
        assert t in typologies, f"Invalid typology: {t}"
    print("  PASS: 20 valid samples")

    # Test WandB metrics
    m = plr.get_wandb_metrics()
    assert "curriculum/mean_regret" in m
    assert m["curriculum/buffer_size"] == 5
    assert m["curriculum/mean_regret"] > 0
    mr = m["curriculum/mean_regret"]
    print(f"  PASS: WandB metrics (mean_regret={mr:.3f})")

    # Test AGUI state
    state = plr.get_current_state()
    assert state["enabled"] is True
    assert state["buffer_size"] == 5
    dl = state["difficulty_label"]
    print(f"  PASS: AGUI state (difficulty={dl})")

    # Test save/load roundtrip
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plr_test_buffer.json")
    plr.save(path)
    plr2 = PLREngine(buffer_size=50)
    plr2.load(path)
    assert len(plr2.buffer) == 5
    assert plr2.iteration == plr.iteration
    os.remove(path)
    print("  PASS: save/load roundtrip")

    # Test CurriculumState model import
    from models import CurriculumState
    cs = CurriculumState(**state)
    assert cs.enabled is True
    assert cs.buffer_size == 5
    print("  PASS: CurriculumState pydantic model")

    print(f"\n{'='*50}")
    print("ALL PLR TESTS PASSED")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
