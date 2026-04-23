"""
Memex OS-Agent Benchmark — Adversarial "GAN-Style" Battle Orchestrator.

Runs the PPO Defender agent against Adversary-generated scenarios and persists
winning adversary scenarios (where the Defender fails) to a SQLite database
for later use in the DPO training pipeline.

Architecture:
  1. Adversary Agent generates N evasive scenarios
  2. Defender agent runs each scenario through AMLEnvironment
  3. If Defender score < threshold → scenario saved to AdversarialSuccesses table
  4. These "winning" adversary scenarios become negative examples in DPO training

Usage:
    python train_adversary.py \
        --adversary-model gpt-4o-mini \
        --episodes 20 \
        --difficulty hard \
        --threshold 0.3

SQLite Schema (AdversarialSuccesses):
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    scenario_json   TEXT NOT NULL
    defender_score  REAL NOT NULL
    adversary_model TEXT NOT NULL
    typology        TEXT NOT NULL
    difficulty      TEXT NOT NULL
    evasion_techniques TEXT
    created_at      TEXT NOT NULL

NOTE: This script does NOT touch train_ppo.py or train_ppo_70b.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scenarios.adversary_agent import AdversaryAgent
from scenarios.procedural_generator import GeneratedScenario

# ---------------------------------------------------------------------------
# SQLite Database Manager
# ---------------------------------------------------------------------------

DB_PATH = PROJECT_ROOT / "adversarial_successes.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS AdversarialSuccesses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_json       TEXT    NOT NULL,
    defender_score      REAL    NOT NULL,
    adversary_model     TEXT    NOT NULL,
    typology            TEXT    NOT NULL,
    difficulty          TEXT    NOT NULL,
    evasion_techniques  TEXT,
    created_at          TEXT    NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO AdversarialSuccesses
    (scenario_json, defender_score, adversary_model, typology, difficulty, evasion_techniques, created_at)
VALUES
    (?, ?, ?, ?, ?, ?, ?);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database and return a connection."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def save_adversary_win(
    conn: sqlite3.Connection,
    scenario_data: Dict[str, Any],
    defender_score: float,
    adversary_model: str,
    typology: str,
    difficulty: str,
    evasion_techniques: Optional[List[str]] = None,
) -> int:
    """Save a winning adversary scenario to the database.

    Returns the row ID of the inserted record.
    """
    cursor = conn.execute(
        _INSERT_SQL,
        (
            json.dumps(scenario_data, default=str),
            defender_score,
            adversary_model,
            typology,
            difficulty,
            json.dumps(evasion_techniques or []),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid or 0


# ---------------------------------------------------------------------------
# Simple Defender Runner (Heuristic Baseline)
# ---------------------------------------------------------------------------

def run_defender_heuristic(scenario_data: Dict[str, Any]) -> float:
    """Run a simple heuristic defender against the scenario.

    This provides a baseline evaluation without requiring a full PPO model.
    In production, replace this with the actual PPO agent checkpoint.

    Returns a score between 0.0 (complete failure) and 1.0 (perfect).
    """
    gt = scenario_data.get("ground_truth", {})
    if not gt:
        return 0.0

    score = 0.0
    max_score = 5.0

    # Check: Does the scenario have enough evidence for detection?
    profiles = scenario_data.get("customer_profiles", {})
    transactions = scenario_data.get("transactions", [])
    device_fps = scenario_data.get("device_fingerprints", {})
    customs = scenario_data.get("customs_invoices", {})

    # Heuristic 1: Can we identify the correct typology?
    typology = gt.get("typology", "")
    if typology == "structuring":
        # Check for sub-threshold deposits
        sub_threshold = [t for t in transactions if t.get("amount", 0) < 10000 and t.get("type") == "cash_deposit"]
        if len(sub_threshold) >= 3:
            score += 1.0
    elif typology == "mule_ring":
        # Check for shared devices
        device_sets = {}
        for eid, fps in device_fps.items():
            for fp in (fps if isinstance(fps, list) else [fps]):
                dev = fp.get("device_id", "")
                if dev:
                    device_sets.setdefault(dev, []).append(eid)
        shared = {d: ents for d, ents in device_sets.items() if len(ents) > 1}
        if shared:
            score += 1.0
    elif typology in ("trade_based_ml", "phantom_invoice"):
        # Check for phantom invoices
        phantoms = [c for c in customs.values() if c.get("is_phantom") or not c.get("bill_of_lading")]
        if phantoms:
            score += 1.0
    elif typology == "pass_through":
        # Check for circular flows
        if len(transactions) >= 3:
            sources = {t.get("customer_id") for t in transactions}
            targets = {t.get("counterparty") for t in transactions}
            if sources & targets:  # Circular if any source is also a target
                score += 1.0

    # Heuristic 2: Can we identify key entities?
    key_entities = set(gt.get("key_entities", []))
    identified = key_entities & set(profiles.keys())
    if key_entities:
        score += (len(identified) / len(key_entities)) * 2.0

    # Heuristic 3: Are there clear red flags visible?
    red_flags = gt.get("red_flags", [])
    if len(red_flags) >= 2:
        score += 1.0
    elif len(red_flags) >= 1:
        score += 0.5

    # Heuristic 4: Noise level (harder = more noise)
    excluded = gt.get("excluded_entities", [])
    noise_ratio = len(excluded) / max(len(profiles), 1)
    if noise_ratio > 0.3:
        score -= 0.5  # Heavy noise makes it harder

    return min(1.0, max(0.0, score / max_score))


# ---------------------------------------------------------------------------
# Battle Orchestrator
# ---------------------------------------------------------------------------

def run_battle(
    adversary_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    episodes: int = 10,
    difficulty: str = "hard",
    typology: Optional[str] = None,
    threshold: float = 0.3,
    db_path: Path = DB_PATH,
    verbose: bool = True,
    is_local: bool = False,
) -> Dict[str, Any]:
    """Run the adversarial battle loop.

    Args:
        adversary_model: LLM model for the adversary.
        api_key: API key (or OPENAI_API_KEY env var).
        episodes: Number of scenarios to generate and test.
        difficulty: easy | medium | hard.
        typology: Specific typology or None for random.
        threshold: Defender score below this = adversary win.
        db_path: Path to the SQLite database.
        verbose: Print progress.

    Returns:
        Summary dict with statistics.
    """
    agent = AdversaryAgent(
        model=adversary_model,
        api_key=api_key,
        temperature=0.9,
        is_local=is_local,
    )

    conn = init_db(db_path)

    results = {
        "total_episodes": episodes,
        "adversary_wins": 0,
        "defender_wins": 0,
        "errors": 0,
        "scores": [],
        "saved_scenarios": [],
    }

    if verbose:
        print("=" * 60)
        print(f" MEMEX Adversarial Battle")
        print(f" Adversary: {adversary_model}")
        print(f" Difficulty: {difficulty.upper()}")
        print(f" Episodes: {episodes}")
        print(f" Threshold: {threshold}")
        print("=" * 60)
        print()

    for ep in range(1, episodes + 1):
        try:
            t0 = time.time()
            scenario_data = agent.generate(typology=typology, difficulty=difficulty)
            gen_time = time.time() - t0

            meta = scenario_data.get("_meta", {})
            typo = meta.get("typology", "unknown")
            evasion = meta.get("evasion_techniques", [])

            # Run defender
            defender_score = run_defender_heuristic(scenario_data)
            results["scores"].append(defender_score)

            if defender_score < threshold:
                # Adversary wins — save to DB
                results["adversary_wins"] += 1
                row_id = save_adversary_win(
                    conn, scenario_data, defender_score,
                    adversary_model, typo, difficulty, evasion,
                )
                results["saved_scenarios"].append(row_id)

                if verbose:
                    print(f"  [{ep:3d}/{episodes}] [FAIL] ADVERSARY WIN  "
                          f"score={defender_score:.3f}  typo={typo}  "
                          f"saved=row#{row_id}  ({gen_time:.1f}s)")
            else:
                results["defender_wins"] += 1
                if verbose:
                    print(f"  [{ep:3d}/{episodes}] [PASS] DEFENDER WIN   "
                          f"score={defender_score:.3f}  typo={typo}  ({gen_time:.1f}s)")

        except Exception as e:
            results["errors"] += 1
            if verbose:
                print(f"  [{ep:3d}/{episodes}] [!] ERROR: {e}")

    conn.close()

    # Summary
    avg_score = sum(results["scores"]) / max(len(results["scores"]), 1)
    results["avg_defender_score"] = avg_score
    results["adversary_win_rate"] = results["adversary_wins"] / max(episodes, 1)

    if verbose:
        print()
        print("=" * 60)
        print(f" Battle Complete")
        print(f" Adversary Wins: {results['adversary_wins']}/{episodes} "
              f"({results['adversary_win_rate']:.0%})")
        print(f" Defender Wins:  {results['defender_wins']}/{episodes}")
        print(f" Avg Defender Score: {avg_score:.3f}")
        print(f" Errors: {results['errors']}")
        print(f" Saved to: {db_path}")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memex Adversarial Battle — GAN-style training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_adversary.py --episodes 20 --difficulty hard
    python train_adversary.py --adversary-model gpt-4o-mini --typology mule_ring
    python train_adversary.py --threshold 0.4 --episodes 50
        """,
    )

    parser.add_argument(
        "--adversary-model",
        default="gpt-4o-mini",
        help="LLM model for the adversary agent (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of adversarial scenarios to generate (default: 10)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Scenario difficulty (default: hard)",
    )
    parser.add_argument(
        "--typology",
        choices=["mule_ring", "pass_through", "phantom_invoice"],
        default=None,
        help="Specific typology (default: random)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Defender score threshold for adversary win (default: 0.3)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run entirely locally using Unsloth (overrides adversary model to Llama 8B if default)",
    )

    args = parser.parse_args()

    if args.local and args.adversary_model == "gpt-4o-mini":
        args.adversary_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

    results = run_battle(
        adversary_model=args.adversary_model,
        api_key=args.api_key,
        episodes=args.episodes,
        difficulty=args.difficulty,
        typology=args.typology,
        threshold=args.threshold,
        db_path=args.db_path,
        verbose=not args.quiet,
        is_local=args.local,
    )

    # Output JSON summary for piping
    if args.quiet:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
