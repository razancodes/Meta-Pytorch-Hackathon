"""
Memex OS-Agent Benchmark — Prioritized Level Replay (PLR) Engine.

Maintains a buffer of past scenarios scored by 'regret' (the gap between
the theoretical maximum score and the protagonist's actual performance).
Scenarios where the detective fails most are replayed most often, creating
an adaptive curriculum that automatically increases difficulty as the agent
improves.

Architecture:
  ┌─ PLR Buffer ─────────────────────────────────────────────────┐
  │  ScenarioRecord(id, typology, difficulty, regret, score)     │
  │  ...                                                          │
  │  Weighted sampler: P(scenario) ∝ regret × staleness_bonus    │
  └──────────────────────────────────────────────────────────────┘
             ↓ sample_scenario()              ↑ update()
       ┌─────────────┐                  ┌──────────────┐
       │ train_ppo.py │  ──episode──→   │ proxy_regret │
       │  episode loop│                  │ 1.0 - score  │
       └─────────────┘                  └──────────────┘

Design decisions:
  - Buffer size 200: large enough for diversity, small enough for fresh signals
  - Temperature 0.1: soft sampling, not pure greedy (prevents mode collapse)
  - Staleness threshold 50: force re-exploration of old buffer entries
  - 20% random exploration: ensures fresh scenarios enter the buffer
  - EMA α=0.3: smooths regret updates for revisited scenarios
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ScenarioRecord:
    """A single scenario entry in the PLR buffer."""

    scenario_id: str              # Unique episode ID
    typology: str                 # structuring | layering | trade_based_ml
    difficulty: str               # easy | medium | hard
    params: Dict[str, Any]        # Full scenario params snapshot for replay
    regret: float = 0.0           # proxy_regret = 1.0 - protagonist_score
    protagonist_score: float = 0.0
    antagonist_score: float = 1.0  # Conservative init: assume max possible
    visit_count: int = 0
    last_updated: float = field(default_factory=lambda: 0.0)


class PLREngine:
    """Prioritized Level Replay for Memex AML Environment.

    Wraps the procedural generator's scenario selection with regret-based
    weighted sampling. Scenarios where the detective performs worst are
    automatically surfaced more often during training.

    Usage::

        plr = PLREngine(buffer_size=200)

        # During training loop:
        diff, typo = plr.sample_scenario()          # replaces random.choice()
        # ... run episode ...
        plr.update(scenario_id, diff, typo, score)   # update regret
        wandb.log(plr.get_wandb_metrics())           # log curriculum stats
    """

    def __init__(
        self,
        buffer_size: int = 200,
        temperature: float = 0.5,
        staleness_threshold: int = 50,
        random_scenario_prob: float = 0.2,
    ):
        self.buffer: deque[ScenarioRecord] = deque(maxlen=buffer_size)
        self.temperature = temperature
        self.staleness_threshold = staleness_threshold
        self.random_scenario_prob = random_scenario_prob
        self.iteration = 0

        # WandB-ready metric history
        self._metrics: Dict[str, List[float]] = {
            "mean_regret": [],
            "max_regret": [],
            "mean_difficulty_score": [],
            "buffer_diversity": [],
        }

    @staticmethod
    def _difficulty_score(difficulty: str) -> float:
        """Map difficulty string to numeric score for tracking."""
        return {"easy": 1.0, "medium": 2.0, "hard": 3.0}.get(difficulty, 2.0)

    def sample_scenario(
        self,
        difficulties: List[str],
        typologies: List[str],
    ) -> tuple[str, str]:
        """Sample a (difficulty, typology) pair for the next episode.

        Returns:
            Tuple of (difficulty, typology) strings.

        Logic:
            - If buffer has < 10 entries OR 20% random chance: return random pair
            - Otherwise: sample from buffer weighted by regret × staleness bonus
        """
        import random as _random

        # Exploration: fresh random scenario
        if len(self.buffer) < 10 or np.random.random() < self.random_scenario_prob:
            return _random.choice(difficulties), _random.choice(typologies)

        # Exploitation: regret-weighted sampling from buffer
        records = list(self.buffer)
        regrets = np.array([r.regret for r in records], dtype=np.float64)

        # Staleness bonus: scenarios not visited recently get upweighted
        staleness = np.array([
            min(1.5, 1.0 + (self.iteration - r.last_updated) / max(self.staleness_threshold, 1))
            for r in records
        ], dtype=np.float64)

        weights = regrets * staleness
        weights = np.clip(weights, 0.0, None)

        # Uniform fallback if all weights are zero
        if weights.sum() == 0:
            weights = np.ones(len(weights), dtype=np.float64)

        # Softmax with temperature for soft sampling
        weights = weights / max(self.temperature, 1e-8)
        weights = np.exp(weights - weights.max())  # numerical stability
        weights = weights / weights.sum()

        selected = np.random.choice(len(records), p=weights)
        record = records[selected]
        return record.difficulty, record.typology

    def update(
        self,
        scenario_id: str,
        difficulty: str,
        typology: str,
        protagonist_score: float,
    ) -> None:
        """Update the PLR buffer after an episode completes.

        Args:
            scenario_id: Unique identifier for this episode.
            difficulty: The difficulty level used.
            typology: The typology used.
            protagonist_score: The agent's terminal composite score.
        """
        from .oracle import proxy_regret

        regret = proxy_regret(protagonist_score)

        # Check if this (difficulty, typology) combo already exists in buffer
        existing = next(
            (r for r in self.buffer if r.difficulty == difficulty and r.typology == typology),
            None,
        )

        if existing:
            # Exponential moving average for stability
            alpha = 0.3
            existing.regret = alpha * regret + (1 - alpha) * existing.regret
            existing.protagonist_score = protagonist_score
            existing.visit_count += 1
            existing.last_updated = float(self.iteration)
        else:
            record = ScenarioRecord(
                scenario_id=scenario_id,
                typology=typology,
                difficulty=difficulty,
                params={"difficulty": difficulty, "typology": typology},
                regret=regret,
                protagonist_score=protagonist_score,
                antagonist_score=1.0,
                visit_count=1,
                last_updated=float(self.iteration),
            )
            self.buffer.append(record)

        self.iteration += 1
        self._log_metrics()

    def _log_metrics(self) -> None:
        """Compute and store curriculum metrics for WandB."""
        if not self.buffer:
            return

        records = list(self.buffer)
        regrets = [r.regret for r in records]

        self._metrics["mean_regret"].append(float(np.mean(regrets)))
        self._metrics["max_regret"].append(float(np.max(regrets)))
        self._metrics["mean_difficulty_score"].append(
            float(np.mean([self._difficulty_score(r.difficulty) for r in records]))
        )
        # Diversity: fraction of 9 possible (typology × difficulty) combos covered
        unique_combos = len(set((r.typology, r.difficulty) for r in records))
        self._metrics["buffer_diversity"].append(unique_combos / 9.0)

    def get_wandb_metrics(self) -> Dict[str, float]:
        """Return latest metrics for WandB logging at each PPO iteration."""
        if not self._metrics["mean_regret"]:
            return {}
        return {
            "curriculum/mean_regret": self._metrics["mean_regret"][-1],
            "curriculum/max_regret": self._metrics["max_regret"][-1],
            "curriculum/mean_difficulty": self._metrics["mean_difficulty_score"][-1],
            "curriculum/buffer_diversity": self._metrics["buffer_diversity"][-1],
            "curriculum/buffer_size": len(self.buffer),
            "curriculum/iteration": self.iteration,
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Return current state for AGUI CurriculumState rendering."""
        if not self.buffer:
            return {
                "enabled": True,
                "buffer_size": 0,
                "mean_regret": 0.0,
                "max_regret": 0.0,
                "mean_difficulty": 1.0,
                "buffer_diversity": 0.0,
                "current_scenario_regret": 0.0,
                "difficulty_label": "easy",
            }

        records = list(self.buffer)
        regrets = [r.regret for r in records]
        latest = records[-1] if records else None

        return {
            "enabled": True,
            "buffer_size": len(self.buffer),
            "mean_regret": float(np.mean(regrets)),
            "max_regret": float(np.max(regrets)),
            "mean_difficulty": float(np.mean([
                self._difficulty_score(r.difficulty) for r in records
            ])),
            "buffer_diversity": len(set((r.typology, r.difficulty) for r in records)) / 9.0,
            "current_scenario_regret": latest.regret if latest else 0.0,
            "difficulty_label": latest.difficulty if latest else "easy",
        }

    def save(self, path: str) -> None:
        """Persist the PLR buffer to disk as JSON."""
        data = []
        for r in self.buffer:
            data.append({
                "scenario_id": r.scenario_id,
                "typology": r.typology,
                "difficulty": r.difficulty,
                "params": r.params,
                "regret": r.regret,
                "protagonist_score": r.protagonist_score,
                "antagonist_score": r.antagonist_score,
                "visit_count": r.visit_count,
                "last_updated": r.last_updated,
            })
        with open(path, "w") as f:
            json.dump({"iteration": self.iteration, "records": data}, f, indent=2)

    def load(self, path: str) -> None:
        """Restore the PLR buffer from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        self.iteration = data.get("iteration", 0)
        self.buffer = deque(maxlen=self.buffer.maxlen)
        for d in data.get("records", []):
            self.buffer.append(ScenarioRecord(**d))
