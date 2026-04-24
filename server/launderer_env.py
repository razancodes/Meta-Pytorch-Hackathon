"""
Memex OS-Agent Benchmark — Launderer Environment (One-Step MDP).

Single-step environment for the Launderer-8B agent.
The Launderer generates an AML scenario in one action; the environment
validates it, runs a frozen Defender episode, and returns:
  reward = max(0, 0.5 - defender_score) for suspicious scenarios
  reward = -2.0 for invalid / malformed scenarios

Contract:
  reset(typology, difficulty) → observation prompt
  step(scenario_json_str)     → (reward, done=True)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Internal imports — dual pattern
try:
    from models import AMLAction, AMLObservation, AMLState, TypologyEnum
    from scenarios.base import BaseScenario
    from scenarios.procedural_generator import GeneratedScenario
    from server.aml_environment import AMLEnvironment
    from graders.grader import AMLGrader
except ImportError:
    from aml_investigation_env.models import AMLAction, AMLObservation, AMLState, TypologyEnum
    from aml_investigation_env.scenarios.base import BaseScenario
    from aml_investigation_env.scenarios.procedural_generator import GeneratedScenario
    from aml_investigation_env.server.aml_environment import AMLEnvironment
    from aml_investigation_env.graders.grader import AMLGrader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INVALID_SCENARIO_PENALTY: float = -2.0

# Required ground truth fields for a valid scenario
REQUIRED_GT_FIELDS = {
    "is_suspicious",
    "correct_decision",
    "typology",
    "key_entities",
    "key_findings",
}

# Required top-level scenario keys
REQUIRED_SCENARIO_KEYS = {
    "initial_alert",
    "customer_profiles",
    "transactions",
    "watchlist_results",
    "network_graph",
    "source_of_funds",
    "ground_truth",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scenario(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a Launderer-generated scenario.

    Returns:
        (is_valid, error_message) — error_message is empty if valid.
    """
    # 1. Top-level keys
    missing_keys = REQUIRED_SCENARIO_KEYS - set(data.keys())
    if missing_keys:
        return False, f"Missing top-level keys: {missing_keys}"

    # 2. Ground truth structure
    gt = data.get("ground_truth", {})
    if not isinstance(gt, dict):
        return False, "ground_truth must be a dict"

    missing_gt = REQUIRED_GT_FIELDS - set(gt.keys())
    if missing_gt:
        return False, f"Missing ground_truth fields: {missing_gt}"

    # 3. is_suspicious must be True (Launderer generates suspicious scenarios)
    if not gt.get("is_suspicious", False):
        return False, "Launderer must generate suspicious scenarios (is_suspicious=True)"

    # 4. Correct decision must be file_sar for suspicious
    if gt.get("correct_decision") != "file_sar":
        return False, "Suspicious scenario must have correct_decision='file_sar'"

    # 5. Typology must be valid
    if gt.get("typology") not in TypologyEnum.values():
        return False, f"Invalid typology '{gt.get('typology')}'. Valid: {TypologyEnum.values()}"

    # 6. Non-empty entities and findings
    if not gt.get("key_entities"):
        return False, "key_entities must be non-empty"
    if not gt.get("key_findings"):
        return False, "key_findings must be non-empty"

    # 7. Alert must have an alert_id
    alert = data.get("initial_alert", {})
    if not isinstance(alert, dict) or not alert.get("alert_id"):
        return False, "initial_alert must contain alert_id"

    # 8. At least one transaction
    txns = data.get("transactions", [])
    if not isinstance(txns, list) or len(txns) == 0:
        return False, "transactions must be a non-empty list"

    # 9. At least one customer profile
    profiles = data.get("customer_profiles", {})
    if not isinstance(profiles, dict) or len(profiles) == 0:
        return False, "customer_profiles must be a non-empty dict"

    return True, ""


# ---------------------------------------------------------------------------
# LaundererEnv
# ---------------------------------------------------------------------------

@dataclass
class LaundererObs:
    """Observation returned to the Launderer agent."""
    prompt: str
    typology: str
    difficulty: str
    done: bool = False
    reward: float = 0.0


class LaundererEnv:
    """One-step MDP environment for the Launderer-8B agent.

    The Launderer's task: generate a valid, evasive AML scenario JSON
    that fools a frozen Defender checkpoint.

    Lifecycle:
        obs = env.reset(typology, difficulty)
        result = env.step(scenario_json_str)
        # result.done is always True (one-step MDP)
    """

    def __init__(
        self,
        defender_env: Optional[AMLEnvironment] = None,
        defender_rollout_fn: Optional[Any] = None,
    ) -> None:
        """
        Args:
            defender_env: Pre-configured AMLEnvironment for Defender rollouts.
                          If None, created on demand.
            defender_rollout_fn: Callable(env, scenario) -> defender_score
                                 Injected from the training script with the
                                 frozen Defender model. If None, uses a dummy
                                 score (for testing/dry-run).
        """
        self._defender_env = defender_env or AMLEnvironment()
        self._defender_rollout_fn = defender_rollout_fn
        self._current_typology: str = ""
        self._current_difficulty: str = ""
        self._episode_count: int = 0

    def reset(
        self,
        typology: str = "structuring",
        difficulty: str = "medium",
        seed: Optional[int] = None,
    ) -> LaundererObs:
        """Reset and return the Launderer's observation prompt.

        The prompt instructs the Launderer to generate a scenario JSON
        for the given typology and difficulty.
        """
        if seed is not None:
            random.seed(seed)

        self._current_typology = typology
        self._current_difficulty = difficulty
        self._episode_count += 1

        prompt = self._build_prompt(typology, difficulty)
        return LaundererObs(
            prompt=prompt,
            typology=typology,
            difficulty=difficulty,
            done=False,
            reward=0.0,
        )

    def step(self, scenario_json_str: str) -> LaundererObs:
        """Process the Launderer's generated scenario.

        Args:
            scenario_json_str: Raw JSON string generated by the Launderer LLM.

        Returns:
            LaundererObs with done=True and computed reward.
        """
        # 1. Parse JSON
        try:
            data = json.loads(scenario_json_str)
            if not isinstance(data, dict):
                raise ValueError("Scenario must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return LaundererObs(
                prompt="",
                typology=self._current_typology,
                difficulty=self._current_difficulty,
                done=True,
                reward=INVALID_SCENARIO_PENALTY,
            )

        # 2. Validate schema
        is_valid, error = validate_scenario(data)
        if not is_valid:
            return LaundererObs(
                prompt="",
                typology=self._current_typology,
                difficulty=self._current_difficulty,
                done=True,
                reward=INVALID_SCENARIO_PENALTY,
            )

        # 3. Run frozen Defender against this scenario
        scenario = GeneratedScenario(data)
        defender_score = self._run_defender(scenario)

        # 4. Compute Launderer reward: high when Defender fails
        # reward ∈ [0, 0.5] — Launderer is rewarded for fooling the Defender
        reward = max(0.0, 0.5 - defender_score)

        return LaundererObs(
            prompt="",
            typology=self._current_typology,
            difficulty=self._current_difficulty,
            done=True,
            reward=round(reward, 4),
        )

    def _run_defender(self, scenario: BaseScenario) -> float:
        """Run a frozen Defender episode against the generated scenario.

        Returns the Defender's terminal score ∈ [-1, +1].
        """
        if self._defender_rollout_fn is not None:
            return self._defender_rollout_fn(self._defender_env, scenario)

        # Fallback: no Defender checkpoint — return a neutral score
        # This allows dry-run testing without a GPU
        return 0.0

    def _build_prompt(self, typology: str, difficulty: str) -> str:
        """Build the system+user prompt for the Launderer."""
        return (
            f"You are an AML scenario generator. Your task is to create a realistic, "
            f"evasive money-laundering scenario that is difficult for an investigator to detect.\n\n"
            f"CONSTRAINTS:\n"
            f"  - Typology: {typology}\n"
            f"  - Difficulty: {difficulty}\n"
            f"  - The scenario MUST be suspicious (is_suspicious: true)\n"
            f"  - correct_decision must be 'file_sar'\n"
            f"  - Include at least 3 transactions with realistic amounts\n"
            f"  - Include customer profiles with plausible KYC data\n"
            f"  - Include a network graph with entity connections\n"
            f"  - Include source of funds information\n"
            f"  - Include watchlist results\n\n"
            f"OUTPUT: A single JSON object with the following top-level keys:\n"
            f"  initial_alert, customer_profiles, transactions, watchlist_results,\n"
            f"  network_graph, source_of_funds, ground_truth\n\n"
            f"The ground_truth object MUST contain:\n"
            f"  is_suspicious (bool), correct_decision (str), typology (str),\n"
            f"  key_entities (list[str]), key_findings (list[str]),\n"
            f"  excluded_entities (list[str]), red_flags (list[str])\n\n"
            f"Generate the scenario JSON now:"
        )
