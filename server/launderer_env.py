"""
Memex OS-Agent Benchmark — Launderer Environment (One-Step MDP).

Single-step environment for the Launderer-8B agent.
The Launderer generates an AML scenario in one action; the environment
validates it, runs a frozen Defender episode, and returns:

Reward shaping (preserves core game: max reward = Defender fails on valid scenario):
  - JSON parse fail (no extractable JSON):         -2.0
  - JSON extracted but schema validation fails:    -1.0
  - Valid scenario, Defender catches it:             0.0 to 0.3
  - Valid scenario, Defender fails:                  0.3 to 1.0

Contract:
  reset(typology, difficulty) → observation prompt
  step(scenario_json_str)     → (reward, done=True)
"""

from __future__ import annotations

import json
import random
import re
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


# Constants

# Reward tiers — shaped to preserve the core game:
#   Primary reward = Defender failure on genuinely suspicious, valid scenario.
#   Penalties scale with severity of invalidity.
PENALTY_JSON_PARSE_FAIL: float = -2.0   # No extractable JSON at all
PENALTY_SCHEMA_FAIL: float = -1.0       # JSON extracted but schema invalid
REWARD_VALID_BASELINE: float = 0.1      # Minimum reward for a valid scenario

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


# JSON Extraction (robust against LLM formatting quirks)

def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from raw LLM output.

    Handles common LLM formatting issues:
      1. ```json ... ``` markdown fences
      2. English preamble before the JSON object
      3. Trailing text after the JSON object
      4. Multiple JSON-like blocks (takes the largest)

    Returns:
        Parsed dict if a valid JSON object is found, None otherwise.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # ── Strategy 1: Strip markdown fences and try direct parse ──
    # Remove ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = cleaned.replace('```', '').strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Strategy 2: Find first '{' and try to parse from there ──
    first_brace = text.find('{')
    if first_brace >= 0:
        # Try parsing from first brace to end
        candidate = text[first_brace:]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Strategy 3: Bracket-matching extraction ──
    # Find the outermost { ... } by counting braces
    if first_brace >= 0:
        depth = 0
        in_str = False
        escape = False
        end_pos = -1
        for i in range(first_brace, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_str:
                escape = True
                continue
            if c == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        if end_pos > first_brace:
            candidate = text[first_brace:end_pos + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# Validation

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


# LaundererEnv

@dataclass
class LaundererObs:
    """Observation returned to the Launderer agent."""
    prompt: str
    typology: str
    difficulty: str
    done: bool = False
    reward: float = 0.0
    error: str = ""        # Validation error message (for debugging)
    is_valid: bool = False  # Whether the scenario passed validation


class LaundererEnv:
    """One-step MDP environment for the Launderer-8B agent.

    The Launderer's task: generate a valid, evasive AML scenario JSON
    that fools a frozen Defender checkpoint.

    Reward shaping (core game: max reward when Defender fails on valid scenario):
      - JSON parse fail:     -2.0  (no extractable JSON)
      - Schema fail:         -1.0  (JSON found but doesn't match required schema)
      - Valid, Defender wins:  0.1 to 0.3  (baseline for valid scenario)
      - Valid, Defender loses:  0.3 to 1.0  (maximum: Defender fully fooled)

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

        Uses robust JSON extraction to handle LLM formatting quirks
        (markdown fences, preamble text, etc.). Shaped rewards give
        partial credit for getting closer to a valid scenario while
        preserving the core game: max reward = Defender fails on valid input.

        Args:
            scenario_json_str: Raw text generated by the Launderer LLM.

        Returns:
            LaundererObs with done=True and computed reward.
        """
        # 1. Extract JSON with robust parser
        data = extract_json(scenario_json_str)

        if data is None:
            # Total parse failure — harshest penalty
            return LaundererObs(
                prompt="", typology=self._current_typology,
                difficulty=self._current_difficulty,
                done=True, reward=PENALTY_JSON_PARSE_FAIL,
                error="json_parse_fail", is_valid=False,
            )

        # 2. Validate schema
        is_valid, error = validate_scenario(data)
        if not is_valid:
            # JSON extracted but schema fails — lighter penalty (partial credit)
            return LaundererObs(
                prompt="", typology=self._current_typology,
                difficulty=self._current_difficulty,
                done=True, reward=PENALTY_SCHEMA_FAIL,
                error=f"schema_fail: {error}", is_valid=False,
            )

        # 3. Run frozen Defender against this scenario
        scenario = GeneratedScenario(data)
        defender_score = self._run_defender(scenario)

        # 4. Compute Launderer reward — core game:
        #    Launderer gets HIGH reward when Defender FAILS (low score).
        #    Launderer gets LOW (but positive) reward when Defender SUCCEEDS.
        #    Range: [REWARD_VALID_BASELINE, 1.0]
        #
        #    defender_score ∈ [-2, +2] typically, clamped to [-1, 1] for reward calc.
        #    When defender_score = -1 (total failure): reward → 1.0
        #    When defender_score = +1 (perfect detection): reward → REWARD_VALID_BASELINE
        clamped_def = max(-1.0, min(1.0, defender_score))
        # Linear mapping: defender_score=-1 → 1.0, defender_score=+1 → baseline
        reward = REWARD_VALID_BASELINE + (1.0 - REWARD_VALID_BASELINE) * (1.0 - clamped_def) / 2.0

        return LaundererObs(
            prompt="", typology=self._current_typology,
            difficulty=self._current_difficulty,
            done=True, reward=round(reward, 4),
            error="", is_valid=True,
        )

    def _run_defender(self, scenario: BaseScenario) -> float:
        """Run a frozen Defender episode against the generated scenario.

        Returns the Defender's terminal score ∈ [-2, +2].
        """
        if self._defender_rollout_fn is not None:
            return self._defender_rollout_fn(self._defender_env, scenario)

        # Fallback: no Defender checkpoint — return a neutral score
        # This allows dry-run testing without a GPU
        return 0.0

    def _build_prompt(self, typology: str, difficulty: str) -> str:
        """Build the system+user prompt for the Launderer.

        Explicit JSON-only instructions to prevent markdown fences and preamble.
        Includes a minimal skeleton to guide the LLM's output format.
        """
        return (
            f"You are an AML scenario generator. Create a realistic, evasive "
            f"money-laundering scenario for typology '{typology}' at '{difficulty}' difficulty.\n\n"
            f"CRITICAL OUTPUT RULES:\n"
            f"  - Output ONLY a raw JSON object. NO markdown, NO code fences, NO explanations.\n"
            f"  - Do NOT wrap output in ```json blocks or add any text before/after the JSON.\n"
            f"  - The very first character of your response must be '{{'.\n\n"
            f"REQUIRED STRUCTURE (all keys mandatory):\n"
            f'{{"initial_alert": {{"alert_id": "ALT-...", "customer_id": "C-...", "summary": "..."}},\n'
            f' "customer_profiles": {{"C-...": {{"name": "...", "occupation": "...", "jurisdiction": "..."}}}},\n'
            f' "transactions": [{{"transaction_id": "T-...", "amount": 50000, "currency": "USD", '
            f'"sender": "C-...", "receiver": "C-...", "date": "2024-01-15", "type": "wire"}}],\n'
            f' "watchlist_results": {{"C-...": {{"hit": false, "details": "No matches"}}}},\n'
            f' "network_graph": {{"nodes": ["C-..."], "edges": [{{"from": "C-...", "to": "C-...", "label": "wire"}}]}},\n'
            f' "source_of_funds": {{"C-...": {{"declared": "Business revenue", "verified": false}}}},\n'
            f' "ground_truth": {{"is_suspicious": true, "correct_decision": "file_sar",\n'
            f'   "typology": "{typology}", "key_entities": ["C-..."],\n'
            f'   "key_findings": ["Finding 1", "Finding 2", "Finding 3"],\n'
            f'   "excluded_entities": [], "red_flags": ["Red flag 1"]}}}}\n\n'
            f"CONSTRAINTS:\n"
            f"  - is_suspicious MUST be true, correct_decision MUST be 'file_sar'\n"
            f"  - typology MUST be '{typology}'\n"
            f"  - Include at least 3 transactions with realistic amounts\n"
            f"  - Include realistic customer profiles with plausible KYC data\n"
            f"  - Make the scenario evasive and difficult to detect\n\n"
            f"Generate the JSON now:"
        )
