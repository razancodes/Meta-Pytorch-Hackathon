"""
Memex OS-Agent Benchmark — Dense Reward Grader.

Implements the two-tier reward system:

  1. **Per-Step Micro-Rewards** (grade_step):
     - Action cost, redundancy penalty, page fault, async timeout,
       successful page, meta-injection.

  2. **Terminal Composite Score** (grade_terminal):
     - Decision correctness, typology, findings coverage, entity F1,
       efficiency — mapped to [-1.0, +1.0] range.

The final episode reward = terminal_score + accumulated_step_rewards,
clamped to [-1.0, +1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

# Dual import
try:
    from scenarios import get_scenario
    from models import AMLState
except ImportError:
    from aml_investigation_env.scenarios import get_scenario
    from aml_investigation_env.models import AMLState


# ---------------------------------------------------------------------------
# Reward Constants
# ---------------------------------------------------------------------------

# Per-step costs/rewards
ACTION_COST:        float = -0.02   # Every step
REDUNDANT_PENALTY:  float = -0.03   # Duplicate tool+params
PAGE_FAULT_PENALTY: float = -0.05   # Evicted data not on disk
ASYNC_TIMEOUT_PENALTY: float = -0.10  # Premature async retrieval
SUCCESSFUL_PAGE:    float = +0.10   # Good write_to_case_file
META_INJECTION:     float = +0.15   # Successful kernel update
UNIQUE_TOOL_BONUS:  float = +0.03   # Novel, useful tool call

# Terminal
CORRECT_SAR_BONUS:    float = +1.00
FALSE_POSITIVE_PENALTY: float = -1.00

# Optimal steps per task (for efficiency scoring)
OPTIMAL_STEPS: Dict[str, int] = {
    "easy": 7,
    "medium": 10,
    "hard": 13,
}
MAX_STEPS: int = 25

# Common aliases for fuzzy finding matching (preserved from original)
ALIASES: Dict[str, List[str]] = {
    "sub_threshold": ["structuring", "below_threshold", "under_threshold", "sub_ctr"],
    "no_source_documentation": ["no_documentation", "undocumented", "no_business_justification", "no_source"],
    "cash_intensive_occupation": ["cash_business", "business_justification", "occupation"],
    "same_branch_repeated": ["same_branch", "single_branch", "branch_pattern"],
    "total_exceeds_ctr_threshold": ["total_above", "exceeds_threshold", "aggregate_amount", "ctr"],
    "rapid_fan_out": ["fan_out", "dispersal", "split_transfer", "multiple_outgoing"],
    "pep_connection": ["pep", "politically_exposed"],
    "shared_registered_address": ["shared_address", "same_address", "common_address"],
    "offshore_source": ["offshore", "foreign_source"],
    "newly_incorporated": ["recent_incorporation", "new_company", "recently_formed"],
    "no_trade_documentation": ["no_documentation", "no_trade_docs", "missing_docs"],
    "over_invoicing": ["over_invoice", "inflated_price", "price_manipulation", "above_market"],
    "beneficial_owner_connection": ["beneficial_owner", "family_connection", "related_party"],
    "fatf_jurisdiction": ["fatf", "high_risk_jurisdiction", "monitored_jurisdiction"],
    "reversed_transaction": ["reversal", "reversed", "corrected_transaction", "amended_transaction"],
    "unexplained_funds": ["unexplained", "unknown_source", "unjustified_funds"],
    "multiple_sub_threshold_deposits": ["multiple_deposits", "structuring", "sub_threshold", "below_threshold"],
    "no_cash_intensive_occupation": ["no_cash_business", "non_cash_occupation", "clerk", "office"],
}


class AMLGrader:
    """Dense reward grader for the Memex OS-Agent benchmark.

    grade_step() — called once per step to compute micro-rewards.
    grade_terminal() — called once at episode end for the composite score.
    grade() — legacy-compatible entry point for terminal scoring.
    """

    # ------------------------------------------------------------------ #
    # Per-Step Rewards                                                     #
    # ------------------------------------------------------------------ #

    def grade_step(
        self,
        tool: str,
        params: Dict[str, Any],
        state: AMLState,
        call_hash: str,
        *,
        is_page_fault: bool = False,
        is_async_timeout: bool = False,
        is_successful_page: bool = False,
        is_meta_injection: bool = False,
    ) -> float:
        """Compute the per-step micro-reward.

        The step reward is the sum of all applicable signals.
        The environment passes boolean flags from the StateManager so
        the grader doesn't need to know about OS internals.

        Returns:
            float — typically in [-0.15, +0.15]
        """
        reward = ACTION_COST  # Base cost per step

        # Redundancy check
        if call_hash in state.tool_call_hashes:
            reward += REDUNDANT_PENALTY

        # OS mechanic signals
        if is_page_fault:
            reward += PAGE_FAULT_PENALTY

        if is_async_timeout:
            reward += ASYNC_TIMEOUT_PENALTY

        if is_successful_page:
            reward += SUCCESSFUL_PAGE

        if is_meta_injection:
            reward += META_INJECTION

        # Small bonus for unique, non-redundant tool calls
        if call_hash not in state.tool_call_hashes and not is_page_fault:
            reward += UNIQUE_TOOL_BONUS

        return round(reward, 4)

    # ------------------------------------------------------------------ #
    # Terminal Grading                                                     #
    # ------------------------------------------------------------------ #

    def grade(
        self,
        task_id: str,
        decision: str,
        findings: List[str],
        entities_flagged: List[str],
        typology: str,
        state: AMLState,
    ) -> float:
        """Compute the final episode score in [-1.0, +1.0].

        Combines a 5-dimension weighted rubric with accumulated step rewards.
        """
        scenario = get_scenario(task_id)
        gt = scenario.ground_truth

        # --------------- 1. Decision correctness (weight 0.35) ---------- #
        decision_correct = decision == gt["correct_decision"]
        decision_score = 0.35 if decision_correct else -0.50

        # --------------- 2. Typology correctness (weight 0.15) ---------- #
        typology_score = 0.0
        if typology and typology.lower().strip() == gt.get("typology", "").lower().strip():
            typology_score = 0.15

        # --------------- 3. Key findings coverage (weight 0.25) --------- #
        gt_findings = gt.get("key_findings", [])
        findings_score = 0.0
        if gt_findings:
            matched = self._count_findings_matched(findings, gt_findings)
            findings_score = 0.25 * (matched / len(gt_findings))

        # --------------- 4. Entity precision/recall F1 (weight 0.15) ---- #
        gt_entities: Set[str] = set(gt.get("key_entities", []))
        flagged_set: Set[str] = set(entities_flagged) if entities_flagged else set()
        excluded: Set[str] = set(gt.get("excluded_entities", []))
        entity_score = 0.0

        if gt_entities:
            false_positives = len(flagged_set & excluded)
            true_positives = len(flagged_set & gt_entities)

            precision = true_positives / max(len(flagged_set) - false_positives + true_positives, 1)
            recall = true_positives / len(gt_entities)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            entity_score = 0.15 * f1

        # --------------- 5. Efficiency (weight 0.10) -------------------- #
        optimal = OPTIMAL_STEPS.get(task_id, 10)
        step_count = state.step_count
        if step_count <= optimal:
            efficiency = 1.0
        else:
            efficiency = max(0.0, 1.0 - (step_count - optimal) / (MAX_STEPS - optimal))
        efficiency_score = 0.10 * efficiency

        # --------------- Composite ------------------------------------ #
        terminal_score = decision_score + typology_score + findings_score + entity_score + efficiency_score

        # Add accumulated step rewards (micro-rewards from OS mechanics)
        total = terminal_score + state.accumulated_reward

        # Clamp to [-1.0, +1.0]
        return round(max(-1.0, min(1.0, total)), 4)

    # ------------------------------------------------------------------ #
    # grade_step legacy alias (for backward compatibility)                 #
    # ------------------------------------------------------------------ #

    def grade_terminal(self, *args: Any, **kwargs: Any) -> float:
        """Alias for grade() for clarity."""
        return self.grade(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Findings Matching                                                    #
    # ------------------------------------------------------------------ #

    def _count_findings_matched(self, agent_findings: List[str], gt_findings: List[str]) -> int:
        """Count ground-truth findings matched by the agent's findings.

        Three-tier fuzzy matching:
        1. ≥50% keyword overlap in any single finding
        2. Alias table lookup
        3. Substring fallback
        """
        matched = 0
        normalised_agent = [f.lower().replace("-", "_").replace(" ", "_") for f in agent_findings]
        joined_agent = " ".join(normalised_agent)

        for gt_f in gt_findings:
            gt_norm = gt_f.lower().replace("-", "_")
            gt_keywords = [kw for kw in gt_norm.split("_") if len(kw) > 2]

            # Tier 1: ≥50% keyword overlap in any single agent finding
            found = False
            for af in normalised_agent:
                hits = sum(1 for kw in gt_keywords if kw in af)
                if gt_keywords and hits / len(gt_keywords) >= 0.5:
                    matched += 1
                    found = True
                    break

            if found:
                continue

            # Tier 2: Alias table
            aliases = ALIASES.get(gt_norm, [])
            found_via_alias = False
            for alias in aliases:
                alias_norm = alias.lower().replace("-", "_").replace(" ", "_")
                if alias_norm in joined_agent:
                    matched += 1
                    found_via_alias = True
                    break
            if found_via_alias:
                continue

            # Tier 3: Any GT keyword appears as substring
            if any(kw in joined_agent for kw in gt_keywords):
                matched += 1

        return matched
