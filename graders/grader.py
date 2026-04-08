"""
Deterministic grader for all three AML investigation tasks.

Grading rubric:
  - Decision correctness    0.30
  - Typology correctness    0.15
  - Key findings coverage   0.25  (proportional to # matched)
  - Entity precision/recall 0.15
  - Efficiency              0.15  (based on step count vs optimal)

Total: 1.00
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Dual import: in-repo vs standalone
try:
    from scenarios import get_scenario
    from models import AMLState
except ImportError:
    from .scenarios import get_scenario
    from .models import AMLState


# ---- Per-task optimal step counts (used for efficiency scoring) --------
OPTIMAL_STEPS: Dict[str, int] = {
    "easy": 5,    # review_alert, get_customer_profile, query_transactions, check_source, file_sar
    "medium": 8,  # alert + profile + transactions + watchlist(x2) + network + source + file_sar
    "hard": 10,   # alert + profile + transactions + watchlist + network(depth2) + source + market + assess_risk + file_sar
}

MAX_STEPS = 25


class AMLGrader:
    """Scores a completed AML investigation episode."""

    def grade(
        self,
        task_id: str,
        decision: str,
        findings: List[str],
        entities_flagged: List[str],
        typology: str,
        state: AMLState,
    ) -> float:
        """
        Compute a [0.0, 1.0] score for the episode.

        Parameters
        ----------
        task_id        : one of 'easy', 'medium', 'hard'
        decision       : 'file_sar' or 'close_alert'
        findings       : list of finding strings the agent reported
        entities_flagged: list of entity IDs the agent flagged
        typology       : typology string the agent stated
        state          : AMLState at episode end

        Returns
        -------
        float in [0.0, 1.0]
        """
        scenario = get_scenario(task_id)
        gt = scenario.ground_truth
        score = 0.0

        # ------------------------------------------------------------------ #
        # 1. Decision correctness  (weight 0.30)                              #
        # ------------------------------------------------------------------ #
        if decision == gt["correct_decision"]:
            score += 0.30

        # ------------------------------------------------------------------ #
        # 2. Typology correctness  (weight 0.15)                              #
        # ------------------------------------------------------------------ #
        if typology and typology.lower().strip() == gt.get("typology", "").lower().strip():
            score += 0.15

        # ------------------------------------------------------------------ #
        # 3. Key findings coverage  (weight 0.25)                             #
        # ------------------------------------------------------------------ #
        gt_findings = gt.get("key_findings", [])
        if gt_findings:
            matched = self._count_findings_matched(findings, gt_findings)
            score += 0.25 * (matched / len(gt_findings))

        # ------------------------------------------------------------------ #
        # 4. Entity precision / recall  (weight 0.15)                         #
        # ------------------------------------------------------------------ #
        gt_entities = set(gt.get("key_entities", []))
        flagged_set = set(entities_flagged) if entities_flagged else set()
        excluded = set(gt.get("excluded_entities", []))

        if gt_entities:
            # Penalise incorrectly flagging excluded entities
            false_positives = len(flagged_set & excluded)
            true_positives = len(flagged_set & gt_entities)
            false_negatives = len(gt_entities - flagged_set)

            precision = true_positives / max(len(flagged_set) - false_positives + true_positives, 1)
            recall = true_positives / len(gt_entities)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            score += 0.15 * f1

        # ------------------------------------------------------------------ #
        # 5. Efficiency  (weight 0.15)                                        #
        # ------------------------------------------------------------------ #
        optimal = OPTIMAL_STEPS.get(task_id, 8)
        step_count = state.step_count
        # Full credit at optimal; linear decay; zero credit at MAX_STEPS
        if step_count <= optimal:
            efficiency = 1.0
        else:
            efficiency = max(0.0, 1.0 - (step_count - optimal) / (MAX_STEPS - optimal))
        score += 0.15 * efficiency

<<<<<<< HEAD
        return min(max(round(score, 4), 0.001), 0.999)
=======
        return min(max(round(score, 4), 0.0), 1.0)
>>>>>>> 10edb24 (chore: First iteration of OpenEnv AML Environment ready for submission~)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _count_findings_matched(self, agent_findings: List[str], gt_findings: List[str]) -> int:
        """
        Count how many ground-truth findings appear in the agent's findings.

        Uses a two-tier matching strategy:
        1. Direct keyword overlap (≥50% of significant GT keywords present in any agent finding)
        2. Semantic alias matching via a small synonym map

        This balances strictness (agent must find the right thing) with flexibility
        (different phrasing is OK).
        """
        # Common aliases so slight phrasing differences still match
        ALIASES = {
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
        }

        matched = 0
        normalised_agent = [f.lower().replace("-", "_").replace(" ", "_") for f in agent_findings]
        joined_agent = " ".join(normalised_agent)  # For broad substring search

        for gt_f in gt_findings:
            gt_norm = gt_f.lower().replace("-", "_")
            gt_keywords = [kw for kw in gt_norm.split("_") if len(kw) > 2]

            # Tier 1: ≥50% keyword overlap in any single agent finding
            for af in normalised_agent:
                hits = sum(1 for kw in gt_keywords if kw in af)
                if gt_keywords and hits / len(gt_keywords) >= 0.5:
                    matched += 1
                    break
            else:
                # Tier 2: Check alias table
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
                # Tier 3: Any GT keyword appears as a substring in the joined agent findings
                if any(kw in joined_agent for kw in gt_keywords):
                    matched += 1
        return matched

    def grade_step(
        self,
        tool: str,
        params: Dict[str, Any],
        state: AMLState,
        call_hash: str,
    ) -> float:
        """
        Compute a per-step reward.

        Returns
        -------
        float — typically in [-0.05, +0.05]
        """
        if call_hash in state.tool_call_hashes:
            # Redundant call — same tool + params seen before
            return -0.02

        # All unique tool calls yield a small positive reward
        # (irrelevant tools penalised more)
        return 0.05
ate.tool_call_hashes:
            # Redundant call — same tool + params seen before
            return -0.02

        # All unique tool calls yield a small positive reward
        # (irrelevant tools penalised more)
        return 0.05
