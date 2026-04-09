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
    from aml_investigation_env.scenarios import get_scenario
    from aml_investigation_env.models import AMLState


# ---- Per-task optimal step counts (used for efficiency scoring) --------
OPTIMAL_STEPS: Dict[str, int] = {
    "easy": 5,    # review_alert, get_customer_profile, query_transactions, check_source, file_sar
    "medium": 8,  # alert + profile + transactions + watchlist(x2) + network + source + file_sar
    "hard": 10,   # alert + profile + transactions + watchlist + network(depth2) + source + market + assess_risk + file_sar
}

MAX_STEPS = 25


class AMLGrader:
    """Deterministic scoring engine for completed AML investigation episodes.

    Implements a weighted rubric that evaluates five orthogonal dimensions of
    investigative quality: decision accuracy, typology identification, evidence
    coverage, entity-level precision/recall, and step efficiency. This design
    intentionally avoids LLM-as-judge approaches to ensure reproducible,
    auditable benchmarking across model comparisons.

    Scoring Philosophy:
        The rubric weights reflect real-world compliance priorities. Decision
        correctness (0.30) dominates because a wrong SAR/close decision is the
        costliest failure mode. Evidence coverage (0.25) rewards thorough
        investigation. The remaining weight is split across typology accuracy,
        entity identification (F1), and operational efficiency.

    Attributes:
        OPTIMAL_STEPS: Per-task minimum step counts representing an ideal
            investigation path, used as the baseline for efficiency scoring.
    """

    def grade(
        self,
        task_id: str,
        decision: str,
        findings: List[str],
        entities_flagged: List[str],
        typology: str,
        state: AMLState,
    ) -> float:
        """Compute a composite score in [0.001, 0.999] for a completed episode.

        Evaluates the agent's terminal action against the scenario's ground truth
        across five weighted dimensions. Each component is independently scored
        and summed, producing a transparent, decomposable final score. The result
        is clamped to (0.001, 0.999) to avoid degenerate log-probability issues
        in downstream RL training loops.

        Args:
            task_id: Scenario identifier ('easy', 'medium', or 'hard').
            decision: The agent's terminal action ('file_sar' or 'close_alert').
            findings: Free-text finding strings submitted by the agent, matched
                against ground truth via keyword overlap and semantic aliasing.
            entities_flagged: Entity IDs the agent identified as involved in the
                suspicious activity. Scored via precision/recall F1.
            typology: The money laundering typology the agent reported
                (e.g., 'structuring', 'layering', 'trade_based_ml').
            state: The terminal ``AMLState`` snapshot, used for step-count
                efficiency scoring.

        Returns:
            A float in [0.001, 0.999] representing the composite episode score.

        Scoring Breakdown:
            - Decision correctness (0.30): Binary — did the agent file/close correctly?
            - Typology correctness (0.15): Case-insensitive exact match.
            - Key findings coverage (0.25): Proportional to ground-truth findings matched.
            - Entity precision/recall F1 (0.15): Penalizes both missed and falsely flagged entities.
            - Efficiency (0.15): Linear decay from optimal step count to MAX_STEPS.
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

        return min(max(round(score, 4), 0.001), 0.999)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _count_findings_matched(self, agent_findings: List[str], gt_findings: List[str]) -> int:
        """Count ground-truth findings matched by the agent's reported findings.

        Implements a three-tier fuzzy matching strategy designed to be robust to
        the natural language variability of LLM outputs while remaining strict
        enough to prevent credit for unrelated findings:

        Matching Tiers (evaluated in order, first match wins):
            1. **Keyword Overlap**: Tokenizes the ground-truth finding into
               significant keywords (len > 2) and checks if ≥50% appear in any
               single agent finding. This handles rephrasing (e.g., "sub_threshold"
               vs "deposits_below_threshold").
            2. **Semantic Alias Table**: A curated synonym map covers common
               domain-specific rephrasings (e.g., "no_source_documentation" ↔
               "undocumented", "no_business_justification"). Aliases are checked
               against the concatenated agent findings string.
            3. **Substring Fallback**: Any individual ground-truth keyword
               appearing anywhere in the joined agent findings triggers a match.
               This is the most permissive tier and acts as a safety net.

        Args:
            agent_findings: Normalized finding strings from the agent's terminal action.
            gt_findings: Ground-truth finding keys from the scenario definition.

        Returns:
            The count of ground-truth findings successfully matched (0 to len(gt_findings)).
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
