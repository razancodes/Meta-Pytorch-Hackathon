"""
Memex OS-Agent Benchmark — Dense Reward Grader.

Implements the two-tier reward system:

  1. **Per-Step Micro-Rewards** (grade_step):
     - Action cost, redundancy penalty, page fault, async timeout,
       successful page, meta-injection.

  2. **Terminal Composite Score** (grade_terminal):
     - Decision correctness, typology, findings coverage, entity F1,
       efficiency — mapped to [-1.0, +1.0] range.

The grader no longer calls get_scenario() at terminal time — it receives
the ground_truth dict directly from the environment to avoid re-generating
a different procedural scenario.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

try:
    from models import AMLState
except ImportError:
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

# Efficiency
MAX_STEPS: int = 25

# Common aliases for fuzzy finding matching
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
    # Phase 3: FinCEN 4-pillar aliases
    "shared_device_fingerprint": ["device_overlap", "shared_device", "mule_ring", "device_fingerprint"],
    "ip_jurisdiction_mismatch": ["vpn", "ip_mismatch", "geo_mismatch", "jurisdiction_mismatch"],
    "phantom_shipment": ["phantom", "zero_weight", "no_bill_of_lading", "missing_bol"],
}


class AMLGrader:
    """Dense reward grader for the Memex OS-Agent benchmark.

    grade_step()    — called per step for micro-rewards.
    grade()         — called at terminal for the composite episode score.
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
        """Compute the per-step micro-reward."""
        reward = ACTION_COST

        if call_hash in state.tool_call_hashes:
            reward += REDUNDANT_PENALTY

        if is_page_fault:
            reward += PAGE_FAULT_PENALTY
        if is_async_timeout:
            reward += ASYNC_TIMEOUT_PENALTY
        if is_successful_page:
            reward += SUCCESSFUL_PAGE
        if is_meta_injection:
            reward += META_INJECTION

        if call_hash not in state.tool_call_hashes and not is_page_fault:
            reward += UNIQUE_TOOL_BONUS

        return round(reward, 4)

    # ------------------------------------------------------------------ #
    # Terminal Grading                                                     #
    # ------------------------------------------------------------------ #

    def grade(
        self,
        ground_truth: Dict[str, Any],
        decision: str,
        findings: List[str],
        entities_flagged: List[str],
        typology: str,
        state: AMLState,
        optimal_steps: int = 10,
        ubo_identified: str | None = None,
    ) -> float:
        """Compute the final episode score in [-1.0, +1.0].

        Args:
            ground_truth: The scenario's ground truth dict.
            decision: "file_sar" or "close_alert".
            findings: Agent-identified findings.
            entities_flagged: Agent-flagged entity IDs.
            typology: Agent-identified typology.
            state: Current AMLState with accumulated_reward.
            optimal_steps: Expected optimal step count.
            ubo_identified: Phase 3 — UBO entity ID identified by agent.
        """
        gt = ground_truth

        # 1. Decision correctness (weight 0.30)
        decision_correct = decision == gt["correct_decision"]
        decision_score = 0.30 if decision_correct else -0.50

        # 2. Typology correctness (weight 0.15)
        typology_score = 0.0
        if typology and typology.lower().strip() == gt.get("typology", "").lower().strip():
            typology_score = 0.15

        # 3. Key findings coverage (weight 0.20)
        gt_findings = gt.get("key_findings", [])
        findings_score = 0.0
        if gt_findings:
            matched = self._count_findings_matched(findings, gt_findings)
            findings_score = 0.20 * (matched / len(gt_findings))

        # 4. Entity precision/recall F1 (weight 0.15)
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

        # 5. UBO identification (weight 0.05) — Phase 3
        ubo_score = 0.0
        gt_ubo = gt.get("ubo_entity_id")
        if gt_ubo and ubo_identified:
            ubo_norm = str(ubo_identified).lower().strip()
            gt_ubo_norm = str(gt_ubo).lower().strip()
            if ubo_norm == gt_ubo_norm:
                ubo_score = 0.05
            else:
                ubo_score = -0.03  # Penalty for wrong UBO

        # 6. Phase 3 pillar tool usage bonus (weight 0.05)
        pillar_score = 0.0
        pillar_checks = [
            getattr(state, "device_overlap_checked", False),
            getattr(state, "customs_invoice_verified", False),
            getattr(state, "beneficial_ownership_queried", False),
        ]
        pillar_used = sum(1 for c in pillar_checks if c)
        pillar_score = 0.05 * (pillar_used / 3.0)

        # 7. Efficiency (weight 0.10)
        step_count = state.step_count
        if step_count <= optimal_steps:
            efficiency = 1.0
        else:
            efficiency = max(0.0, 1.0 - (step_count - optimal_steps) / (MAX_STEPS - optimal_steps))
        efficiency_score = 0.10 * efficiency

        # Composite
        terminal_score = (
            decision_score + typology_score + findings_score
            + entity_score + ubo_score + pillar_score + efficiency_score
        )

        # Add accumulated step rewards (micro-rewards from OS mechanics)
        total = terminal_score + state.accumulated_reward

        return round(max(-1.0, min(1.0, total)), 4)

    # ------------------------------------------------------------------ #
    # Findings Matching                                                    #
    # ------------------------------------------------------------------ #

    def _count_findings_matched(self, agent_findings: List[str], gt_findings: List[str]) -> int:
        """Count ground-truth findings matched by the agent's findings."""
        matched = 0
        normalised_agent = [f.lower().replace("-", "_").replace(" ", "_") for f in agent_findings]
        joined_agent = " ".join(normalised_agent)

        for gt_f in gt_findings:
            gt_norm = gt_f.lower().replace("-", "_")
            gt_keywords = [kw for kw in gt_norm.split("_") if len(kw) > 2]

            # Tier 1: ≥50% keyword overlap
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

            # Tier 3: Substring fallback
            if any(kw in joined_agent for kw in gt_keywords):
                matched += 1

        return matched
