"""
Memex OS-Agent Benchmark — Dense Reward Grader.

Implements the two-tier reward system:

  1. **Per-Step Micro-Rewards** (grade_step):
     - Action cost, redundancy penalty, page fault, async timeout,
       successful page, meta-injection.

  2. **Terminal Composite Score** (grade_terminal / grade):
     - Detection (TP/TN/FP/FN based on is_suspicious ground truth)
     - Entity-level F1
     - Typology correctness
     - Efficiency
     - OS component (page faults, case writes, async, kernel modes)

All weights are explicit in RewardWeights and documented. The composition
is designed so that lazy policies (always SAR or always close) have lower
expected reward than a reasonable policy.

NOTE: Reward farming hard caps are enforced at the environment level
(aml_environment.py), not here. The environment suppresses the
is_successful_page flag after 3 rewarded writes and is_meta_injection
after 2 rewarded injections per episode. The grader itself is stateless
w.r.t. these caps — it trusts the flags it receives.

The grader no longer calls get_scenario() at terminal time — it receives
the ground_truth dict directly from the environment to avoid re-generating
a different procedural scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Set, runtime_checkable


# The grader uses duck-typing for state: it only reads .accumulated_reward,
# .step_count, and .tool_call_hashes. No hard dependency on models.py.
@runtime_checkable
class _StateProtocol(Protocol):
    accumulated_reward: float
    step_count: int
    tool_call_hashes: List[str]



# Reward Weights — Explicit & Configurable

@dataclass
class RewardWeights:
    """Composite reward weights for the Defender terminal score.

    Constraints (enforced by design, not runtime):
      - detect MUST be the largest weight (main task).
      - Under 70% suspicious / 30% clean scenario mix:
          * "Always SAR" policy:  E[R] = 0.7*(1.0) + 0.3*(-0.75) = 0.475
          * "Always close" policy: E[R] = 0.7*(-2.0) + 0.3*(0.5) = -1.25
          * Reasonable policy:     E[R] ≈ 0.7*(0.8) + 0.3*(0.4) = 0.68
        So lazy policies always score lower than reasonable ones.
    """
    detect: float = 1.0       # Detection: TP/TN/FP/FN (largest weight)
    entity_f1: float = 0.5    # Entity-level precision/recall F1
    typology: float = 0.3     # Typology label accuracy
    efficiency: float = 0.2   # Step efficiency + budget bonus
    os: float = 0.2           # OS mechanics (page faults, async, kernel)


# Detection Component Constants

# TP/TN/FP/FN rewards for the detection component (R_detect)
R_TP: float = +1.00    # Suspicious + SAR filed (correct)
R_TN: float = +0.50    # Clean + alert closed (correct)
R_FN: float = -2.00    # Suspicious + alert closed (missed ML)
R_FP: float = -0.75    # Clean + SAR filed (false alarm)

# Per-Step Constants

ACTION_COST:            float = -0.02   # Every step
REDUNDANT_PENALTY:      float = -0.03   # Duplicate tool+params
PAGE_FAULT_PENALTY:     float = -0.05   # Evicted data not on disk
ASYNC_TIMEOUT_PENALTY:  float = -0.10   # Premature async retrieval
SUCCESSFUL_PAGE:        float = +0.10   # Good write_to_case_file
META_INJECTION:         float = +0.15   # Successful kernel update
UNIQUE_TOOL_BONUS:      float = +0.03   # Novel, useful tool call

# Investigation progress bonuses — first use of each tool TYPE only.
# These make positive reward discoverable through random exploration.
# Small enough (~0.19 total cap) not to dominate terminal reward (~±1.0).
INVESTIGATION_BONUSES: Dict[str, float] = {
    "review_alert": 0.03,
    "get_customer_profile": 0.02,
    "query_transactions": 0.02,
    "check_watchlist": 0.02,
    "trace_network": 0.02,
    "check_source_of_funds": 0.02,
    "write_to_case_file": 0.03,
    "file_sar": 0.05,
    "close_alert": 0.05,
}

# OS component sub-weights (per-event)
OS_PAGE_FAULT_COST:         float = -0.05   # Per page fault
OS_CASE_WRITE_BONUS:        float = +0.02   # Per successful write_to_case_file on critical entity
OS_ASYNC_PREMATURE_COST:    float = -0.01   # Per premature async poll
OS_ASYNC_CORRECT_BONUS:     float = +0.01   # Per correct async poll
OS_KERNEL_MODE_BONUS:       float = +0.05   # If ≥1 kernel mode used AND R_detect > 0

# Efficiency
STEP_COST:          float = -0.005
BUDGET_BONUS:       float = +0.20
BUDGET_THRESHOLD:   int   = 15      # Complete under this many steps for bonus
MAX_STEPS:          int   = 25

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

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or RewardWeights()

    # ------------------------------------------------------------------ #
    # Per-Step Rewards                                                     #
    # ------------------------------------------------------------------ #

    def grade_step(
        self,
        tool: str,
        params: Dict[str, Any],
        state: _StateProtocol,
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
        state: _StateProtocol,
        optimal_steps: int = 10,
        ubo_identified: str | None = None,
        os_metrics: Dict[str, int] | None = None,
    ) -> Dict[str, Any]:
        """Compute the final episode score with explicit component breakdown.

        Returns a dict with:
            total:       float — clipped composite reward in [-2.0, +2.0]
            components:  dict  — individual component rewards
            detection:   str   — "TP", "TN", "FP", or "FN"
            os_metrics:  dict  — OS metric counts used in reward

        The returned total is what the PPO trainer uses as the terminal reward.
        """
        gt = ground_truth
        w = self.w

        is_suspicious = gt.get("is_suspicious", True)
        is_sar = decision == "file_sar"

        # ── 1. Detection Component (R_detect) ──
        if is_suspicious and is_sar:
            r_detect = R_TP
            detection_label = "TP"
        elif not is_suspicious and not is_sar:
            r_detect = R_TN
            detection_label = "TN"
        elif is_suspicious and not is_sar:
            r_detect = R_FN
            detection_label = "FN"
        else:  # not suspicious and SAR
            r_detect = R_FP
            detection_label = "FP"

        # ── 2. Entity-F1 Component (R_entityF1) ──
        gt_entities: Set[str] = set(gt.get("key_entities", []))
        flagged_set: Set[str] = set(entities_flagged) if entities_flagged else set()

        if gt_entities:
            true_positives = len(flagged_set & gt_entities)
            precision = true_positives / max(len(flagged_set), 1)
            recall = true_positives / len(gt_entities)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        elif not is_suspicious:
            # Clean scenario: no entities to flag. F1 = 1.0 if agent flagged nothing.
            f1 = 1.0 if len(flagged_set) == 0 else 0.0
        else:
            f1 = 0.0

        # Scale to [-1, 1]: R_entityF1 = 2*F1 - 1
        r_entity_f1 = 2.0 * f1 - 1.0

        # ── 3. Typology Correctness (R_typology) ──
        TYPOLOGY_ALIASES = {
            "phantom_invoice": "trade_based_ml",
            "mule_ring": "layering",
            "pass_through": "layering",
        }
        agent_typo = TYPOLOGY_ALIASES.get(
            typology.lower().strip(), typology.lower().strip()
        ) if typology else ""
        gt_typo = TYPOLOGY_ALIASES.get(
            gt.get("typology", "").lower().strip(),
            gt.get("typology", "").lower().strip(),
        )

        if not is_suspicious:
            # Clean scenario: reward if agent recognized it as clean
            r_typology = 0.5 if agent_typo in ("clean", "false_positive", "") else 0.0
        elif agent_typo and agent_typo == gt_typo:
            r_typology = 0.5   # All correct
        elif agent_typo and gt_typo and agent_typo != gt_typo:
            r_typology = 0.0   # Wrong typology
        else:
            r_typology = 0.0

        # ── 4. Findings Coverage ──
        gt_findings = gt.get("key_findings", [])
        if gt_findings:
            matched = self._count_findings_matched(findings, gt_findings)
            findings_ratio = matched / len(gt_findings)
        else:
            # Clean scenario: no findings expected. Full credit if agent found nothing bad.
            findings_ratio = 1.0 if len(findings) == 0 else 0.5

        # ── 5. Efficiency Component (R_eff) ──
        step_count = state.step_count
        r_eff = STEP_COST * step_count
        if step_count <= BUDGET_THRESHOLD:
            r_eff += BUDGET_BONUS

        # ── 6. OS Component (R_OS) ──
        om = os_metrics or {}
        page_fault_count = om.get("page_fault_count", 0)
        case_writes_critical = om.get("case_writes_critical", 0)
        async_premature_polls = om.get("async_premature_polls", 0)
        async_successful_polls = om.get("async_successful_polls", 0)
        kernel_mode_uses = om.get("kernel_mode_uses", 0)

        r_os = (
            OS_PAGE_FAULT_COST * page_fault_count
            + OS_CASE_WRITE_BONUS * case_writes_critical
            + OS_ASYNC_PREMATURE_COST * async_premature_polls
            + OS_ASYNC_CORRECT_BONUS * async_successful_polls
        )
        # Kernel mode bonus: only if detection was correct
        if kernel_mode_uses >= 1 and r_detect > 0:
            r_os += OS_KERNEL_MODE_BONUS

        # ── 7. UBO Identification (Phase 3 bonus) ──
        r_ubo = 0.0
        gt_ubo = gt.get("ubo_entity_id")
        if gt_ubo and ubo_identified:
            if str(ubo_identified).lower().strip() == str(gt_ubo).lower().strip():
                r_ubo = 0.05
            else:
                r_ubo = -0.03

        # ── Composite Reward ──
        # Findings ratio is folded into entity_f1 weight as sub-component
        r_total = (
            w.detect     * r_detect
            + w.entity_f1  * (r_entity_f1 * 0.6 + (2.0 * findings_ratio - 1.0) * 0.4)  # F1 + findings
            + w.typology   * r_typology
            + w.efficiency * r_eff
            + w.os         * r_os
            + r_ubo  # Small, unweighted bonus
        )

        # Add accumulated per-step micro-rewards
        r_total += state.accumulated_reward

        # Clip to safe range
        r_total = round(max(-2.0, min(2.0, r_total)), 4)

        return {
            "total": r_total,
            "detection": detection_label,
            "components": {
                "r_detect": round(r_detect, 4),
                "r_entity_f1": round(r_entity_f1, 4),
                "r_typology": round(r_typology, 4),
                "r_findings_ratio": round(findings_ratio, 4),
                "r_efficiency": round(r_eff, 4),
                "r_os": round(r_os, 4),
                "r_ubo": round(r_ubo, 4),
                "accumulated_step_reward": round(state.accumulated_reward, 4),
            },
            "entity_f1": round(f1, 4),
            "os_metrics": {
                "page_fault_count": page_fault_count,
                "case_writes_critical": case_writes_critical,
                "async_premature_polls": async_premature_polls,
                "async_successful_polls": async_successful_polls,
                "kernel_mode_uses": kernel_mode_uses,
            },
            "weights": {
                "detect": w.detect,
                "entity_f1": w.entity_f1,
                "typology": w.typology,
                "efficiency": w.efficiency,
                "os": w.os,
            },
        }

    # Backward-compatible total-only score (used by existing train_ppo.py)
    def grade_total(
        self,
        ground_truth: Dict[str, Any],
        decision: str,
        findings: List[str],
        entities_flagged: List[str],
        typology: str,
        state: _StateProtocol,
        optimal_steps: int = 10,
        ubo_identified: str | None = None,
        os_metrics: Dict[str, int] | None = None,
    ) -> float:
        """Return just the scalar reward for backward compatibility."""
        result = self.grade(
            ground_truth, decision, findings, entities_flagged,
            typology, state, optimal_steps, ubo_identified, os_metrics,
        )
        return result["total"]

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

            # Tier 3: Substring fallback (require ≥2 keyword matches to avoid false positives)
            matching_kw = sum(1 for kw in gt_keywords if kw in joined_agent)
            if matching_kw >= 2:
                matched += 1

        return matched
