"""
AML Investigation Environment — core environment implementation.

Compatible with the OpenEnv Environment ABC contract:
  reset(seed, episode_id, **kwargs) -> AMLObservation
  step(action, timeout_s, **kwargs) -> AMLObservation
  state -> AMLState  (property)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

# Dual import: try openenv ABC first, fall back to plain ABC
try:
    from openenv.core import Environment  # type: ignore
except ImportError:
    from abc import ABC as Environment  # type: ignore

# Internal imports — dual pattern
try:
    from models import AMLAction, AMLObservation, AMLState
    from scenarios import get_scenario
    from graders.grader import AMLGrader
except ImportError:
    from aml_investigation_env.models import AMLAction, AMLObservation, AMLState
    from aml_investigation_env.scenarios import get_scenario
    from aml_investigation_env.graders.grader import AMLGrader


MAX_STEPS = 25

AVAILABLE_TOOLS = [
    "review_alert",
    "get_customer_profile",
    "query_transactions",
    "check_watchlist",
    "trace_network",
    "check_source_of_funds",
    "assess_risk",
    "file_sar",
    "close_alert",
]


class AMLEnvironment(Environment):
    """
    AML Investigation Environment.

    An agent is given an AML alert and must use the available tools to
    gather evidence and decide whether to file a SAR or close the alert.
    """

    def __init__(self) -> None:
        self._state: AMLState = AMLState()
        self._current_scenario = None
        self._grader = AMLGrader()

    # ------------------------------------------------------------------ #
    # OpenEnv ABC interface                                                #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> AMLState:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AMLObservation:
        """
        Reset the environment and return the initial observation.

        kwargs
        ------
        task_id : str
            One of 'easy', 'medium', 'hard'. Defaults to 'easy'.
        """
        task_id = kwargs.get("task_id", "easy")
        ep_id = episode_id or str(uuid.uuid4())

        self._current_scenario = get_scenario(task_id)
        self._state = AMLState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
        )

        alert = self._current_scenario.initial_alert
        return AMLObservation(
            tool_result={"alert": alert},
            available_tools=AVAILABLE_TOOLS,
            message=(
                f"Episode started. Alert {alert['alert_id']} assigned. "
                f"Investigate and decide: file_sar or close_alert."
            ),
            done=False,
            reward=None,
            metadata={"episode_id": ep_id, "task_id": task_id, "step": 0},
        )

    def step(
        self,
        action: AMLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AMLObservation:
        """Execute a single investigation action and advance the environment state.

        Implements the core action-routing pipeline of the RL environment. Each
        call progresses through four stages:

        Pipeline Stages:
            1. **Guard Checks**: Validates that the environment has been reset and
               the episode has not already terminated. Enforces the MAX_STEPS budget
               with a penalty reward if exceeded.
            2. **Redundancy Detection**: Computes a stable MD5 hash of (tool, params)
               and checks it against previously seen call hashes. Redundant calls
               receive a -0.02 penalty; unique calls earn +0.05.
            3. **Tool Dispatch**: Routes the action to the appropriate scenario-aware
               handler (one of 9 domain tools) via a handler map. Each handler reads
               from the scenario's data layer and updates evidence-tracking flags on
               the AMLState.
            4. **Terminal Grading**: If the action is terminal (``file_sar`` or
               ``close_alert``), invokes the deterministic ``AMLGrader`` to compute
               the composite final score and ends the episode.

        Args:
            action: The agent's tool call, containing a tool name and parameters dict.
            timeout_s: Optional per-step timeout (reserved for future use).
            **kwargs: Additional keyword arguments (forwarded for OpenEnv compatibility).

        Returns:
            An ``AMLObservation`` containing the tool result payload, updated list
            of available tools, a human-readable message, the step reward, and a
            done flag indicating whether the episode has terminated.
        """
        if self._current_scenario is None:
            return AMLObservation(
                message="Environment not reset. Call reset() first.",
                done=True,
                reward=-1.0,
                metadata={"error": "not_reset"},
            )

        if self._state.decision_made:
            return AMLObservation(
                message="Episode already finished.",
                done=True,
                reward=0.0,
                metadata={"step": self._state.step_count},
            )

        self._state.step_count += 1

        # Max steps guard
        if self._state.step_count > MAX_STEPS and not self._state.decision_made:
            self._state.decision_made = True
            penalty = -0.10
            self._state.accumulated_reward += penalty
            return AMLObservation(
                message="Maximum steps reached without a decision. Episode ended with penalty.",
                done=True,
                reward=penalty,
                metadata={
                    "step": self._state.step_count,
                    "final_score": self._state.accumulated_reward,
                    "reason": "max_steps_exceeded",
                },
            )

        tool = action.tool.strip().lower()
        params = action.parameters or {}

        # Compute call hash for redundancy detection
        call_hash = self._compute_hash(tool, params)

        # Compute step reward
        step_reward = self._grader.grade_step(tool, params, self._state, call_hash)
        self._state.accumulated_reward += step_reward

        # Record hash
        if call_hash not in self._state.tool_call_hashes:
            self._state.tool_call_hashes.append(call_hash)

        # Route to handler
        handler_map = {
            "review_alert": self._handle_review_alert,
            "get_customer_profile": self._handle_get_customer_profile,
            "query_transactions": self._handle_query_transactions,
            "check_watchlist": self._handle_check_watchlist,
            "trace_network": self._handle_trace_network,
            "check_source_of_funds": self._handle_check_source_of_funds,
            "assess_risk": self._handle_assess_risk,
            "file_sar": self._handle_file_sar,
            "close_alert": self._handle_close_alert,
        }

        if tool not in handler_map:
            obs = AMLObservation(
                tool_result={"error": f"Unknown tool '{tool}'"},
                available_tools=AVAILABLE_TOOLS,
                message=f"Tool '{tool}' is not available. Choose from: {AVAILABLE_TOOLS}",
                done=False,
                reward=step_reward,
                metadata={"step": self._state.step_count},
            )
            return obs

        try:
            result, message, done = handler_map[tool](params)
        except Exception as exc:
            result = {"error": str(exc)}
            message = f"Tool '{tool}' raised an error: {exc}"
            done = False

        # If terminal action, compute final grader score
        if done:
            final_reward = self._state.accumulated_reward
            return AMLObservation(
                tool_result=result,
                available_tools=[],
                message=message,
                done=True,
                reward=final_reward,
                metadata={
                    "step": self._state.step_count,
                    "final_score": final_reward,
                    "task_id": self._state.task_id,
                },
            )

        return AMLObservation(
            tool_result=result,
            available_tools=AVAILABLE_TOOLS,
            message=message,
            done=False,
            reward=step_reward,
            metadata={"step": self._state.step_count},
        )

    # ------------------------------------------------------------------ #
    # Tool handlers                                                        #
    # ------------------------------------------------------------------ #

    def _handle_review_alert(self, params: Dict[str, Any]):
        """Return the full alert details."""
        self._state.alert_reviewed = True
        result = {"alert": self._current_scenario.initial_alert}
        message = "Alert details retrieved. Review the alert summary and begin your investigation."
        return result, message, False

    def _handle_get_customer_profile(self, params: Dict[str, Any]):
        """Return KYC profile for a customer."""
        customer_id = params.get("customer_id", "")
        profiles = self._current_scenario.customer_profiles
        if customer_id in profiles:
            self._state.customer_profiled = True
            result = {"customer_profile": profiles[customer_id]}
            message = f"Customer profile retrieved for {customer_id}."
        else:
            # Try fuzzy match by name
            matched = None
            for cid, profile in profiles.items():
                if customer_id.lower() in profile.get("name", "").lower():
                    matched = profile
                    self._state.customer_profiled = True
                    break
            if matched:
                result = {"customer_profile": matched}
                message = f"Customer profile retrieved (fuzzy match on '{customer_id}')."
            else:
                result = {
                    "error": f"No customer found with id/name '{customer_id}'",
                    "available_ids": list(profiles.keys()),
                }
                message = f"No profile found for '{customer_id}'."
        return result, message, False

    def _handle_query_transactions(self, params: Dict[str, Any]):
        """Return transactions matching the given filters."""
        self._state.transactions_queried = True
        customer_id = params.get("customer_id", "")
        date_from = params.get("date_from", None) or params.get("date_range_start", None)
        date_to = params.get("date_to", None) or params.get("date_range_end", None)
        min_amount = params.get("min_amount", None)
        max_amount = params.get("max_amount", None)

        if customer_id:
            txns = self._current_scenario.get_transactions_for(
                customer_id, date_from, date_to, min_amount, max_amount
            )
        else:
            txns = self._current_scenario.transactions

        result = {
            "transactions": txns,
            "count": len(txns),
            "filters_applied": {
                "customer_id": customer_id,
                "date_from": date_from,
                "date_to": date_to,
                "min_amount": min_amount,
                "max_amount": max_amount,
            },
        }
        message = f"Found {len(txns)} transaction(s) matching your query."
        return result, message, False

    def _handle_check_watchlist(self, params: Dict[str, Any]):
        """Screen an entity against watchlists."""
        entity_name = params.get("entity_name", params.get("entity", ""))
        list_type = params.get("list_type", "all")

        if entity_name:
            self._state.watchlist_checked.append(entity_name)

        wl = self._current_scenario.watchlist_results
        # Try exact match first, then case-insensitive
        result_data = wl.get(entity_name)
        if result_data is None:
            for key, val in wl.items():
                if key.lower() == entity_name.lower():
                    result_data = val
                    break
        if result_data is None:
            result_data = {
                "entity": entity_name,
                "hit": False,
                "lists_checked": ["OFAC SDN", "PEP", "UN Sanctions"],
                "result": f"No record found for '{entity_name}' in watchlist database.",
            }

        result = {"watchlist_result": result_data}
        hit_str = "HIT — review required" if result_data.get("hit") else "No matches"
        message = f"Watchlist check for '{entity_name}': {hit_str}."
        return result, message, False

    def _handle_trace_network(self, params: Dict[str, Any]):
        """Trace entity network connections."""
        entity_id = params.get("entity_id", params.get("entity", ""))
        depth = int(params.get("depth", 1))

        self._state.network_traced = True
        graph = self._current_scenario.network_graph

        result_data = graph.get(entity_id)
        if result_data is None:
            # Try case-insensitive
            for key, val in graph.items():
                if key.lower() == entity_id.lower():
                    result_data = val
                    break
        if result_data is None:
            result_data = {
                "entity_id": entity_id,
                "connections": [],
                "note": f"No network data found for entity '{entity_id}'.",
            }

        # At depth >= 2, also include depth_2_connections if present
        if depth >= 2 and "depth_2_connections" in result_data:
            result_data = dict(result_data)
            result_data["connections"] = (
                result_data.get("depth_1_connections", result_data.get("connections", []))
                + result_data.get("depth_2_connections", [])
            )

        result = {"network": result_data, "depth": depth}
        conn_count = len(result_data.get("connections", []))
        message = f"Network trace for '{entity_id}' (depth {depth}): {conn_count} connection(s) found."
        return result, message, False

    def _handle_check_source_of_funds(self, params: Dict[str, Any]):
        """Check source of funds documentation for a transaction."""
        transaction_id = params.get("transaction_id", params.get("transaction", ""))

        if transaction_id:
            self._state.source_checked.append(transaction_id)

        sof = self._current_scenario.source_of_funds
        result_data = sof.get(transaction_id)
        if result_data is None:
            result_data = {
                "transaction_id": transaction_id,
                "source": "unknown",
                "verified": False,
                "notes": f"No source-of-funds record found for transaction '{transaction_id}'.",
            }

        result = {"source_of_funds": result_data}
        verified = "Verified" if result_data.get("verified") else "NOT verified"
        message = f"Source of funds for transaction '{transaction_id}': {verified}."
        return result, message, False

    def _handle_assess_risk(self, params: Dict[str, Any]):
        """Return a computed risk assessment based on gathered evidence."""
        self._state.risk_assessed = True
        customer_id = params.get("customer_id", "")

        # Build risk score based on state flags
        risk_factors = []
        risk_score = 20  # baseline

        if self._state.alert_reviewed:
            risk_score += 5
        if self._state.transactions_queried:
            risk_score += 10
            risk_factors.append("Unusual transaction pattern noted")
        if self._state.watchlist_checked:
            if any("PEP" in w or "Viktor" in w or "Korev" in w for w in self._state.watchlist_checked):
                risk_score += 30
                risk_factors.append("PEP connection identified")
        if self._state.network_traced:
            risk_score += 15
            risk_factors.append("Network connections traced")
        if self._state.source_checked:
            risk_score += 10
            risk_factors.append("Source of funds reviewed")
        if self._state.findings:
            risk_score += 5 * len(self._state.findings)
            risk_factors.extend(self._state.findings)

        risk_score = min(risk_score, 100)

        if risk_score >= 75:
            risk_level = "CRITICAL"
            recommendation = "SAR filing strongly recommended"
        elif risk_score >= 50:
            risk_level = "HIGH"
            recommendation = "SAR filing recommended"
        elif risk_score >= 25:
            risk_level = "MEDIUM"
            recommendation = "Enhanced due diligence required"
        else:
            risk_level = "LOW"
            recommendation = "Monitor; no immediate action required"

        result = {
            "customer_id": customer_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "risk_factors": risk_factors,
            "evidence_gathered": {
                "alert_reviewed": self._state.alert_reviewed,
                "customer_profiled": self._state.customer_profiled,
                "transactions_queried": self._state.transactions_queried,
                "watchlist_checked": self._state.watchlist_checked,
                "network_traced": self._state.network_traced,
                "source_checked": self._state.source_checked,
            },
        }
        message = f"Risk assessment complete. Risk level: {risk_level} (score: {risk_score}/100). {recommendation}."
        return result, message, False

    def _handle_file_sar(self, params: Dict[str, Any]):
        """Terminal action — file a SAR and compute the final score."""
        self._state.decision_made = True
        findings = params.get("findings", [])
        typology = params.get("typology", "")
        entities_involved = params.get("entities_involved", params.get("entities", []))

        if isinstance(findings, str):
            findings = [findings]
        if isinstance(entities_involved, str):
            entities_involved = [entities_involved]

        # Record findings in state
        self._state.findings.extend(findings)

        # Grade
        final_score = self._grader.grade(
            task_id=self._state.task_id,
            decision="file_sar",
            findings=findings,
            entities_flagged=entities_involved,
            typology=typology,
            state=self._state,
        )
        self._state.accumulated_reward = final_score

        result = {
            "sar_filed": True,
            "typology": typology,
            "entities_flagged": entities_involved,
            "findings_submitted": findings,
            "final_score": final_score,
            "grader_breakdown": self._build_grader_breakdown("file_sar", findings, entities_involved, typology, final_score),
        }
        message = (
            f"SAR filed successfully. Final episode score: {final_score:.4f}. "
            f"Typology: {typology}. Entities flagged: {entities_involved}."
        )
        return result, message, True

    def _handle_close_alert(self, params: Dict[str, Any]):
        """Terminal action — close alert as false positive and compute final score."""
        self._state.decision_made = True
        reason = params.get("reason", "")
        findings = params.get("findings", [])
        entities_involved = params.get("entities_involved", [])
        typology = params.get("typology", "false_positive")

        if isinstance(findings, str):
            findings = [findings]

        self._state.findings.extend(findings)

        final_score = self._grader.grade(
            task_id=self._state.task_id,
            decision="close_alert",
            findings=findings,
            entities_flagged=entities_involved,
            typology=typology,
            state=self._state,
        )
        self._state.accumulated_reward = final_score

        result = {
            "alert_closed": True,
            "reason": reason,
            "final_score": final_score,
            "grader_breakdown": self._build_grader_breakdown("close_alert", findings, entities_involved, typology, final_score),
        }
        message = (
            f"Alert closed as false positive. Reason: {reason}. "
            f"Final episode score: {final_score:.4f}."
        )
        return result, message, True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_hash(tool: str, params: Dict[str, Any]) -> str:
        """Stable hash of a tool call for redundancy detection."""
        payload = json.dumps({"tool": tool, "params": params}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _build_grader_breakdown(
        self,
        decision: str,
        findings: List[str],
        entities: List[str],
        typology: str,
        final_score: float,
    ) -> Dict[str, Any]:
        """Human-readable breakdown of what contributed to the final score."""
        scenario = self._current_scenario
        gt = scenario.ground_truth

        gt_findings = gt.get("key_findings", [])
        gt_entities = set(gt.get("key_entities", []))

        matched_findings = sum(
            1 for f in gt_findings
            if any(kw in " ".join(findings).lower() for kw in f.split("_"))
        )

        return {
            "decision_correct": decision == gt["correct_decision"],
            "typology_correct": typology.lower().strip() == gt.get("typology", "").lower(),
            "findings_matched": f"{matched_findings}/{len(gt_findings)}",
            "entities_flagged": entities,
            "entities_expected": list(gt_entities),
            "step_count": self._state.step_count,
            "final_score": final_score,
        }
