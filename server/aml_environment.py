"""
Memex OS-Agent Benchmark — Core Environment.

Integrates the StateManager (OS mechanics) with the AML scenario data and tool
dispatch. Implements the OpenEnv contract: reset() / step() / state property.

Tool Roster (15 tools):
  Legacy domain tools (9):
    review_alert, get_customer_profile, query_transactions, check_watchlist,
    trace_network, check_source_of_funds, assess_risk, file_sar, close_alert

  OS-Mechanic tools (6):
    write_to_case_file, request_wire_trace, retrieve_async_result,
    search_compliance_manual, update_system_prompt, check_market_price
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Dual import: OpenEnv ABC or plain ABC
try:
    from openenv.core import Environment  # type: ignore
except ImportError:
    from abc import ABC as Environment  # type: ignore

# Internal imports — dual pattern
try:
    from models import AMLAction, AMLObservation, AMLState
    from scenarios import get_scenario
    from scenarios.compliance_manual import search_compliance_manual
    from graders.grader import AMLGrader
    from state_manager import StateManager
except ImportError:
    from aml_investigation_env.models import AMLAction, AMLObservation, AMLState
    from aml_investigation_env.scenarios import get_scenario
    from aml_investigation_env.scenarios.compliance_manual import search_compliance_manual
    from aml_investigation_env.graders.grader import AMLGrader
    from aml_investigation_env.state_manager import StateManager


MAX_STEPS = 25

AVAILABLE_TOOLS: List[str] = [
    # Domain investigation tools
    "review_alert",
    "get_customer_profile",
    "query_transactions",
    "check_watchlist",
    "trace_network",
    "check_source_of_funds",
    "assess_risk",
    "file_sar",
    "close_alert",
    # OS-Mechanic tools
    "write_to_case_file",
    "request_wire_trace",
    "retrieve_async_result",
    "search_compliance_manual",
    "update_system_prompt",
    "check_market_price",
]


class AMLEnvironment(Environment):
    """Memex AML Investigation Environment with OS mechanics.

    Layers three OS subsystems on the AML investigation:
      I.   Virtual Memory — RAM eviction forces paging to disk
      II.  Interrupts — Async wire traces with ETA countdown
      III. Kernel Updates — Agent can inject compliance rules into its prompt
    """

    def __init__(self) -> None:
        self._state: AMLState = AMLState()
        self._current_scenario: Any = None
        self._grader: AMLGrader = AMLGrader()
        self._sm: Optional[StateManager] = None

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
        """Reset the environment and return the initial observation."""
        task_id = kwargs.get("task_id", "easy")
        ep_id = episode_id or str(uuid.uuid4())

        if seed is not None:
            random.seed(seed)

        self._current_scenario = get_scenario(task_id)
        self._state = AMLState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
        )

        # Initialize StateManager
        self._sm = StateManager()

        alert = self._current_scenario.initial_alert
        alert_summary = (
            f"Alert {alert['alert_id']}: {alert.get('summary', '')} "
            f"Customer: {alert.get('customer_id', 'N/A')}"
        )

        # Push initial observation into RAM
        self._sm.push_observation(alert_summary)
        self._sm.sync_to_state(self._state)

        return AMLObservation(
            tool_result={"alert": alert},
            available_tools=AVAILABLE_TOOLS,
            message=(
                f"Episode started. Alert {alert['alert_id']} assigned. "
                f"Investigate and decide: file_sar or close_alert. "
                f"[RAM: 1/{2} | Disk: 0 entries | Async: 0 jobs | Kernel: 1 directive]"
            ),
            done=False,
            reward=None,
            metadata={
                "episode_id": ep_id,
                "task_id": task_id,
                "step": 0,
                "agui_state": self._sm.build_agui_state(),
            },
        )

    def step(
        self,
        action: AMLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AMLObservation:
        """Execute a single investigation action and advance the environment."""
        if self._current_scenario is None or self._sm is None:
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

        # --- OS Mechanic: Tick async jobs (decrement ETAs) ---
        self._sm.tick_async_jobs()

        # Max steps guard
        if self._state.step_count > MAX_STEPS:
            self._state.decision_made = True
            penalty = -0.10
            self._state.accumulated_reward += penalty
            self._sm.sync_to_state(self._state)
            return AMLObservation(
                message="Maximum steps reached without a decision. Episode ended with penalty.",
                done=True,
                reward=self._state.accumulated_reward,
                metadata={
                    "step": self._state.step_count,
                    "final_score": self._state.accumulated_reward,
                    "reason": "max_steps_exceeded",
                    "agui_state": self._sm.build_agui_state(),
                },
            )

        tool = action.tool.strip().lower()
        params = action.parameters or {}

        # Compute call hash for redundancy detection
        call_hash = self._compute_hash(tool, params)

        # --- OS Mechanic: Check page fault ---
        is_page_fault = self._sm.check_page_fault(tool, params)
        if is_page_fault:
            self._state.page_fault_count += 1

        # --- Route to handler ---
        handler_map = {
            # Domain tools
            "review_alert": self._handle_review_alert,
            "get_customer_profile": self._handle_get_customer_profile,
            "query_transactions": self._handle_query_transactions,
            "check_watchlist": self._handle_check_watchlist,
            "trace_network": self._handle_trace_network,
            "check_source_of_funds": self._handle_check_source_of_funds,
            "assess_risk": self._handle_assess_risk,
            "file_sar": self._handle_file_sar,
            "close_alert": self._handle_close_alert,
            # OS-Mechanic tools
            "write_to_case_file": self._handle_write_to_case_file,
            "request_wire_trace": self._handle_request_wire_trace,
            "retrieve_async_result": self._handle_retrieve_async_result,
            "search_compliance_manual": self._handle_search_compliance_manual,
            "update_system_prompt": self._handle_update_system_prompt,
            "check_market_price": self._handle_check_market_price,
        }

        is_async_timeout = False
        is_successful_page = False
        is_meta_injection = False

        if tool not in handler_map:
            step_reward = self._grader.grade_step(
                tool, params, self._state, call_hash,
                is_page_fault=is_page_fault,
            )
            self._state.accumulated_reward += step_reward
            if call_hash not in self._state.tool_call_hashes:
                self._state.tool_call_hashes.append(call_hash)
            self._sm.sync_to_state(self._state)
            return AMLObservation(
                tool_result={"error": f"Unknown tool '{tool}'"},
                available_tools=AVAILABLE_TOOLS,
                message=f"Tool '{tool}' is not available. Choose from: {AVAILABLE_TOOLS}",
                done=False,
                reward=step_reward,
                metadata={
                    "step": self._state.step_count,
                    "agui_state": self._sm.build_agui_state(),
                },
            )

        try:
            result, message, done, flags = handler_map[tool](params)
            is_async_timeout = flags.get("async_timeout", False)
            is_successful_page = flags.get("successful_page", False)
            is_meta_injection = flags.get("meta_injection", False)
        except Exception as exc:
            result = {"error": str(exc)}
            message = f"Tool '{tool}' raised an error: {exc}"
            done = False
            flags = {}

        # Compute step reward with all OS mechanic flags
        step_reward = self._grader.grade_step(
            tool, params, self._state, call_hash,
            is_page_fault=is_page_fault,
            is_async_timeout=is_async_timeout,
            is_successful_page=is_successful_page,
            is_meta_injection=is_meta_injection,
        )
        self._state.accumulated_reward += step_reward

        # Record hash
        if call_hash not in self._state.tool_call_hashes:
            self._state.tool_call_hashes.append(call_hash)

        # Update OS mechanic counters
        if is_async_timeout:
            self._state.async_timeout_count += 1
        if is_successful_page:
            self._state.successful_pages += 1
        if is_meta_injection:
            self._state.meta_injections += 1

        # --- OS Mechanic: Push observation summary into RAM ---
        obs_summary = f"Step {self._state.step_count} [{tool}]: {message}"
        evicted = self._sm.push_observation(obs_summary)

        # Terminal grading
        if done:
            self._state.decision_made = True
            final_reward = self._state.accumulated_reward
            self._sm.sync_to_state(self._state)
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
                    "agui_state": self._sm.build_agui_state(),
                },
            )

        # Build status line
        ram_count = len(self._sm.ram_contents)
        disk_count = len(self._sm.disk_contents)
        async_count = len(self._sm.active_jobs)
        kernel_count = len(self._sm.kernel_directives)
        eviction_notice = " [!RAM EVICTION — page important data to disk!]" if evicted else ""

        self._sm.sync_to_state(self._state)

        return AMLObservation(
            tool_result=result,
            available_tools=AVAILABLE_TOOLS,
            message=(
                f"{message}{eviction_notice} "
                f"[RAM: {ram_count}/2 | Disk: {disk_count} | "
                f"Async: {async_count} | Kernel: {kernel_count}]"
            ),
            done=False,
            reward=step_reward,
            metadata={
                "step": self._state.step_count,
                "agui_state": self._sm.build_agui_state(),
                "page_fault": is_page_fault,
                "eviction_occurred": bool(evicted),
            },
        )

    # ================================================================== #
    # DOMAIN TOOL HANDLERS (legacy, preserved)                             #
    # ================================================================== #
    # All handlers now return: (result, message, done, flags_dict)

    def _handle_review_alert(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Return the full alert details."""
        self._state.alert_reviewed = True
        result = {"alert": self._current_scenario.initial_alert}
        message = "Alert details retrieved. Review the alert summary and begin your investigation."
        return result, message, False, {}

    def _handle_get_customer_profile(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Return KYC profile for a customer."""
        customer_id = params.get("customer_id", "")
        profiles = self._current_scenario.customer_profiles
        if customer_id in profiles:
            self._state.customer_profiled = True
            result = {"customer_profile": profiles[customer_id]}
            message = f"Customer profile retrieved for {customer_id}."
        else:
            # Fuzzy match by name
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
        return result, message, False, {}

    def _handle_query_transactions(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Return transactions matching the given filters."""
        self._state.transactions_queried = True
        customer_id = params.get("customer_id", "")
        date_from = params.get("date_from") or params.get("date_range_start")
        date_to = params.get("date_to") or params.get("date_range_end")
        min_amount = params.get("min_amount")
        max_amount = params.get("max_amount")

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
        return result, message, False, {}

    def _handle_check_watchlist(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Screen an entity against watchlists."""
        entity_name = params.get("entity_name", params.get("entity", ""))
        if entity_name:
            self._state.watchlist_checked.append(entity_name)

        wl = self._current_scenario.watchlist_results
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
        return result, message, False, {}

    def _handle_trace_network(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Trace entity network connections."""
        entity_id = params.get("entity_id", params.get("entity", ""))
        depth = int(params.get("depth", 1))

        self._state.network_traced = True
        graph = self._current_scenario.network_graph

        result_data = graph.get(entity_id)
        if result_data is None:
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

        if depth >= 2 and "depth_2_connections" in result_data:
            result_data = dict(result_data)
            result_data["connections"] = (
                result_data.get("depth_1_connections", result_data.get("connections", []))
                + result_data.get("depth_2_connections", [])
            )

        result = {"network": result_data, "depth": depth}
        conn_count = len(result_data.get("connections", []))
        message = f"Network trace for '{entity_id}' (depth {depth}): {conn_count} connection(s) found."
        return result, message, False, {}

    def _handle_check_source_of_funds(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Check source of funds documentation."""
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
        return result, message, False, {}

    def _handle_assess_risk(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Return a computed risk assessment based on gathered evidence."""
        self._state.risk_assessed = True
        customer_id = params.get("customer_id", "")

        risk_factors: List[str] = []
        risk_score = 20

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

        # OS mechanic bonus: kernel directives enrich risk scoring
        if self._state.meta_injections > 0:
            risk_score += 5 * self._state.meta_injections
            risk_factors.append(f"Compliance rules consulted ({self._state.meta_injections})")

        risk_score = min(risk_score, 100)

        if risk_score >= 75:
            risk_level, recommendation = "CRITICAL", "SAR filing strongly recommended"
        elif risk_score >= 50:
            risk_level, recommendation = "HIGH", "SAR filing recommended"
        elif risk_score >= 25:
            risk_level, recommendation = "MEDIUM", "Enhanced due diligence required"
        else:
            risk_level, recommendation = "LOW", "Monitor; no immediate action required"

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
                "kernel_rules_injected": self._state.meta_injections,
            },
        }
        message = f"Risk assessment complete. Risk level: {risk_level} (score: {risk_score}/100). {recommendation}."
        return result, message, False, {}

    def _handle_file_sar(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Terminal action — file a SAR and compute the final score."""
        self._state.decision_made = True
        findings = params.get("findings", [])
        typology = params.get("typology", "")
        entities_involved = params.get("entities_involved", params.get("entities", []))

        if isinstance(findings, str):
            findings = [findings]
        if isinstance(entities_involved, str):
            entities_involved = [entities_involved]

        self._state.findings.extend(findings)

        final_score = self._grader.grade(
            ground_truth=self._current_scenario.ground_truth,
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
            "grader_breakdown": self._build_grader_breakdown(
                "file_sar", findings, entities_involved, typology, final_score
            ),
        }
        message = (
            f"SAR filed successfully. Final episode score: {final_score:.4f}. "
            f"Typology: {typology}. Entities flagged: {entities_involved}."
        )
        return result, message, True, {}

    def _handle_close_alert(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Terminal action — close alert as false positive."""
        self._state.decision_made = True
        reason = params.get("reason", "")
        findings = params.get("findings", [])
        entities_involved = params.get("entities_involved", [])
        typology = params.get("typology", "false_positive")

        if isinstance(findings, str):
            findings = [findings]

        self._state.findings.extend(findings)

        final_score = self._grader.grade(
            ground_truth=self._current_scenario.ground_truth,
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
            "grader_breakdown": self._build_grader_breakdown(
                "close_alert", findings, entities_involved, typology, final_score
            ),
        }
        message = (
            f"Alert closed as false positive. Reason: {reason}. "
            f"Final episode score: {final_score:.4f}."
        )
        return result, message, True, {}

    # ================================================================== #
    # OS-MECHANIC TOOL HANDLERS                                            #
    # ================================================================== #

    def _handle_write_to_case_file(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Page data to the persistent case file (Disk).

        This is the agent's only way to prevent data loss from RAM eviction.
        """
        content = params.get("content", params.get("note", ""))
        if not content:
            return (
                {"error": "Missing 'content' parameter"},
                "write_to_case_file requires a 'content' parameter.",
                False,
                {},
            )

        self._sm.write_to_disk(content)
        result = {
            "written": True,
            "disk_entry_count": len(self._sm.disk_contents),
            "content_saved": content,
        }
        message = f"Data paged to case file. Total disk entries: {len(self._sm.disk_contents)}."
        return result, message, False, {"successful_page": True}

    def _handle_request_wire_trace(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Enqueue an async wire trace job (Interrupt mechanic).

        Returns a Job ID and ETA. The result is NOT immediately available.
        """
        entity_id = params.get("entity_id", params.get("customer_id", ""))
        transaction_id = params.get("transaction_id", "")

        if not entity_id and not transaction_id:
            return (
                {"error": "Provide 'entity_id' or 'transaction_id'"},
                "request_wire_trace requires entity_id or transaction_id.",
                False,
                {},
            )

        # Build the deferred result from scenario data
        target = entity_id or transaction_id
        sof = self._current_scenario.source_of_funds
        nw = self._current_scenario.network_graph

        wire_result: Dict[str, Any] = {
            "trace_target": target,
            "correspondent_banks": ["Deutsche Bank AG", "HSBC Holdings"],
            "intermediary_count": random.randint(1, 3),
        }
        # Enrich with source-of-funds data if available
        if transaction_id and transaction_id in sof:
            wire_result["source_info"] = sof[transaction_id]
        # Enrich with network data if available
        if entity_id and entity_id in nw:
            wire_result["network_connections"] = nw[entity_id].get("connections", [])[:3]

        eta = random.randint(2, 4)
        job_id = self._sm.enqueue_async(
            tool="request_wire_trace",
            params=params,
            eta_steps=eta,
            result_payload=wire_result,
        )

        result = {
            "job_id": job_id,
            "status": "pending",
            "eta_steps": eta,
            "message": f"Wire trace request enqueued. Result will be available in {eta} steps. "
                       f"Use retrieve_async_result(job_id='{job_id}') when ETA reaches 0.",
        }
        message = (
            f"Wire trace enqueued as {job_id} (ETA: {eta} steps). "
            f"Continue investigating — do NOT wait idle."
        )
        return result, message, False, {}

    def _handle_retrieve_async_result(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Retrieve the result of a completed async job.

        If the job's ETA hasn't reached 0, triggers an async timeout penalty.
        """
        job_id = params.get("job_id", "")
        if not job_id:
            return (
                {"error": "Missing 'job_id' parameter"},
                "retrieve_async_result requires a 'job_id' parameter.",
                False,
                {},
            )

        result_data, is_timeout = self._sm.retrieve_async(job_id)

        if is_timeout:
            # Premature retrieval
            job = self._sm._async_jobs.get(job_id)
            eta = job.eta_remaining if job else "?"
            result = {
                "error": "Job not ready",
                "job_id": job_id,
                "eta_remaining": eta,
                "penalty": "ASYNC_TIMEOUT (-0.10)",
            }
            message = (
                f"Job {job_id} is still pending (ETA: {eta} steps remaining). "
                f"Async timeout penalty applied. Do NOT retrieve before ETA=0."
            )
            return result, message, False, {"async_timeout": True}

        if result_data is None:
            result = {"error": f"No active job found with id '{job_id}'"}
            message = f"Job '{job_id}' not found or already retrieved."
            return result, message, False, {}

        result = {"job_id": job_id, "status": "completed", "wire_trace": result_data}
        message = f"Async job {job_id} retrieved successfully. Wire trace data available."
        return result, message, False, {}

    def _handle_search_compliance_manual(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Search the enterprise compliance intranet for relevant rules."""
        query = params.get("query", params.get("keyword", ""))
        category = params.get("category", None)
        max_results = int(params.get("max_results", 3))

        if not query:
            return (
                {"error": "Missing 'query' parameter"},
                "search_compliance_manual requires a 'query' parameter.",
                False,
                {},
            )

        hits = search_compliance_manual(query, max_results=max_results, category_filter=category)

        result = {"query": query, "results": hits, "count": len(hits)}
        if hits:
            message = f"Found {len(hits)} compliance rule(s) matching '{query}'. Use update_system_prompt to inject relevant rules."
        else:
            message = f"No compliance rules found matching '{query}'."
        return result, message, False, {}

    def _handle_update_system_prompt(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Inject a compliance rule into the agent's kernel directives."""
        rule = params.get("rule", params.get("directive", params.get("content", "")))
        if not rule:
            return (
                {"error": "Missing 'rule' parameter"},
                "update_system_prompt requires a 'rule' parameter (the text to inject).",
                False,
                {},
            )

        self._sm.inject_directive(rule, self._state.step_count)
        result = {
            "injected": True,
            "directive_count": len(self._sm.kernel_directives),
            "rule_added": rule[:100] + ("..." if len(rule) > 100 else ""),
        }
        message = f"Kernel directive injected. Total directives: {len(self._sm.kernel_directives)}."
        return result, message, False, {"meta_injection": True}

    def _handle_check_market_price(self, params: Dict[str, Any]) -> Tuple[Dict, str, bool, Dict]:
        """Return market price data for commodity comparison (TBML detection)."""
        commodity = params.get("commodity", params.get("item", ""))
        if not commodity:
            return (
                {"error": "Missing 'commodity' parameter"},
                "check_market_price requires a 'commodity' parameter.",
                False,
                {},
            )

        market_data = self._current_scenario.market_data
        if not market_data:
            return (
                {"commodity": commodity, "price": None, "note": "No market data available for this scenario."},
                f"No market data available for commodity '{commodity}'.",
                False,
                {},
            )

        # Fuzzy match
        price_data = market_data.get(commodity)
        if price_data is None:
            for key, val in market_data.items():
                if commodity.lower() in key.lower() or key.lower() in commodity.lower():
                    price_data = val
                    commodity = key
                    break

        if price_data is None:
            result = {"commodity": commodity, "price": None, "available_commodities": list(market_data.keys())}
            message = f"No market price found for '{commodity}'. Available: {list(market_data.keys())}."
        else:
            result = {"commodity": commodity, "market_price": price_data}
            message = f"Market price for '{commodity}': {price_data}."
        return result, message, False, {}

    # ================================================================== #
    # INTERNAL HELPERS                                                     #
    # ================================================================== #

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
            "os_mechanic_stats": {
                "page_faults": self._state.page_fault_count,
                "async_timeouts": self._state.async_timeout_count,
                "successful_pages": self._state.successful_pages,
                "meta_injections": self._state.meta_injections,
                "disk_entries": len(self._sm.disk_contents) if self._sm else 0,
                "kernel_directives": len(self._sm.kernel_directives) if self._sm else 0,
            },
            "final_score": final_score,
        }
