"""
Memex OS-Agent Benchmark — HTTP client.

Thin Python wrapper around the FastAPI server endpoints with convenience
methods for all 15 tools (9 domain + 6 OS-mechanic).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class AMLEnvironmentClient:
    """HTTP client for the Memex AML Investigation Environment server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AML_ENV_URL", "http://localhost:8000")
        ).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Core endpoints                                                       #
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(
        self,
        tool: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action": {
                "tool": tool,
                "parameters": parameters or {},
                "metadata": metadata or {},
            },
        }
        if timeout_s is not None:
            payload["timeout_s"] = timeout_s
        resp = self._client.post("/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def get_agui(self) -> Dict[str, Any]:
        """Fetch the latest AGUI visualization payload."""
        resp = self._client.get("/agui")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Domain tool wrappers                                                 #
    # ------------------------------------------------------------------ #

    def review_alert(self, alert_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"alert_id": alert_id} if alert_id else {}
        return self.step("review_alert", parameters=params)

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        return self.step("get_customer_profile", parameters={"customer_id": customer_id})

    def query_transactions(
        self,
        customer_id: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"customer_id": customer_id}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if min_amount is not None:
            params["min_amount"] = min_amount
        if max_amount is not None:
            params["max_amount"] = max_amount
        return self.step("query_transactions", parameters=params)

    def check_watchlist(self, entity_name: str, list_type: str = "all") -> Dict[str, Any]:
        return self.step("check_watchlist", parameters={"entity_name": entity_name, "list_type": list_type})

    def trace_network(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        return self.step("trace_network", parameters={"entity_id": entity_id, "depth": depth})

    def check_source_of_funds(self, transaction_id: str) -> Dict[str, Any]:
        return self.step("check_source_of_funds", parameters={"transaction_id": transaction_id})

    def assess_risk(self, customer_id: str) -> Dict[str, Any]:
        return self.step("assess_risk", parameters={"customer_id": customer_id})

    def file_sar(
        self,
        findings: List[str],
        typology: str,
        entities_involved: List[str],
    ) -> Dict[str, Any]:
        return self.step(
            "file_sar",
            parameters={
                "findings": findings,
                "typology": typology,
                "entities_involved": entities_involved,
            },
        )

    def close_alert(self, reason: str, findings: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.step("close_alert", parameters={"reason": reason, "findings": findings or []})

    # ------------------------------------------------------------------ #
    # OS-Mechanic tool wrappers                                            #
    # ------------------------------------------------------------------ #

    def write_to_case_file(self, content: str) -> Dict[str, Any]:
        """Page data to the persistent case file (Virtual Memory → Disk)."""
        return self.step("write_to_case_file", parameters={"content": content})

    def request_wire_trace(
        self,
        entity_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request an async wire trace (Interrupt mechanic)."""
        params: Dict[str, Any] = {}
        if entity_id:
            params["entity_id"] = entity_id
        if transaction_id:
            params["transaction_id"] = transaction_id
        return self.step("request_wire_trace", parameters=params)

    def retrieve_async_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the result of a completed async job."""
        return self.step("retrieve_async_result", parameters={"job_id": job_id})

    def search_compliance_manual(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 3,
    ) -> Dict[str, Any]:
        """Search the compliance intranet for rules (Kernel Update prep)."""
        params: Dict[str, Any] = {"query": query}
        if category:
            params["category"] = category
        params["max_results"] = max_results
        return self.step("search_compliance_manual", parameters=params)

    def update_system_prompt(self, rule: str) -> Dict[str, Any]:
        """Inject a compliance rule into the agent's kernel directives."""
        return self.step("update_system_prompt", parameters={"rule": rule})

    def check_market_price(self, commodity: str) -> Dict[str, Any]:
        """Get market price for a commodity (TBML detection)."""
        return self.step("check_market_price", parameters={"commodity": commodity})

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "AMLEnvironmentClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()
