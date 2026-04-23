"""
Memex OS-Agent Benchmark — HTTP client.

Thin Python wrapper around the FastAPI server endpoints. Supports
domain tools (12), OS-mechanic tools (6) for a total of 18 tools.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class AMLEnvironmentClient:
    """HTTP client for the Memex AML Investigation Environment.

    Usage
    -----
    with AMLEnvironmentClient() as client:
        obs = client.reset(task_id="easy")
        obs = client.review_alert()
        obs = client.write_to_case_file("PEP connection confirmed for ENT_B")
        obs = client.file_sar(
            findings=["pep_connection", "shared_address"],
            typology="layering",
            entities_involved=["CUST002", "ENT_B"],
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AML_ENV_URL", "http://localhost:8000")
        ).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core endpoints
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
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
        """Execute a tool action and return the observation."""
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
        """Return the current environment state snapshot."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Domain tool wrappers (12 tools)
    # ------------------------------------------------------------------

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

    def check_market_price(self, commodity: str) -> Dict[str, Any]:
        return self.step("check_market_price", parameters={"commodity": commodity})

    def assess_risk(self, customer_id: str) -> Dict[str, Any]:
        return self.step("assess_risk", parameters={"customer_id": customer_id})

    def file_sar(
        self,
        findings: List[str],
        typology: str,
        entities_involved: List[str],
        ubo_identified: Optional[str] = None,
        evidence_chain: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "findings": findings,
            "typology": typology,
            "entities_involved": entities_involved,
        }
        if ubo_identified is not None:
            params["ubo_identified"] = ubo_identified
        if evidence_chain is not None:
            params["evidence_chain"] = evidence_chain
        return self.step("file_sar", parameters=params)

    def close_alert(
        self,
        reason: str,
        findings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.step("close_alert", parameters={
            "reason": reason,
            "findings": findings or [],
        })

    # ------------------------------------------------------------------
    # Phase 3: FinCEN 4-Pillar tool wrappers (3 tools)
    # ------------------------------------------------------------------

    def check_device_overlap(self, entity_id: str) -> Dict[str, Any]:
        """Check for shared device fingerprints / mule ring indicators."""
        return self.step("check_device_overlap", parameters={"entity_id": entity_id})

    def verify_customs_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Verify a customs invoice for phantom shipment / over-invoicing."""
        return self.step("verify_customs_invoice", parameters={"invoice_id": invoice_id})

    def query_beneficial_ownership(self, entity_id: str) -> Dict[str, Any]:
        """Trace beneficial ownership through shell layers to find UBOs."""
        return self.step("query_beneficial_ownership", parameters={"entity_id": entity_id})

    # ------------------------------------------------------------------
    # OS-Mechanic tool wrappers (6 tools)
    # ------------------------------------------------------------------

    def write_to_case_file(self, content: str) -> Dict[str, Any]:
        """Page important findings to persistent disk storage."""
        return self.step("write_to_case_file", parameters={"content": content})

    def request_wire_trace(
        self,
        entity_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enqueue an async background trace. Returns a job_id + ETA."""
        params: Dict[str, Any] = {}
        if entity_id:
            params["entity_id"] = entity_id
        if transaction_id:
            params["transaction_id"] = transaction_id
        return self.step("request_wire_trace", parameters=params)

    def retrieve_async_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve a completed async job result."""
        return self.step("retrieve_async_result", parameters={"job_id": job_id})

    def search_compliance_manual(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 3,
    ) -> Dict[str, Any]:
        """Search the compliance manual for AML rules."""
        params: Dict[str, Any] = {"query": query, "max_results": max_results}
        if category:
            params["category"] = category
        return self.step("search_compliance_manual", parameters=params)

    def update_system_prompt(self, rule: str) -> Dict[str, Any]:
        """Inject a compliance rule into the active kernel directives."""
        return self.step("update_system_prompt", parameters={"rule": rule})

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AMLEnvironmentClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()
