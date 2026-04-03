"""
AML Investigation Environment — HTTP client.

Provides a thin Python wrapper around the FastAPI server endpoints,
mimicking an MCP-tool-client interface so agents can call tools through
a clean Python API rather than raw HTTP.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class AMLEnvironmentClient:
    """
    HTTP client for the AML Investigation Environment server.

    Usage
    -----
    client = AMLEnvironmentClient(base_url="http://localhost:8000")
    obs = client.reset(task_id="easy")
    obs = client.step(tool="review_alert", parameters={"alert_id": "ALERT-2024-0042"})
    state = client.get_state()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AML_ENV_URL", "http://localhost:8000")).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Core endpoints                                                       #
    # ------------------------------------------------------------------ #

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
        """
        Reset the environment.

        Returns the initial observation dict.
        """
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
        """
        Execute a tool action.

        Returns the observation dict.
        """
        payload: Dict[str, Any] = {
            "tool": tool,
            "parameters": parameters or {},
            "metadata": metadata or {},
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

    # ------------------------------------------------------------------ #
    # Convenience tool wrappers                                            #
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
        return self.step(
            "check_watchlist",
            parameters={"entity_name": entity_name, "list_type": list_type},
        )

    def trace_network(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        return self.step(
            "trace_network",
            parameters={"entity_id": entity_id, "depth": depth},
        )

    def check_source_of_funds(self, transaction_id: str) -> Dict[str, Any]:
        return self.step(
            "check_source_of_funds",
            parameters={"transaction_id": transaction_id},
        )

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

    def close_alert(
        self,
        reason: str,
        findings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.step(
            "close_alert",
            parameters={
                "reason": reason,
                "findings": findings or [],
            },
        )

    # ------------------------------------------------------------------ #
    # Context manager support                                              #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "AMLEnvironmentClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()
