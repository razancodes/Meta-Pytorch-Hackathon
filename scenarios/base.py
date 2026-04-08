"""Base scenario class — all AML scenarios inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseScenario(ABC):
    """
    Abstract base class for AML investigation scenarios.

    Each concrete scenario exposes:
    - initial_alert          : dict  — alert summary presented to the agent at reset
    - customer_profiles      : dict  — customer_id -> profile data (KYC)
    - transactions           : list  — list of transaction dicts
    - watchlist_results      : dict  — entity_name -> watchlist hit/miss
    - network_graph          : dict  — entity_id -> connected entities
    - source_of_funds        : dict  — transaction_id -> source verification result
    - ground_truth           : dict  — correct_decision, typology, key_entities, key_findings
    - market_data            : dict  — commodity -> {unit_price, currency} (optional)
    """

    # ------------------------------------------------------------------ #
    # Abstract properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def initial_alert(self) -> Dict[str, Any]:
        """Alert summary shown to the agent at the start of the episode."""

    @property
    @abstractmethod
    def customer_profiles(self) -> Dict[str, Any]:
        """Mapping of customer_id -> KYC profile dict."""

    @property
    @abstractmethod
    def transactions(self) -> List[Dict[str, Any]]:
        """List of transaction records."""

    @property
    @abstractmethod
    def watchlist_results(self) -> Dict[str, Any]:
        """Mapping of entity_name -> watchlist screening result."""

    @property
    @abstractmethod
    def network_graph(self) -> Dict[str, Any]:
        """Mapping of entity_id -> connected-entity list."""

    @property
    @abstractmethod
    def source_of_funds(self) -> Dict[str, Any]:
        """Mapping of transaction_id -> source-of-funds verification result."""

    @property
    @abstractmethod
    def ground_truth(self) -> Dict[str, Any]:
        """
        Ground truth dict with at least:
        - correct_decision : "file_sar" | "close_alert"
        - typology         : str
        - key_entities     : List[str]
        - key_findings     : List[str]
        """

    # ------------------------------------------------------------------ #
    # Optional helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def market_data(self) -> Dict[str, Any]:
        """Optional commodity market price data. Override in subclass."""
        return {}

    def get_transactions_for(
        self,
        customer_id: str,
        date_from: str | None = None,
        date_to: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Return transactions matching the given filters."""
        results = []
        for tx in self.transactions:
            # Customer filter
            if (
                tx.get("customer_id") != customer_id
                and tx.get("sender_id") != customer_id
                and tx.get("receiver_id") != customer_id
            ):
                continue
            # Date filters
            if date_from and tx.get("date", "") < date_from:
                continue
            if date_to and tx.get("date", "") > date_to:
                continue
            # Amount filters
            amount = tx.get("amount", 0.0)
            if min_amount is not None and amount < min_amount:
                continue
            if max_amount is not None and amount > max_amount:
                continue
            results.append(tx)
        return results
