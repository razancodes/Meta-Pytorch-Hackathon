"""
Task 1: Structuring Detection (Easy)

Customer 'John Doe' (CUST001) makes 5 cash deposits just below the $10,000
CTR reporting threshold over 5 consecutive days — a textbook structuring pattern.

Ground truth: file_sar, typology='structuring'
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseScenario


class EasyScenario(BaseScenario):

    # ------------------------------------------------------------------ #
    # Alert                                                                #
    # ------------------------------------------------------------------ #

    @property
    def initial_alert(self) -> Dict[str, Any]:
        return {
            "alert_id": "ALERT-2024-0042",
            "alert_date": "2024-01-10",
            "alert_type": "Multiple Cash Deposits Below Reporting Threshold",
            "risk_score": 72,
            "priority": "HIGH",
            "customer_id": "CUST001",
            "account_id": "ACC-7701",
            "summary": (
                "Customer CUST001 has made 5 cash deposits totalling $47,900 "
                "over a 5-day period (2024-01-03 to 2024-01-07). "
                "All deposits are below the $10,000 CTR reporting threshold. "
                "No business justification is on file."
            ),
            "flagged_rule": "RULE-STRUCT-001: Multiple sub-threshold cash deposits",
            "branch": "Downtown Retail — Branch 12",
        }

    # ------------------------------------------------------------------ #
    # Customer profiles                                                    #
    # ------------------------------------------------------------------ #

    @property
    def customer_profiles(self) -> Dict[str, Any]:
        return {
            "CUST001": {
                "customer_id": "CUST001",
                "name": "John Doe",
                "type": "individual",
                "account_type": "Personal Checking",
                "account_age_months": 36,
                "kyc_risk_tier": "Medium",
                "occupation": "Retail Store Clerk",
                "annual_income_declared": 38000,
                "address": "142 Elm Street, Springfield, IL 62701",
                "date_of_birth": "1985-06-14",
                "pep_status": False,
                "adverse_media": False,
                "expected_monthly_cash": 1500,
                "business_type": None,
                "notes": "No prior SARs. No cash-intensive occupation. No business account.",
            }
        }

    # ------------------------------------------------------------------ #
    # Transactions                                                         #
    # ------------------------------------------------------------------ #

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        return [
            {
                "transaction_id": "TXN-001-A",
                "customer_id": "CUST001",
                "date": "2024-01-03",
                "type": "cash_deposit",
                "amount": 9500.00,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": "Downtown Retail — Branch 12",
                "description": "Cash deposit",
                "counterparty": None,
            },
            {
                "transaction_id": "TXN-001-B",
                "customer_id": "CUST001",
                "date": "2024-01-04",
                "type": "cash_deposit",
                "amount": 9800.00,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": "Downtown Retail — Branch 12",
                "description": "Cash deposit",
                "counterparty": None,
            },
            {
                "transaction_id": "TXN-001-C",
                "customer_id": "CUST001",
                "date": "2024-01-05",
                "type": "cash_deposit",
                "amount": 9400.00,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": "Downtown Retail — Branch 12",
                "description": "Cash deposit",
                "counterparty": None,
            },
            {
                "transaction_id": "TXN-001-D",
                "customer_id": "CUST001",
                "date": "2024-01-06",
                "type": "cash_deposit",
                "amount": 9700.00,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": "Downtown Retail — Branch 12",
                "description": "Cash deposit",
                "counterparty": None,
            },
            {
                "transaction_id": "TXN-001-E",
                "customer_id": "CUST001",
                "date": "2024-01-07",
                "type": "cash_deposit",
                "amount": 9500.00,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": "Downtown Retail — Branch 12",
                "description": "Cash deposit",
                "counterparty": None,
            },
        ]

    # ------------------------------------------------------------------ #
    # Watchlist                                                            #
    # ------------------------------------------------------------------ #

    @property
    def watchlist_results(self) -> Dict[str, Any]:
        return {
            "John Doe": {
                "entity": "John Doe",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
                "checked_date": "2024-01-10",
            },
            "CUST001": {
                "entity": "CUST001",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
                "checked_date": "2024-01-10",
            },
        }

    # ------------------------------------------------------------------ #
    # Network graph                                                        #
    # ------------------------------------------------------------------ #

    @property
    def network_graph(self) -> Dict[str, Any]:
        return {
            "CUST001": {
                "entity_id": "CUST001",
                "entity_name": "John Doe",
                "connections": [],
                "note": "No significant counterparty relationships identified.",
            }
        }

    # ------------------------------------------------------------------ #
    # Source of funds                                                      #
    # ------------------------------------------------------------------ #

    @property
    def source_of_funds(self) -> Dict[str, Any]:
        result = {}
        for txn_id in ["TXN-001-A", "TXN-001-B", "TXN-001-C", "TXN-001-D", "TXN-001-E"]:
            result[txn_id] = {
                "transaction_id": txn_id,
                "source": "cash",
                "documentation": None,
                "verified": False,
                "notes": (
                    "Cash deposited at branch teller. No supporting documentation provided. "
                    "Customer could not provide explanation for source of funds when asked."
                ),
            }
        return result

    # ------------------------------------------------------------------ #
    # Ground truth                                                         #
    # ------------------------------------------------------------------ #

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return {
            "correct_decision": "file_sar",
            "typology": "structuring",
            "key_entities": ["CUST001"],
            "key_findings": [
                "multiple_sub_threshold_deposits",
                "no_cash_intensive_occupation",
                "same_branch_repeated",
                "no_source_documentation",
                "total_exceeds_ctr_threshold",
            ],
        }
