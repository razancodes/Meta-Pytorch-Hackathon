"""
Task 3: Trade-Based Money Laundering (Hard)

NovaTech Industries (CUST003) buys 'machine parts' from OceanPrime Exports (ENT_F)
at $50K/unit — 317% above market ($12K). Over 6 months, 12 invoices totalling $600K
vs. expected $144K at market rates (overpayment $456K). ENT_F's beneficial owner is
related to NovaTech's director. One transaction was reversed and re-sent to avoid
a round number.

Red herring: TechDirect Corp (ENT_H) is a legitimate supplier at market rates.

Ground truth: file_sar, typology='trade_based_ml',
              entities=['CUST003','ENT_F','Marcus Webb']
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseScenario


class HardScenario(BaseScenario):

    # ------------------------------------------------------------------ #
    # Alert                                                                #
    # ------------------------------------------------------------------ #

    @property
    def initial_alert(self) -> Dict[str, Any]:
        return {
            "alert_id": "ALERT-2024-0391",
            "alert_date": "2024-04-01",
            "alert_type": "Potential Trade-Based Money Laundering",
            "risk_score": 91,
            "priority": "CRITICAL",
            "customer_id": "CUST003",
            "account_id": "ACC-5503",
            "summary": (
                "NovaTech Industries (CUST003) has conducted 12 payments over 6 months "
                "to OceanPrime Exports (ENT_F), an offshore entity registered in a "
                "FATF-monitored jurisdiction, totalling $600,000 for 'machine parts'. "
                "Market intelligence suggests comparable parts trade at approximately $12,000/unit. "
                "Additionally, CUST003 received an unexplained $200,000 wire from "
                "Cayman Islands Investment Fund (ENT_G). "
                "One transaction was reversed and re-initiated with a slightly different amount."
            ),
            "flagged_rule": "RULE-TBML-003: Systematic over-invoicing pattern",
            "total_outflow_to_ENT_F": 600000.00,
            "unexplained_inflow": 200000.00,
            "units_purchased": 12,
        }

    # ------------------------------------------------------------------ #
    # Customer profiles                                                    #
    # ------------------------------------------------------------------ #

    @property
    def customer_profiles(self) -> Dict[str, Any]:
        return {
            "CUST003": {
                "customer_id": "CUST003",
                "name": "NovaTech Industries",
                "type": "business",
                "account_type": "Business Checking",
                "account_age_months": 24,
                "kyc_risk_tier": "Medium-High",
                "business_type": "Electronics Import / Distribution",
                "jurisdiction": "California, USA",
                "director": "Alan Chen",
                "annual_revenue_declared": 5000000,
                "pep_status": False,
                "adverse_media": False,
                "beneficial_owners": [{"name": "Alan Chen", "ownership_pct": 75}, {"name": "Sarah Lee", "ownership_pct": 25}],
                "suppliers_declared": ["OceanPrime Exports", "TechDirect Corp"],
                "notes": (
                    "Two-year-old company. Moderate transaction volume historically. "
                    "Sudden spike in high-value outflows to offshore counterparty in last 6 months."
                ),
            },
            "ENT_F": {
                "customer_id": "ENT_F",
                "name": "OceanPrime Exports",
                "type": "business",
                "jurisdiction": "Myanmar",
                "fatf_flagged": True,
                "fatf_status": "FATF Monitored Jurisdiction (Increased Monitoring)",
                "registered_address": "Unit 7, Yangon Trade Centre, Myanmar",
                "director": "Li Wei",
                "beneficial_owner": "Marcus Webb",
                "beneficial_owner_notes": (
                    "Marcus Webb is the beneficial owner of OceanPrime Exports. "
                    "Webb is the brother-in-law of Alan Chen (director of CUST003 / NovaTech Industries)."
                ),
                "notes": "Offshore exporter in FATF-monitored jurisdiction. Limited public business presence.",
            },
            "ENT_G": {
                "customer_id": "ENT_G",
                "name": "Cayman Islands Investment Fund",
                "type": "fund",
                "jurisdiction": "Cayman Islands",
                "notes": "Offshore investment fund. No disclosed relationship with NovaTech. Source of $200K wire unexplained.",
            },
            "ENT_H": {
                "customer_id": "ENT_H",
                "name": "TechDirect Corp",
                "type": "business",
                "jurisdiction": "USA",
                "registered_address": "220 Silicon Ave, San Jose, CA",
                "notes": (
                    "Well-established domestic electronics distributor. "
                    "NovaTech has purchased from TechDirect for 3+ years at market prices. "
                    "Legitimate long-term supplier relationship."
                ),
            },
        }

    # ------------------------------------------------------------------ #
    # Transactions                                                         #
    # ------------------------------------------------------------------ #

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        # 12 over-invoiced payments to OceanPrime (ENT_F)
        # One reversed and re-sent (March)
        records = []
        dates_and_amounts = [
            ("2023-10-05", 50000.00, "TXN-003-F01"),
            ("2023-11-03", 50000.00, "TXN-003-F02"),
            ("2023-12-04", 50000.00, "TXN-003-F03"),
            ("2024-01-06", 50000.00, "TXN-003-F04"),
            ("2024-01-28", 50000.00, "TXN-003-F05"),
            ("2024-02-10", 50000.00, "TXN-003-F06"),
            ("2024-02-25", 50000.00, "TXN-003-F07"),
            ("2024-03-08", 50000.00, "TXN-003-F08"),
            ("2024-03-20", 50000.00, "TXN-003-F09"),
            ("2024-04-01", 50000.00, "TXN-003-F10"),
        ]
        for dt, amt, tid in dates_and_amounts:
            records.append({
                "transaction_id": tid,
                "customer_id": "CUST003",
                "sender_id": "CUST003",
                "receiver_id": "ENT_F",
                "date": dt,
                "type": "wire_outbound",
                "amount": amt,
                "currency": "USD",
                "description": "Purchase of machine parts — 1 unit",
                "invoice_ref": f"INV-OP-{tid[-2:]}",
                "goods": "machine_parts",
                "units": 1,
                "unit_price": amt,
            })

        # Reversed transaction (2024-03-15) and re-sent (2024-03-16)
        records.append({
            "transaction_id": "TXN-003-F11-REV",
            "customer_id": "CUST003",
            "sender_id": "CUST003",
            "receiver_id": "ENT_F",
            "date": "2024-03-15",
            "type": "wire_outbound_reversed",
            "amount": 50000.00,
            "currency": "USD",
            "description": "Purchase of machine parts — 1 unit [REVERSED]",
            "reversal": True,
            "reversal_reason": "Beneficiary bank correction",
            "invoice_ref": "INV-OP-11",
        })
        records.append({
            "transaction_id": "TXN-003-F11",
            "customer_id": "CUST003",
            "sender_id": "CUST003",
            "receiver_id": "ENT_F",
            "date": "2024-03-16",
            "type": "wire_outbound",
            "amount": 49750.00,
            "currency": "USD",
            "description": "Purchase of machine parts — 1 unit [re-issued]",
            "reversal_of": "TXN-003-F11-REV",
            "invoice_ref": "INV-OP-11-B",
            "goods": "machine_parts",
            "units": 1,
            "unit_price": 49750.00,
            "note": "Amount adjusted from $50,000 to $49,750 — reason not documented",
        })

        # Unexplained inbound from Cayman Islands Investment Fund
        records.append({
            "transaction_id": "TXN-003-G1",
            "customer_id": "CUST003",
            "sender_id": "ENT_G",
            "receiver_id": "CUST003",
            "date": "2024-02-20",
            "type": "wire_inbound",
            "amount": 200000.00,
            "currency": "USD",
            "description": "Investment disbursement",
            "reference": "CIIF-2024-0022",
            "note": "No investment agreement on file. No prior relationship disclosed.",
        })

        # Legitimate payments to TechDirect Corp (ENT_H) — red herring
        legit_dates = [
            ("2023-11-15", 36000.00, "TXN-003-H01"),
            ("2024-01-20", 24000.00, "TXN-003-H02"),
            ("2024-03-10", 12000.00, "TXN-003-H03"),
        ]
        for dt, amt, tid in legit_dates:
            records.append({
                "transaction_id": tid,
                "customer_id": "CUST003",
                "sender_id": "CUST003",
                "receiver_id": "ENT_H",
                "date": dt,
                "type": "wire_outbound",
                "amount": amt,
                "currency": "USD",
                "description": "Electronics components purchase",
                "invoice_ref": f"INV-TD-{tid[-2:]}",
                "goods": "electronics_components",
                "unit_price_market_aligned": True,
                "note": "Price consistent with market rates. Long-term supplier.",
            })

        return records

    # ------------------------------------------------------------------ #
    # Watchlist                                                            #
    # ------------------------------------------------------------------ #

    @property
    def watchlist_results(self) -> Dict[str, Any]:
        return {
            "NovaTech Industries": {
                "entity": "NovaTech Industries",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
            },
            "OceanPrime Exports": {
                "entity": "OceanPrime Exports",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No direct sanction match; entity registered in FATF-monitored jurisdiction (Myanmar)",
                "risk_flag": "FATF monitored jurisdiction",
            },
            "Marcus Webb": {
                "entity": "Marcus Webb",
                "hit": False,
                "lists_checked": ["OFAC SDN", "PEP", "Interpol"],
                "result": "No sanction or PEP match",
                "note": "Beneficial owner of OceanPrime Exports.",
            },
            "Alan Chen": {
                "entity": "Alan Chen",
                "hit": False,
                "lists_checked": ["OFAC SDN", "PEP"],
                "result": "No matches found",
            },
            "Cayman Islands Investment Fund": {
                "entity": "Cayman Islands Investment Fund",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated"],
                "result": "No direct match; Cayman Islands is a high-risk jurisdiction for fund transparency",
                "risk_flag": "High-risk jurisdiction",
            },
            "TechDirect Corp": {
                "entity": "TechDirect Corp",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found. Well-known US distributor.",
            },
        }

    # ------------------------------------------------------------------ #
    # Network graph                                                        #
    # ------------------------------------------------------------------ #

    @property
    def network_graph(self) -> Dict[str, Any]:
        return {
            "CUST003": {
                "entity_id": "CUST003",
                "entity_name": "NovaTech Industries",
                "connections": [
                    {
                        "entity_id": "ENT_F",
                        "entity_name": "OceanPrime Exports",
                        "relationship": "primary_supplier_outbound",
                        "total_amount": 600000,
                        "transaction_count": 12,
                        "jurisdiction": "Myanmar (FATF monitored)",
                    },
                    {
                        "entity_id": "ENT_G",
                        "entity_name": "Cayman Islands Investment Fund",
                        "relationship": "unexplained_inbound",
                        "amount": 200000,
                    },
                    {
                        "entity_id": "ENT_H",
                        "entity_name": "TechDirect Corp",
                        "relationship": "legitimate_supplier_outbound",
                        "total_amount": 72000,
                        "note": "Market-rate transactions; long-term relationship",
                    },
                    {
                        "entity_id": "Alan Chen",
                        "entity_name": "Alan Chen",
                        "relationship": "director",
                    },
                ],
            },
            "ENT_F": {
                "entity_id": "ENT_F",
                "entity_name": "OceanPrime Exports",
                "depth_1_connections": [
                    {
                        "entity_id": "Li Wei",
                        "entity_name": "Li Wei",
                        "relationship": "registered_director",
                    },
                    {
                        "entity_id": "Marcus Webb",
                        "entity_name": "Marcus Webb",
                        "relationship": "beneficial_owner",
                    },
                ],
                "depth_2_connections": [
                    {
                        "entity_id": "Alan Chen",
                        "entity_name": "Alan Chen",
                        "via": "Marcus Webb",
                        "relationship": "brother_in_law",
                        "note": (
                            "Marcus Webb (beneficial owner of OceanPrime Exports) is the "
                            "brother-in-law of Alan Chen (director of NovaTech Industries / CUST003). "
                            "This creates a potential undisclosed conflict of interest."
                        ),
                    }
                ],
            },
            "ENT_H": {
                "entity_id": "ENT_H",
                "entity_name": "TechDirect Corp",
                "connections": [],
                "note": "No suspicious connections. Clean counterparty.",
            },
        }

    # ------------------------------------------------------------------ #
    # Source of funds                                                      #
    # ------------------------------------------------------------------ #

    @property
    def source_of_funds(self) -> Dict[str, Any]:
        result = {}
        for i in range(1, 11):
            tid = f"TXN-003-F{i:02d}"
            result[tid] = {
                "transaction_id": tid,
                "source": "Business operating account",
                "documentation": f"INV-OP-{i:02d} (copy on file)",
                "verified": False,
                "notes": (
                    "Invoice presented for machine parts. Unit price $50,000. "
                    "Market data shows comparable parts at $12,000/unit. "
                    "Invoice does not include part number, model spec, or shipping manifest."
                ),
            }
        result["TXN-003-F11"] = {
            "transaction_id": "TXN-003-F11",
            "source": "Business operating account",
            "documentation": "INV-OP-11-B",
            "verified": False,
            "notes": (
                "Re-issued payment after reversal. Amount changed from $50,000 to $49,750. "
                "No explanation for amount change on file."
            ),
        }
        result["TXN-003-F11-REV"] = {
            "transaction_id": "TXN-003-F11-REV",
            "source": "Business operating account",
            "documentation": "INV-OP-11",
            "verified": False,
            "notes": "Transaction reversed on same day. Stated reason: beneficiary bank correction.",
        }
        result["TXN-003-G1"] = {
            "transaction_id": "TXN-003-G1",
            "source": "Cayman Islands Investment Fund",
            "documentation": None,
            "verified": False,
            "notes": (
                "No investment agreement or subscription documents on file. "
                "No prior relationship between NovaTech and this fund disclosed at onboarding. "
                "Source of funds for the fund itself is unknown."
            ),
        }
        for tid in ["TXN-003-H01", "TXN-003-H02", "TXN-003-H03"]:
            result[tid] = {
                "transaction_id": tid,
                "source": "Business operating account",
                "documentation": f"INV-TD-{tid[-2:]} (verified)",
                "verified": True,
                "notes": "Legitimate purchase from TechDirect Corp at market rates. No concerns.",
            }
        return result

    # ------------------------------------------------------------------ #
    # Market data                                                          #
    # ------------------------------------------------------------------ #

    @property
    def market_data(self) -> Dict[str, Any]:
        return {
            "machine_parts": {
                "description": "General industrial machine parts (comparable grade)",
                "market_unit_price_usd": 12000,
                "price_range_usd": [8000, 15000],
                "source": "Industry trade database Q1-2024",
                "note": (
                    "Invoiced price of $50,000/unit represents a 317% premium over market. "
                    "12 units totalling $600,000 vs expected $144,000 — overpayment of $456,000."
                ),
            },
            "electronics_components": {
                "description": "Standard electronics components",
                "market_unit_price_usd": "variable — consistent with invoiced prices from TechDirect",
                "note": "TechDirect Corp pricing is market-aligned.",
            },
        }

    # ------------------------------------------------------------------ #
    # Ground truth                                                         #
    # ------------------------------------------------------------------ #

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return {
            "correct_decision": "file_sar",
            "typology": "trade_based_ml",
            "key_entities": ["CUST003", "ENT_F", "Marcus Webb"],
            "excluded_entities": ["ENT_H"],
            "key_findings": [
                "over_invoicing",
                "beneficial_owner_connection",
                "fatf_jurisdiction",
                "reversed_transaction",
                "unexplained_funds",
            ],
        }
