"""
Task 2: Layering Through Shell Companies (Medium)

'GlobalTrade LLC' (CUST002) receives $500K from 'Apex Holdings' and immediately
fans it out to three entities. Two share a registered address; one has a PEP director.
Entity E is a legitimate supplier and should NOT be flagged.

Ground truth: file_sar, typology='layering',
              entities=['CUST002','ENT_A','ENT_B','ENT_C','ENT_D']
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseScenario


class MediumScenario(BaseScenario):

    # ------------------------------------------------------------------ #
    # Alert                                                                #
    # ------------------------------------------------------------------ #

    @property
    def initial_alert(self) -> Dict[str, Any]:
        return {
            "alert_id": "ALERT-2024-0187",
            "alert_date": "2024-02-20",
            "alert_type": "Rapid Fan-Out Wire Transfers",
            "risk_score": 85,
            "priority": "HIGH",
            "customer_id": "CUST002",
            "account_id": "ACC-9902",
            "summary": (
                "GlobalTrade LLC (CUST002) received an inbound wire of $500,000 from Apex Holdings "
                "on 2024-02-15 and dispersed $200,000 to Bright Solutions Ltd, $150,000 to "
                "Crescent Ventures, and $150,000 to Delta Resources within 24 hours. "
                "The account was opened 6 months ago. No trade documentation on file."
            ),
            "flagged_rule": "RULE-LAYER-002: Rapid fan-out of large inbound wire",
            "total_inflow": 505000.00,
            "total_outflow": 500000.00,
            "net_remaining": 5000.00,
        }

    # ------------------------------------------------------------------ #
    # Customer profiles                                                    #
    # ------------------------------------------------------------------ #

    @property
    def customer_profiles(self) -> Dict[str, Any]:
        return {
            "CUST002": {
                "customer_id": "CUST002",
                "name": "GlobalTrade LLC",
                "type": "business",
                "account_type": "Business Checking",
                "account_age_months": 6,
                "kyc_risk_tier": "High",
                "business_type": "Import/Export",
                "jurisdiction": "Delaware, USA",
                "director": "Samuel Park",
                "annual_revenue_declared": 2000000,
                "pep_status": False,
                "adverse_media": False,
                "beneficial_owners": [{"name": "Samuel Park", "ownership_pct": 100}],
                "notes": (
                    "Newly incorporated company. Limited trading history. "
                    "No audited financials on file. "
                    "Declared import/export business but no trade contracts presented."
                ),
            },
            "ENT_A": {
                "customer_id": "ENT_A",
                "name": "Apex Holdings",
                "type": "business",
                "jurisdiction": "British Virgin Islands",
                "beneficial_owners": "Unknown",
                "notes": "Offshore holding company. Beneficial ownership not disclosed.",
            },
            "ENT_B": {
                "customer_id": "ENT_B",
                "name": "Bright Solutions Ltd",
                "type": "business",
                "jurisdiction": "Singapore",
                "registered_address": "45 Marina Bay, Singapore 018982",
                "beneficial_owners": "Unknown",
                "notes": "Shell entity; no public business activity.",
            },
            "ENT_C": {
                "customer_id": "ENT_C",
                "name": "Crescent Ventures",
                "type": "business",
                "jurisdiction": "Singapore",
                "registered_address": "45 Marina Bay, Singapore 018982",
                "beneficial_owners": "Unknown",
                "notes": "Shell entity; shares registered address with Bright Solutions Ltd.",
            },
            "ENT_D": {
                "customer_id": "ENT_D",
                "name": "Delta Resources",
                "type": "business",
                "jurisdiction": "Cyprus",
                "registered_address": "12 Nicosia Business Park, Nicosia",
                "director": "Viktor Korev",
                "notes": "Director Viktor Korev is a PEP (former deputy finance minister).",
            },
            "ENT_E": {
                "customer_id": "ENT_E",
                "name": "EverGreen Supplies",
                "type": "business",
                "jurisdiction": "USA",
                "registered_address": "800 Industrial Blvd, Houston, TX",
                "notes": (
                    "Long-established supplier. Regular monthly payments. "
                    "Legitimate commercial relationship with GlobalTrade LLC."
                ),
            },
        }

    # ------------------------------------------------------------------ #
    # Transactions                                                         #
    # ------------------------------------------------------------------ #

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        return [
            # Inbound from Apex Holdings
            {
                "transaction_id": "TXN-002-A1",
                "customer_id": "CUST002",
                "sender_id": "ENT_A",
                "receiver_id": "CUST002",
                "date": "2024-02-15",
                "type": "wire_inbound",
                "amount": 500000.00,
                "currency": "USD",
                "description": "Consultancy fees",
                "reference": "INV-AH-0032",
                "same_day_outflow": True,
            },
            # Fan-out to Bright Solutions
            {
                "transaction_id": "TXN-002-B1",
                "customer_id": "CUST002",
                "sender_id": "CUST002",
                "receiver_id": "ENT_B",
                "date": "2024-02-15",
                "type": "wire_outbound",
                "amount": 200000.00,
                "currency": "USD",
                "description": "Service retainer",
                "reference": "REF-BS-1001",
            },
            # Fan-out to Crescent Ventures (same day)
            {
                "transaction_id": "TXN-002-C1",
                "customer_id": "CUST002",
                "sender_id": "CUST002",
                "receiver_id": "ENT_C",
                "date": "2024-02-15",
                "type": "wire_outbound",
                "amount": 150000.00,
                "currency": "USD",
                "description": "Project advance",
                "reference": "REF-CV-0045",
            },
            # Fan-out to Delta Resources (next day)
            {
                "transaction_id": "TXN-002-D1",
                "customer_id": "CUST002",
                "sender_id": "CUST002",
                "receiver_id": "ENT_D",
                "date": "2024-02-16",
                "type": "wire_outbound",
                "amount": 150000.00,
                "currency": "USD",
                "description": "Logistics fees",
                "reference": "REF-DR-0078",
            },
            # Legitimate monthly payment from EverGreen Supplies
            {
                "transaction_id": "TXN-002-E1",
                "customer_id": "CUST002",
                "sender_id": "ENT_E",
                "receiver_id": "CUST002",
                "date": "2024-02-05",
                "type": "wire_inbound",
                "amount": 5000.00,
                "currency": "USD",
                "description": "Monthly supply payment — January",
                "reference": "INV-EGS-2024-01",
            },
            {
                "transaction_id": "TXN-002-E0",
                "customer_id": "CUST002",
                "sender_id": "ENT_E",
                "receiver_id": "CUST002",
                "date": "2024-01-05",
                "type": "wire_inbound",
                "amount": 5000.00,
                "currency": "USD",
                "description": "Monthly supply payment — December",
                "reference": "INV-EGS-2023-12",
            },
        ]

    # ------------------------------------------------------------------ #
    # Watchlist                                                            #
    # ------------------------------------------------------------------ #

    @property
    def watchlist_results(self) -> Dict[str, Any]:
        return {
            "GlobalTrade LLC": {
                "entity": "GlobalTrade LLC",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
            },
            "Apex Holdings": {
                "entity": "Apex Holdings",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No direct match; beneficial ownership unknown — enhanced due diligence recommended",
            },
            "Bright Solutions Ltd": {
                "entity": "Bright Solutions Ltd",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found",
            },
            "Crescent Ventures": {
                "entity": "Crescent Ventures",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found",
            },
            "Delta Resources": {
                "entity": "Delta Resources",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No direct sanction match",
            },
            "Viktor Korev": {
                "entity": "Viktor Korev",
                "hit": True,
                "lists_checked": ["PEP"],
                "result": (
                    "MATCH — PEP list: Viktor Korev, former Deputy Minister of Finance, "
                    "Republic of Coravia (2010–2018). Current director of Delta Resources."
                ),
                "risk_level": "HIGH",
            },
            "Samuel Park": {
                "entity": "Samuel Park",
                "hit": False,
                "lists_checked": ["OFAC SDN", "PEP"],
                "result": "No matches found",
            },
            "EverGreen Supplies": {
                "entity": "EverGreen Supplies",
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found",
            },
        }

    # ------------------------------------------------------------------ #
    # Network graph                                                        #
    # ------------------------------------------------------------------ #

    @property
    def network_graph(self) -> Dict[str, Any]:
        return {
            "CUST002": {
                "entity_id": "CUST002",
                "entity_name": "GlobalTrade LLC",
                "connections": [
                    {
                        "entity_id": "ENT_A",
                        "entity_name": "Apex Holdings",
                        "relationship": "inbound_wire_sender",
                        "amount": 500000,
                    },
                    {
                        "entity_id": "ENT_B",
                        "entity_name": "Bright Solutions Ltd",
                        "relationship": "outbound_wire_receiver",
                        "amount": 200000,
                        "registered_address": "45 Marina Bay, Singapore 018982",
                    },
                    {
                        "entity_id": "ENT_C",
                        "entity_name": "Crescent Ventures",
                        "relationship": "outbound_wire_receiver",
                        "amount": 150000,
                        "registered_address": "45 Marina Bay, Singapore 018982",
                        "note": "SHARED ADDRESS with Bright Solutions Ltd",
                    },
                    {
                        "entity_id": "ENT_D",
                        "entity_name": "Delta Resources",
                        "relationship": "outbound_wire_receiver",
                        "amount": 150000,
                        "director": "Viktor Korev (PEP)",
                    },
                    {
                        "entity_id": "ENT_E",
                        "entity_name": "EverGreen Supplies",
                        "relationship": "legitimate_supplier_inbound",
                        "amount": 5000,
                        "pattern": "monthly_recurring",
                    },
                ],
            },
            "ENT_B": {
                "entity_id": "ENT_B",
                "entity_name": "Bright Solutions Ltd",
                "connections": [
                    {
                        "entity_id": "ENT_C",
                        "entity_name": "Crescent Ventures",
                        "relationship": "shared_registered_address",
                        "address": "45 Marina Bay, Singapore 018982",
                    }
                ],
            },
            "ENT_C": {
                "entity_id": "ENT_C",
                "entity_name": "Crescent Ventures",
                "connections": [
                    {
                        "entity_id": "ENT_B",
                        "entity_name": "Bright Solutions Ltd",
                        "relationship": "shared_registered_address",
                        "address": "45 Marina Bay, Singapore 018982",
                    }
                ],
            },
            "ENT_D": {
                "entity_id": "ENT_D",
                "entity_name": "Delta Resources",
                "connections": [
                    {
                        "entity_id": "Viktor Korev",
                        "entity_name": "Viktor Korev",
                        "relationship": "director",
                        "pep": True,
                    }
                ],
            },
            "ENT_E": {
                "entity_id": "ENT_E",
                "entity_name": "EverGreen Supplies",
                "connections": [],
                "note": "Legitimate supplier; no suspicious connections.",
            },
        }

    # ------------------------------------------------------------------ #
    # Source of funds                                                      #
    # ------------------------------------------------------------------ #

    @property
    def source_of_funds(self) -> Dict[str, Any]:
        return {
            "TXN-002-A1": {
                "transaction_id": "TXN-002-A1",
                "source": "Apex Holdings (BVI)",
                "documentation": None,
                "verified": False,
                "notes": (
                    "Apex Holdings is an offshore BVI entity. No underlying trade contract or "
                    "invoice provided. 'Consultancy fees' description is vague and unsubstantiated."
                ),
            },
            "TXN-002-B1": {
                "transaction_id": "TXN-002-B1",
                "source": "Redistributed from inbound wire",
                "documentation": None,
                "verified": False,
                "notes": "No service contract on file for Bright Solutions Ltd.",
            },
            "TXN-002-C1": {
                "transaction_id": "TXN-002-C1",
                "source": "Redistributed from inbound wire",
                "documentation": None,
                "verified": False,
                "notes": "No project documentation on file for Crescent Ventures.",
            },
            "TXN-002-D1": {
                "transaction_id": "TXN-002-D1",
                "source": "Redistributed from inbound wire",
                "documentation": None,
                "verified": False,
                "notes": "No logistics contract on file for Delta Resources.",
            },
            "TXN-002-E1": {
                "transaction_id": "TXN-002-E1",
                "source": "EverGreen Supplies — trade payment",
                "documentation": "INV-EGS-2024-01 (verified)",
                "verified": True,
                "notes": "Legitimate monthly supply invoice. No concerns.",
            },
        }

    # ------------------------------------------------------------------ #
    # Ground truth                                                         #
    # ------------------------------------------------------------------ #

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return {
            "correct_decision": "file_sar",
            "typology": "layering",
            "key_entities": ["CUST002", "ENT_A", "ENT_B", "ENT_C", "ENT_D"],
            "excluded_entities": ["ENT_E"],
            "key_findings": [
                "rapid_fan_out",
                "pep_connection",
                "shared_registered_address",
                "offshore_source",
                "newly_incorporated",
                "no_trade_documentation",
            ],
        }
