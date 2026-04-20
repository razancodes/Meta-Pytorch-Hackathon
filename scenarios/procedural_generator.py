"""
Memex OS-Agent Benchmark — Procedural Scenario Generator (Synthetic Data Engine).

Replaces static easy/medium/hard scenarios with a POMDP graph generator that
builds a unique AML investigation on every ``reset()`` call.

Each episode procedurally generates:
  - A random typology: structuring | layering | trade_based_ml
  - A difficulty tier: easy | medium | hard (controls noise volume + graph depth)
  - Dynamic entity IDs, names, timestamps, and amounts
  - Decoy noise: legitimate profiles + clean transactions to clutter the ledger
  - Ground truth: correct decision, typology, key entities, key findings, excluded entities

The generated scenario satisfies the ``BaseScenario`` contract so the existing
tool handlers in ``aml_environment.py`` work without modification.
"""

from __future__ import annotations

import random
import string
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseScenario

# ===========================================================================
# Name / ID pools for randomization
# ===========================================================================

_FIRST_NAMES = [
    "James", "Maria", "Robert", "Patricia", "David", "Jennifer", "Ricardo",
    "Elena", "Ahmed", "Yuki", "Chen", "Ingrid", "Omar", "Svetlana", "Marcus",
    "Fatima", "Raj", "Anastasia", "Kwame", "Mei", "Dmitri", "Isabelle",
    "Tariq", "Chiara", "Hassan", "Yeva", "Emeka", "Sari", "Viktor", "Leila",
    "Anton", "Priya", "Vladimir", "Amara", "Santiago", "Nadia", "Kofi",
    "Lena", "Rafael", "Olga", "Tomoko", "Arjun", "Celine", "Idris",
]

_LAST_NAMES = [
    "Doe", "Smith", "Park", "Chen", "Kim", "Novak", "Webb", "Santos",
    "Korev", "Petrov", "Nguyen", "Okafor", "Al-Rashid", "Tanaka", "Mueller",
    "Garcia", "Johansson", "Patel", "Lee", "Fischer", "Romanov", "Abubakar",
    "Dubois", "Yamamoto", "Ivanov", "O'Brien", "Kumar", "Volkov", "Moreau",
    "Takahashi", "Hoffman", "Costa", "Fernandez", "Zhang", "Andersen",
]

_COMPANY_PREFIXES = [
    "Apex", "Nova", "Prime", "Global", "Crescent", "Delta", "Zenith",
    "Bright", "Meridian", "Summit", "Pacific", "Atlantic", "Nordic",
    "Aurora", "Pinnacle", "Quantum", "Vertex", "Horizon", "Cascade",
    "Sterling", "Eclipse", "Vanguard", "Titan", "Cobalt", "Nexus",
]

_COMPANY_SUFFIXES = [
    "Holdings", "Solutions", "Ventures", "Exports", "Industries", "Trading",
    "Resources", "Capital", "Partners", "Group", "Enterprises", "Corp",
    "International", "Logistics", "Consulting", "Investments", "Systems",
    "Technologies", "Services", "Associates", "Fund", "Global",
]

_JURISDICTIONS_CLEAN = [
    "Delaware, USA", "California, USA", "New York, USA", "London, UK",
    "Singapore", "Hong Kong", "Tokyo, Japan", "Frankfurt, Germany",
    "Zurich, Switzerland", "Toronto, Canada", "Amsterdam, Netherlands",
    "Sydney, Australia", "Stockholm, Sweden", "Dublin, Ireland",
]

_JURISDICTIONS_RISKY = [
    "British Virgin Islands", "Cayman Islands", "Cyprus", "Panama",
    "Seychelles", "Malta", "Liechtenstein", "Isle of Man", "Jersey",
    "Belize", "Mauritius", "Samoa", "Vanuatu",
]

_JURISDICTIONS_FATF = [
    "Myanmar", "Iran", "DPRK", "Syria", "South Sudan", "Yemen",
]

_ADDRESSES = [
    "45 Marina Bay, Singapore 018982",
    "12 Nicosia Business Park, Nicosia, Cyprus",
    "800 Industrial Blvd, Houston, TX",
    "220 Silicon Ave, San Jose, CA",
    "Unit 7, Yangon Trade Centre, Yangon",
    "155 Queen Street, Auckland, NZ",
    "88 Harbour Front, Hong Kong",
    "42 Bahnhofstrasse, Zurich CH-8001",
    "1100 Avenue des Champs-Élysées, Paris",
    "320 King's Road, London SW10",
    "99 Orchard Road, Singapore 238874",
    "10 Raffles Place, Singapore 048619",
]

_OCCUPATIONS_NON_CASH = [
    "Retail Store Clerk", "IT Support Specialist", "Office Administrator",
    "Data Entry Operator", "School Teacher", "Receptionist",
    "Software Developer", "Accountant", "Graphic Designer",
    "Marketing Associate", "Research Analyst", "Library Assistant",
]

_OCCUPATIONS_CASH = [
    "Restaurant Owner", "Laundromat Operator", "Market Vendor",
    "Car Wash Operator", "Parking Lot Manager", "Nightclub Owner",
    "Street Food Vendor", "Taxi Fleet Owner", "Cash-and-Carry Wholesaler",
]

_BRANCH_NAMES = [
    "Downtown Retail — Branch 12", "Midtown Financial — Branch 7",
    "Harbor District — Branch 3", "Westside Plaza — Branch 18",
    "Eastgate Mall — Branch 24", "Central Station — Branch 1",
    "Airport Terminal — Branch 31", "University District — Branch 9",
]

_COMMODITIES = [
    ("machine_parts", "General industrial machine parts", 12000, (8000, 15000)),
    ("electronics_components", "Standard electronics components", 800, (500, 1200)),
    ("textiles", "Bulk textile fabric (per bale)", 5000, (3000, 7000)),
    ("agricultural_products", "Grain/rice per metric ton", 400, (250, 600)),
    ("medical_equipment", "Diagnostic imaging equipment", 45000, (30000, 60000)),
    ("automotive_parts", "OEM automotive components", 3500, (2000, 5000)),
    ("chemicals", "Industrial chemicals (per barrel)", 2000, (1200, 3000)),
    ("precious_metals", "Gold bullion per troy ounce", 2000, (1800, 2200)),
]

_WIRE_DESCRIPTIONS_LEGIT = [
    "Monthly supply payment", "Quarterly retainer", "Service fee",
    "Equipment rental", "Consulting invoice", "Maintenance contract",
    "Software license renewal", "Freight forwarding payment",
]

_WIRE_DESCRIPTIONS_SUSPECT = [
    "Consultancy fees", "Project advance", "Service retainer",
    "Logistics fees", "Investment disbursement", "Advisory services",
    "Commission payment", "Management fee",
]


# ===========================================================================
# Helper functions
# ===========================================================================

def _uid(prefix: str = "C", length: int = 5) -> str:
    """Generate a random alphanumeric ID with a prefix."""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=length))
    return f"{prefix}{suffix}"


def _random_name() -> str:
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def _random_company() -> str:
    return f"{random.choice(_COMPANY_PREFIXES)} {random.choice(_COMPANY_SUFFIXES)}"


def _random_date(start: datetime, end: datetime) -> str:
    """Random date between start and end as YYYY-MM-DD string."""
    delta = (end - start).days
    if delta <= 0:
        return start.strftime("%Y-%m-%d")
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def _dates_in_window(base: datetime, count: int, window_days: int = 7) -> List[str]:
    """Generate `count` sequential dates within a window from `base`."""
    if count <= 0:
        return []
    step = max(1, window_days // count)
    return [(base + timedelta(days=i * step)).strftime("%Y-%m-%d") for i in range(count)]


# ===========================================================================
# Generated Scenario (conforms to BaseScenario)
# ===========================================================================

class GeneratedScenario(BaseScenario):
    """A procedurally generated scenario instance.

    This is a concrete BaseScenario populated from dicts rather than
    hard-coded property methods.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @property
    def initial_alert(self) -> Dict[str, Any]:
        return self._data["initial_alert"]

    @property
    def customer_profiles(self) -> Dict[str, Any]:
        return self._data["customer_profiles"]

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        return self._data["transactions"]

    @property
    def watchlist_results(self) -> Dict[str, Any]:
        return self._data["watchlist_results"]

    @property
    def network_graph(self) -> Dict[str, Any]:
        return self._data["network_graph"]

    @property
    def source_of_funds(self) -> Dict[str, Any]:
        return self._data["source_of_funds"]

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return self._data["ground_truth"]

    @property
    def market_data(self) -> Dict[str, Any]:
        return self._data.get("market_data", {})


# ===========================================================================
# Scenario Generator
# ===========================================================================

class ScenarioGenerator:
    """Procedurally generates AML investigation scenarios.

    Usage::

        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(difficulty="medium")  # or omit for random
        # scenario is a BaseScenario subclass with all 8 data properties

    Each call to ``generate()`` produces a unique scenario with random IDs,
    names, amounts, and timestamps — preventing LLM memorization.
    """

    TYPOLOGIES = ["structuring", "layering", "trade_based_ml"]
    DIFFICULTIES = ["easy", "medium", "hard"]

    # Noise volume: how many decoy profiles/transactions per difficulty
    _NOISE = {
        "easy":   {"decoy_profiles": 0, "decoy_txns": 0},
        "medium": {"decoy_profiles": 2, "decoy_txns": 4},
        "hard":   {"decoy_profiles": 3, "decoy_txns": 8},
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self._txn_counter = 0
        self._ent_counter = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        difficulty: Optional[str] = None,
        typology: Optional[str] = None,
    ) -> GeneratedScenario:
        """Generate a complete scenario.

        Args:
            difficulty: easy | medium | hard (random if None).
            typology: structuring | layering | trade_based_ml (random if None).

        Returns:
            GeneratedScenario conforming to BaseScenario.
        """
        diff = difficulty or random.choice(self.DIFFICULTIES)
        typo = typology or random.choice(self.TYPOLOGIES)

        self._txn_counter = 0
        self._ent_counter = 0

        # Base date range for the episode
        epoch_start = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
        epoch_end = epoch_start + timedelta(days=random.randint(30, 120))

        if typo == "structuring":
            data = self._gen_structuring(diff, epoch_start, epoch_end)
        elif typo == "layering":
            data = self._gen_layering(diff, epoch_start, epoch_end)
        elif typo == "trade_based_ml":
            data = self._gen_tbml(diff, epoch_start, epoch_end)
        else:
            raise ValueError(f"Unknown typology: {typo}")

        # Inject decoy noise into the scenario
        noise_cfg = self._NOISE[diff]
        self._inject_noise(data, noise_cfg, epoch_start, epoch_end)

        # Metadata
        data["_meta"] = {
            "difficulty": diff,
            "typology": typo,
            "generated_at": datetime.utcnow().isoformat(),
        }

        return GeneratedScenario(data)

    # ------------------------------------------------------------------ #
    # ID generators                                                        #
    # ------------------------------------------------------------------ #

    def _next_cust_id(self) -> str:
        return _uid("CUST", 4)

    def _next_ent_id(self) -> str:
        self._ent_counter += 1
        return f"ENT_{string.ascii_uppercase[min(self._ent_counter - 1, 25)]}{random.randint(10,99)}"

    def _next_txn_id(self, prefix: str = "") -> str:
        self._txn_counter += 1
        tag = f"-{prefix}" if prefix else ""
        return f"TXN{tag}-{self._txn_counter:03d}"

    def _next_alert_id(self) -> str:
        return f"ALERT-{random.randint(2024,2025)}-{random.randint(1000,9999)}"

    def _next_acc_id(self) -> str:
        return f"ACC-{random.randint(1000,9999)}"

    # ================================================================== #
    # STRUCTURING GENERATOR                                                #
    # ================================================================== #

    def _gen_structuring(
        self, diff: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Generate a structuring (smurfing) scenario.

        Core pattern: individual makes N sub-$10k cash deposits at the
        same branch over a short window.
        """
        # --- Subject ---
        subject_id = self._next_cust_id()
        subject_name = _random_name()
        account_id = self._next_acc_id()
        branch = random.choice(_BRANCH_NAMES)
        occupation = random.choice(_OCCUPATIONS_NON_CASH)
        income = random.randint(25000, 55000)

        # --- Deposit pattern ---
        num_deposits = random.randint(4, 7)
        deposit_amounts = [
            round(random.uniform(7500, 9900), 2) for _ in range(num_deposits)
        ]
        total = sum(deposit_amounts)
        deposit_dates = _dates_in_window(epoch_start, num_deposits, window_days=num_deposits + 2)

        # --- Build alert ---
        alert_id = self._next_alert_id()
        alert_date = (epoch_start + timedelta(days=num_deposits + 3)).strftime("%Y-%m-%d")

        alert = {
            "alert_id": alert_id,
            "alert_date": alert_date,
            "alert_type": "Multiple Cash Deposits Below Reporting Threshold",
            "risk_score": random.randint(65, 85),
            "priority": "HIGH",
            "customer_id": subject_id,
            "account_id": account_id,
            "summary": (
                f"Customer {subject_id} has made {num_deposits} cash deposits totalling "
                f"${total:,.2f} over a {len(deposit_dates)}-day period "
                f"({deposit_dates[0]} to {deposit_dates[-1]}). "
                f"All deposits are below the $10,000 CTR reporting threshold. "
                f"No business justification is on file."
            ),
            "flagged_rule": "RULE-STRUCT-001: Multiple sub-threshold cash deposits",
            "branch": branch,
        }

        # --- Customer profile ---
        profiles = {
            subject_id: {
                "customer_id": subject_id,
                "name": subject_name,
                "type": "individual",
                "account_type": "Personal Checking",
                "account_age_months": random.randint(6, 60),
                "kyc_risk_tier": "Medium",
                "occupation": occupation,
                "annual_income_declared": income,
                "address": f"{random.randint(100,999)} {random.choice(['Elm','Oak','Pine','Maple','Cedar'])} Street, Springfield, IL",
                "date_of_birth": f"{random.randint(1965,1998)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "pep_status": False,
                "adverse_media": False,
                "expected_monthly_cash": random.randint(500, 2000),
                "business_type": None,
                "notes": f"No prior SARs. Non-cash occupation ({occupation}). No business account.",
            },
        }

        # --- Transactions ---
        transactions = []
        txn_ids = []
        for i, (date, amount) in enumerate(zip(deposit_dates, deposit_amounts)):
            tid = self._next_txn_id("STR")
            txn_ids.append(tid)
            transactions.append({
                "transaction_id": tid,
                "customer_id": subject_id,
                "date": date,
                "type": "cash_deposit",
                "amount": amount,
                "currency": "USD",
                "channel": "branch_teller",
                "branch": branch,
                "description": "Cash deposit",
                "counterparty": None,
            })

        # --- Watchlist ---
        watchlist = {
            subject_name: {
                "entity": subject_name,
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
                "checked_date": alert_date,
            },
            subject_id: {
                "entity": subject_id,
                "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
                "checked_date": alert_date,
            },
        }

        # --- Network ---
        network = {
            subject_id: {
                "entity_id": subject_id,
                "entity_name": subject_name,
                "connections": [],
                "note": "No significant counterparty relationships identified.",
            },
        }

        # --- Source of funds ---
        sof = {}
        for tid in txn_ids:
            sof[tid] = {
                "transaction_id": tid,
                "source": "cash",
                "documentation": None,
                "verified": False,
                "notes": (
                    "Cash deposited at branch teller. No supporting documentation provided. "
                    "Customer could not provide explanation for source of funds when asked."
                ),
            }

        # --- Ground truth ---
        ground_truth = {
            "correct_decision": "file_sar",
            "typology": "structuring",
            "key_entities": [subject_id],
            "excluded_entities": [],
            "key_findings": [
                "multiple_sub_threshold_deposits",
                "no_cash_intensive_occupation",
                "same_branch_repeated",
                "no_source_documentation",
                "total_exceeds_ctr_threshold",
            ],
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": watchlist,
            "network_graph": network,
            "source_of_funds": sof,
            "ground_truth": ground_truth,
            "market_data": {},
        }

    # ================================================================== #
    # LAYERING GENERATOR                                                   #
    # ================================================================== #

    def _gen_layering(
        self, diff: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Generate a layering / shell company scenario.

        Core pattern: entity receives large inbound wire and fans out to
        N shell entities within 24-48h. Includes PEP connection and
        shared registered addresses.
        """
        # --- Subject (hub entity) ---
        subject_id = self._next_cust_id()
        subject_name = _random_company()
        subject_director = _random_name()
        account_id = self._next_acc_id()

        # --- Source entity (offshore) ---
        source_id = self._next_ent_id()
        source_name = _random_company()
        source_jurisdiction = random.choice(_JURISDICTIONS_RISKY)

        # --- Fan-out shell entities ---
        num_shells = random.randint(3, 5)
        shared_address = random.choice(_ADDRESSES)
        pep_name = _random_name()

        shells = []
        for i in range(num_shells):
            eid = self._next_ent_id()
            ename = _random_company()
            is_pep_shell = (i == num_shells - 1)  # last one is PEP-connected
            is_shared_addr = (i < 2)  # first two share an address

            shell_profile = {
                "customer_id": eid,
                "name": ename,
                "type": "business",
                "jurisdiction": random.choice(_JURISDICTIONS_RISKY) if not is_shared_addr else "Singapore",
                "registered_address": shared_address if is_shared_addr else random.choice(_ADDRESSES),
                "beneficial_owners": "Unknown",
                "notes": "Shell entity; no public business activity."
                         + (f" Shares registered address with {shells[0]['name']}." if is_shared_addr and i > 0 else ""),
            }

            if is_pep_shell:
                shell_profile["director"] = pep_name
                shell_profile["notes"] = f"Director {pep_name} is a PEP (former government minister)."

            shells.append({**shell_profile, "_is_pep_shell": is_pep_shell, "_is_shared": is_shared_addr})

        # --- Legitimate entity (decoy) ---
        legit_id = self._next_ent_id()
        legit_name = _random_company()

        # --- Build amounts ---
        inbound_amount = random.choice([250_000, 350_000, 500_000, 750_000, 1_000_000])
        fan_amounts = self._split_amount(inbound_amount, num_shells)

        inbound_date = epoch_start.strftime("%Y-%m-%d")
        fan_date = (epoch_start + timedelta(days=random.choice([0, 1]))).strftime("%Y-%m-%d")
        fan_date_2 = (epoch_start + timedelta(days=1)).strftime("%Y-%m-%d")

        alert_id = self._next_alert_id()
        alert_date = (epoch_start + timedelta(days=5)).strftime("%Y-%m-%d")

        # --- Alert ---
        shell_names = [s["name"] for s in shells]
        alert = {
            "alert_id": alert_id,
            "alert_date": alert_date,
            "alert_type": "Rapid Fan-Out Wire Transfers",
            "risk_score": random.randint(80, 95),
            "priority": "HIGH",
            "customer_id": subject_id,
            "account_id": account_id,
            "summary": (
                f"{subject_name} ({subject_id}) received an inbound wire of ${inbound_amount:,.0f} from "
                f"{source_name} on {inbound_date} and dispersed funds to {', '.join(shell_names[:3])} "
                f"within 24 hours. The account was opened {random.randint(3,9)} months ago. "
                f"No trade documentation on file."
            ),
            "flagged_rule": "RULE-LAYER-002: Rapid fan-out of large inbound wire",
            "total_inflow": float(inbound_amount + 5000),
            "total_outflow": float(inbound_amount),
            "net_remaining": 5000.00,
        }

        # --- Profiles ---
        profiles = {
            subject_id: {
                "customer_id": subject_id,
                "name": subject_name,
                "type": "business",
                "account_type": "Business Checking",
                "account_age_months": random.randint(3, 9),
                "kyc_risk_tier": "High",
                "business_type": "Import/Export",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN[:3]),
                "director": subject_director,
                "annual_revenue_declared": random.randint(1_000_000, 5_000_000),
                "pep_status": False,
                "adverse_media": False,
                "beneficial_owners": [{"name": subject_director, "ownership_pct": 100}],
                "notes": (
                    "Newly incorporated company. Limited trading history. "
                    "No audited financials on file."
                ),
            },
            source_id: {
                "customer_id": source_id,
                "name": source_name,
                "type": "business",
                "jurisdiction": source_jurisdiction,
                "beneficial_owners": "Unknown",
                "notes": "Offshore holding company. Beneficial ownership not disclosed.",
            },
            legit_id: {
                "customer_id": legit_id,
                "name": legit_name,
                "type": "business",
                "jurisdiction": "USA",
                "registered_address": f"{random.randint(100,999)} Industrial Blvd, Houston, TX",
                "notes": f"Long-established supplier. Regular monthly payments. Legitimate commercial relationship with {subject_name}.",
            },
        }
        for s in shells:
            s_copy = {k: v for k, v in s.items() if not k.startswith("_")}
            profiles[s["customer_id"]] = s_copy

        # --- Transactions ---
        transactions = []

        # Inbound from source
        inbound_tid = self._next_txn_id("L")
        transactions.append({
            "transaction_id": inbound_tid,
            "customer_id": subject_id,
            "sender_id": source_id,
            "receiver_id": subject_id,
            "date": inbound_date,
            "type": "wire_inbound",
            "amount": float(inbound_amount),
            "currency": "USD",
            "description": random.choice(_WIRE_DESCRIPTIONS_SUSPECT),
            "reference": f"INV-{_uid('',4)}",
            "same_day_outflow": True,
        })

        # Fan-outs
        for i, (shell, amt) in enumerate(zip(shells, fan_amounts)):
            tid = self._next_txn_id("L")
            transactions.append({
                "transaction_id": tid,
                "customer_id": subject_id,
                "sender_id": subject_id,
                "receiver_id": shell["customer_id"],
                "date": fan_date if i < len(shells) - 1 else fan_date_2,
                "type": "wire_outbound",
                "amount": float(amt),
                "currency": "USD",
                "description": random.choice(_WIRE_DESCRIPTIONS_SUSPECT),
                "reference": f"REF-{_uid('',4)}",
            })

        # Legit recurring from decoy
        for m in range(2):
            tid = self._next_txn_id("L")
            d = (epoch_start - timedelta(days=30 * (m + 1))).strftime("%Y-%m-%d")
            transactions.append({
                "transaction_id": tid,
                "customer_id": subject_id,
                "sender_id": legit_id,
                "receiver_id": subject_id,
                "date": d,
                "type": "wire_inbound",
                "amount": 5000.00,
                "currency": "USD",
                "description": random.choice(_WIRE_DESCRIPTIONS_LEGIT),
                "reference": f"INV-{_uid('',4)}",
            })

        # --- Watchlist ---
        watchlist: Dict[str, Any] = {}
        watchlist[subject_name] = {
            "entity": subject_name, "hit": False,
            "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
            "result": "No matches found",
        }
        watchlist[source_name] = {
            "entity": source_name, "hit": False,
            "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
            "result": f"No direct match; beneficial ownership unknown — enhanced due diligence recommended",
        }
        watchlist[subject_director] = {
            "entity": subject_director, "hit": False,
            "lists_checked": ["OFAC SDN", "PEP"],
            "result": "No matches found",
        }
        watchlist[legit_name] = {
            "entity": legit_name, "hit": False,
            "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
            "result": "No matches found",
        }
        for s in shells:
            watchlist[s["name"]] = {
                "entity": s["name"], "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found",
            }
            if s["_is_pep_shell"]:
                watchlist[pep_name] = {
                    "entity": pep_name, "hit": True,
                    "lists_checked": ["PEP"],
                    "result": (
                        f"MATCH — PEP list: {pep_name}, former Deputy Minister of Finance, "
                        f"Republic of Coravia. Current director of {s['name']}."
                    ),
                    "risk_level": "HIGH",
                }

        # --- Network graph ---
        network: Dict[str, Any] = {}
        hub_connections = []
        hub_connections.append({
            "entity_id": source_id, "entity_name": source_name,
            "relationship": "inbound_wire_sender", "amount": inbound_amount,
        })
        for s, amt in zip(shells, fan_amounts):
            conn: Dict[str, Any] = {
                "entity_id": s["customer_id"], "entity_name": s["name"],
                "relationship": "outbound_wire_receiver", "amount": amt,
            }
            if s["_is_shared"]:
                conn["registered_address"] = shared_address
                if shells.index(s) > 0:
                    conn["note"] = f"SHARED ADDRESS with {shells[0]['name']}"
            if s["_is_pep_shell"]:
                conn["director"] = f"{pep_name} (PEP)"
            hub_connections.append(conn)
        hub_connections.append({
            "entity_id": legit_id, "entity_name": legit_name,
            "relationship": "legitimate_supplier_inbound", "amount": 5000,
            "pattern": "monthly_recurring",
        })
        network[subject_id] = {
            "entity_id": subject_id, "entity_name": subject_name,
            "connections": hub_connections,
        }

        # Shared-address connections between shells
        if len(shells) >= 2 and shells[0]["_is_shared"] and shells[1]["_is_shared"]:
            for a, b in [(0, 1), (1, 0)]:
                network[shells[a]["customer_id"]] = {
                    "entity_id": shells[a]["customer_id"],
                    "entity_name": shells[a]["name"],
                    "connections": [{
                        "entity_id": shells[b]["customer_id"],
                        "entity_name": shells[b]["name"],
                        "relationship": "shared_registered_address",
                        "address": shared_address,
                    }],
                }

        # PEP shell director link
        pep_shell = [s for s in shells if s["_is_pep_shell"]][0]
        network[pep_shell["customer_id"]] = {
            "entity_id": pep_shell["customer_id"],
            "entity_name": pep_shell["name"],
            "connections": [{
                "entity_id": pep_name, "entity_name": pep_name,
                "relationship": "director", "pep": True,
            }],
        }
        network[legit_id] = {
            "entity_id": legit_id, "entity_name": legit_name,
            "connections": [], "note": "Legitimate supplier; no suspicious connections.",
        }

        # --- Source of funds ---
        sof: Dict[str, Any] = {}
        sof[inbound_tid] = {
            "transaction_id": inbound_tid,
            "source": f"{source_name} ({source_jurisdiction})",
            "documentation": None, "verified": False,
            "notes": f"{source_name} is an offshore entity. No underlying trade contract or invoice provided.",
        }
        for txn in transactions:
            if txn["transaction_id"] == inbound_tid:
                continue
            if txn["sender_id"] == subject_id:
                sof[txn["transaction_id"]] = {
                    "transaction_id": txn["transaction_id"],
                    "source": "Redistributed from inbound wire",
                    "documentation": None, "verified": False,
                    "notes": f"No service contract on file for {txn['receiver_id']}.",
                }
            elif txn["sender_id"] == legit_id:
                sof[txn["transaction_id"]] = {
                    "transaction_id": txn["transaction_id"],
                    "source": f"{legit_name} — trade payment",
                    "documentation": f"{txn.get('reference', 'INV')} (verified)",
                    "verified": True,
                    "notes": "Legitimate monthly supply invoice. No concerns.",
                }

        # --- Ground truth ---
        key_entities = [subject_id, source_id] + [s["customer_id"] for s in shells]
        ground_truth = {
            "correct_decision": "file_sar",
            "typology": "layering",
            "key_entities": key_entities,
            "excluded_entities": [legit_id],
            "key_findings": [
                "rapid_fan_out",
                "pep_connection",
                "shared_registered_address",
                "offshore_source",
                "newly_incorporated",
                "no_trade_documentation",
            ],
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": watchlist,
            "network_graph": network,
            "source_of_funds": sof,
            "ground_truth": ground_truth,
            "market_data": {},
        }

    # ================================================================== #
    # TRADE-BASED ML GENERATOR                                             #
    # ================================================================== #

    def _gen_tbml(
        self, diff: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Generate a Trade-Based Money Laundering scenario.

        Core pattern: entity buys commodity from offshore supplier at
        grossly inflated prices. Beneficial owner of the supplier is
        related to the buyer's director.
        """
        # --- Subject (buyer) ---
        subject_id = self._next_cust_id()
        subject_name = _random_company()
        subject_director = _random_name()
        account_id = self._next_acc_id()

        # --- Offshore supplier (over-invoiced) ---
        supplier_id = self._next_ent_id()
        supplier_name = _random_company()
        supplier_director = _random_name()
        beneficial_owner = _random_name()  # related to subject_director
        supplier_jurisdiction = random.choice(_JURISDICTIONS_FATF)

        # --- Pick a commodity ---
        commodity_name, commodity_desc, market_price, price_range = random.choice(_COMMODITIES)
        over_invoice_multiplier = random.uniform(2.5, 5.0)
        invoiced_price = round(market_price * over_invoice_multiplier)
        premium_pct = round((invoiced_price / market_price - 1) * 100)

        # --- Unexplained inflow entity ---
        mystery_id = self._next_ent_id()
        mystery_name = _random_company()
        mystery_jurisdiction = random.choice(_JURISDICTIONS_RISKY)
        mystery_amount = random.choice([100_000, 150_000, 200_000, 300_000])

        # --- Legitimate supplier (red herring) ---
        legit_id = self._next_ent_id()
        legit_name = _random_company()

        # --- Invoice transactions ---
        num_invoices = random.randint(8, 14)
        invoice_dates = []
        d = epoch_start
        for _ in range(num_invoices):
            d += timedelta(days=random.randint(10, 25))
            invoice_dates.append(d.strftime("%Y-%m-%d"))

        total_outflow = invoiced_price * num_invoices
        expected_at_market = market_price * num_invoices
        overpayment = total_outflow - expected_at_market

        alert_id = self._next_alert_id()
        alert_date = (d + timedelta(days=3)).strftime("%Y-%m-%d")

        # --- Reversed transaction ---
        reversed_index = random.randint(num_invoices // 2, num_invoices - 1)
        adjusted_price = invoiced_price - random.randint(100, 500)

        # --- Alert ---
        alert = {
            "alert_id": alert_id,
            "alert_date": alert_date,
            "alert_type": "Potential Trade-Based Money Laundering",
            "risk_score": random.randint(85, 98),
            "priority": "CRITICAL",
            "customer_id": subject_id,
            "account_id": account_id,
            "summary": (
                f"{subject_name} ({subject_id}) has conducted {num_invoices} payments over "
                f"{len(invoice_dates)} months to {supplier_name} ({supplier_id}), an offshore entity "
                f"registered in a FATF-monitored jurisdiction ({supplier_jurisdiction}), "
                f"totalling ${total_outflow:,.0f} for '{commodity_name}'. "
                f"Market intelligence suggests comparable goods trade at approximately "
                f"${market_price:,.0f}/unit. "
                f"Additionally, {subject_id} received an unexplained ${mystery_amount:,.0f} wire from "
                f"{mystery_name} ({mystery_id}). "
                f"One transaction was reversed and re-initiated with a slightly different amount."
            ),
            "flagged_rule": "RULE-TBML-003: Systematic over-invoicing pattern",
            "total_outflow_to_supplier": float(total_outflow),
            "unexplained_inflow": float(mystery_amount),
            "units_purchased": num_invoices,
        }

        # --- Profiles ---
        relationship_type = random.choice(["brother-in-law", "cousin", "business partner", "uncle"])
        profiles = {
            subject_id: {
                "customer_id": subject_id,
                "name": subject_name,
                "type": "business",
                "account_type": "Business Checking",
                "account_age_months": random.randint(12, 36),
                "kyc_risk_tier": "Medium-High",
                "business_type": f"{commodity_desc} Import / Distribution",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN[:3]),
                "director": subject_director,
                "annual_revenue_declared": random.randint(2_000_000, 10_000_000),
                "pep_status": False,
                "adverse_media": False,
                "beneficial_owners": [
                    {"name": subject_director, "ownership_pct": random.randint(50, 80)},
                    {"name": _random_name(), "ownership_pct": random.randint(20, 50)},
                ],
                "suppliers_declared": [supplier_name, legit_name],
                "notes": (
                    f"Company with moderate transaction volume historically. "
                    f"Sudden spike in high-value outflows to offshore counterparty."
                ),
            },
            supplier_id: {
                "customer_id": supplier_id,
                "name": supplier_name,
                "type": "business",
                "jurisdiction": supplier_jurisdiction,
                "fatf_flagged": True,
                "fatf_status": "FATF Monitored Jurisdiction (Increased Monitoring)",
                "registered_address": random.choice(_ADDRESSES),
                "director": supplier_director,
                "beneficial_owner": beneficial_owner,
                "beneficial_owner_notes": (
                    f"{beneficial_owner} is the beneficial owner of {supplier_name}. "
                    f"{beneficial_owner} is the {relationship_type} of {subject_director} "
                    f"(director of {subject_id} / {subject_name})."
                ),
                "notes": "Offshore exporter in FATF-monitored jurisdiction. Limited public business presence.",
            },
            mystery_id: {
                "customer_id": mystery_id,
                "name": mystery_name,
                "type": "fund",
                "jurisdiction": mystery_jurisdiction,
                "notes": f"Offshore investment fund. No disclosed relationship with {subject_name}. Source of ${mystery_amount:,.0f} wire unexplained.",
            },
            legit_id: {
                "customer_id": legit_id,
                "name": legit_name,
                "type": "business",
                "jurisdiction": "USA",
                "registered_address": f"{random.randint(100,999)} Commerce Blvd, San Jose, CA",
                "notes": (
                    f"Well-established domestic distributor. "
                    f"{subject_name} has purchased from {legit_name} at market prices. "
                    f"Legitimate long-term supplier relationship."
                ),
            },
        }

        # --- Transactions ---
        transactions = []
        txn_ids_supplier = []

        for i, date in enumerate(invoice_dates):
            if i == reversed_index:
                # Reversed
                tid_rev = self._next_txn_id("TB")
                transactions.append({
                    "transaction_id": tid_rev,
                    "customer_id": subject_id,
                    "sender_id": subject_id,
                    "receiver_id": supplier_id,
                    "date": date,
                    "type": "wire_outbound_reversed",
                    "amount": float(invoiced_price),
                    "currency": "USD",
                    "description": f"Purchase of {commodity_name} — 1 unit [REVERSED]",
                    "reversal": True,
                    "reversal_reason": "Beneficiary bank correction",
                    "invoice_ref": f"INV-{_uid('',3)}-REV",
                })
                txn_ids_supplier.append(tid_rev)
                # Re-sent
                next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                tid_fix = self._next_txn_id("TB")
                transactions.append({
                    "transaction_id": tid_fix,
                    "customer_id": subject_id,
                    "sender_id": subject_id,
                    "receiver_id": supplier_id,
                    "date": next_day,
                    "type": "wire_outbound",
                    "amount": float(adjusted_price),
                    "currency": "USD",
                    "description": f"Purchase of {commodity_name} — 1 unit [re-issued]",
                    "reversal_of": tid_rev,
                    "invoice_ref": f"INV-{_uid('',3)}-B",
                    "goods": commodity_name,
                    "units": 1,
                    "unit_price": float(adjusted_price),
                    "note": f"Amount adjusted from ${invoiced_price:,.0f} to ${adjusted_price:,.0f} — reason not documented",
                })
                txn_ids_supplier.append(tid_fix)
            else:
                tid = self._next_txn_id("TB")
                txn_ids_supplier.append(tid)
                transactions.append({
                    "transaction_id": tid,
                    "customer_id": subject_id,
                    "sender_id": subject_id,
                    "receiver_id": supplier_id,
                    "date": date,
                    "type": "wire_outbound",
                    "amount": float(invoiced_price),
                    "currency": "USD",
                    "description": f"Purchase of {commodity_name} — 1 unit",
                    "invoice_ref": f"INV-{_uid('',3)}",
                    "goods": commodity_name,
                    "units": 1,
                    "unit_price": float(invoiced_price),
                })

        # Unexplained inbound
        mystery_tid = self._next_txn_id("TB")
        mystery_date = _random_date(epoch_start, epoch_end)
        transactions.append({
            "transaction_id": mystery_tid,
            "customer_id": subject_id,
            "sender_id": mystery_id,
            "receiver_id": subject_id,
            "date": mystery_date,
            "type": "wire_inbound",
            "amount": float(mystery_amount),
            "currency": "USD",
            "description": "Investment disbursement",
            "reference": f"CIIF-{random.randint(2024,2025)}-{random.randint(1000,9999)}",
            "note": "No investment agreement on file. No prior relationship disclosed.",
        })

        # Legitimate purchases (red herring)
        for m in range(3):
            tid = self._next_txn_id("TB")
            ld = _random_date(epoch_start, epoch_end)
            legit_amt = random.randint(market_price // 2, market_price * 2)
            transactions.append({
                "transaction_id": tid,
                "customer_id": subject_id,
                "sender_id": subject_id,
                "receiver_id": legit_id,
                "date": ld,
                "type": "wire_outbound",
                "amount": float(legit_amt),
                "currency": "USD",
                "description": f"{commodity_desc} purchase",
                "invoice_ref": f"INV-{_uid('',3)}",
                "goods": commodity_name,
                "unit_price_market_aligned": True,
                "note": "Price consistent with market rates. Long-term supplier.",
            })

        # --- Watchlist ---
        watchlist = {
            subject_name: {
                "entity": subject_name, "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "PEP"],
                "result": "No matches found",
            },
            supplier_name: {
                "entity": supplier_name, "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": f"No direct sanction match; entity registered in FATF-monitored jurisdiction ({supplier_jurisdiction})",
                "risk_flag": "FATF monitored jurisdiction",
            },
            beneficial_owner: {
                "entity": beneficial_owner, "hit": False,
                "lists_checked": ["OFAC SDN", "PEP", "Interpol"],
                "result": "No sanction or PEP match",
                "note": f"Beneficial owner of {supplier_name}.",
            },
            subject_director: {
                "entity": subject_director, "hit": False,
                "lists_checked": ["OFAC SDN", "PEP"],
                "result": "No matches found",
            },
            mystery_name: {
                "entity": mystery_name, "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated"],
                "result": f"No direct match; {mystery_jurisdiction} is a high-risk jurisdiction for fund transparency",
                "risk_flag": "High-risk jurisdiction",
            },
            legit_name: {
                "entity": legit_name, "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found. Well-known domestic distributor.",
            },
        }

        # --- Network ---
        network = {
            subject_id: {
                "entity_id": subject_id,
                "entity_name": subject_name,
                "connections": [
                    {
                        "entity_id": supplier_id, "entity_name": supplier_name,
                        "relationship": "primary_supplier_outbound",
                        "total_amount": total_outflow,
                        "transaction_count": num_invoices,
                        "jurisdiction": f"{supplier_jurisdiction} (FATF monitored)",
                    },
                    {
                        "entity_id": mystery_id, "entity_name": mystery_name,
                        "relationship": "unexplained_inbound", "amount": mystery_amount,
                    },
                    {
                        "entity_id": legit_id, "entity_name": legit_name,
                        "relationship": "legitimate_supplier_outbound",
                        "total_amount": market_price * 3,
                        "note": "Market-rate transactions; long-term relationship",
                    },
                    {
                        "entity_id": subject_director, "entity_name": subject_director,
                        "relationship": "director",
                    },
                ],
            },
            supplier_id: {
                "entity_id": supplier_id,
                "entity_name": supplier_name,
                "depth_1_connections": [
                    {"entity_id": supplier_director, "entity_name": supplier_director, "relationship": "registered_director"},
                    {"entity_id": beneficial_owner, "entity_name": beneficial_owner, "relationship": "beneficial_owner"},
                ],
                "depth_2_connections": [
                    {
                        "entity_id": subject_director, "entity_name": subject_director,
                        "via": beneficial_owner,
                        "relationship": relationship_type,
                        "note": (
                            f"{beneficial_owner} (beneficial owner of {supplier_name}) is the "
                            f"{relationship_type} of {subject_director} (director of {subject_name} / {subject_id}). "
                            f"This creates a potential undisclosed conflict of interest."
                        ),
                    },
                ],
            },
            legit_id: {
                "entity_id": legit_id, "entity_name": legit_name,
                "connections": [], "note": "No suspicious connections. Clean counterparty.",
            },
        }

        # --- Source of funds ---
        sof: Dict[str, Any] = {}
        for tid in txn_ids_supplier:
            sof[tid] = {
                "transaction_id": tid,
                "source": "Business operating account",
                "documentation": f"INV (copy on file)",
                "verified": False,
                "notes": (
                    f"Invoice presented for {commodity_name}. Unit price ${invoiced_price:,.0f}. "
                    f"Market data shows comparable goods at ${market_price:,.0f}/unit. "
                    f"Invoice does not include part number, model spec, or shipping manifest."
                ),
            }
        sof[mystery_tid] = {
            "transaction_id": mystery_tid,
            "source": mystery_name,
            "documentation": None,
            "verified": False,
            "notes": f"No investment agreement or subscription documents on file. Source of funds unknown.",
        }

        # --- Market data ---
        market_data = {
            commodity_name: {
                "description": commodity_desc,
                "market_unit_price_usd": market_price,
                "price_range_usd": list(price_range),
                "source": "Industry trade database Q1-2024",
                "note": (
                    f"Invoiced price of ${invoiced_price:,.0f}/unit represents a {premium_pct}% premium "
                    f"over market. {num_invoices} units totalling ${total_outflow:,.0f} vs expected "
                    f"${expected_at_market:,.0f} — overpayment of ${overpayment:,.0f}."
                ),
            },
        }

        # --- Ground truth ---
        ground_truth = {
            "correct_decision": "file_sar",
            "typology": "trade_based_ml",
            "key_entities": [subject_id, supplier_id, beneficial_owner],
            "excluded_entities": [legit_id],
            "key_findings": [
                "over_invoicing",
                "beneficial_owner_connection",
                "fatf_jurisdiction",
                "reversed_transaction",
                "unexplained_funds",
            ],
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": watchlist,
            "network_graph": network,
            "source_of_funds": sof,
            "ground_truth": ground_truth,
            "market_data": market_data,
        }

    # ================================================================== #
    # NOISE INJECTION                                                      #
    # ================================================================== #

    def _inject_noise(
        self,
        data: Dict[str, Any],
        noise_cfg: Dict[str, int],
        epoch_start: datetime,
        epoch_end: datetime,
    ) -> None:
        """Add decoy profiles and legitimate transactions to the scenario."""
        num_profiles = noise_cfg["decoy_profiles"]
        num_txns = noise_cfg["decoy_txns"]

        existing_subject = data["initial_alert"]["customer_id"]

        for _ in range(num_profiles):
            decoy_id = self._next_ent_id()
            decoy_name = _random_company()
            decoy_jurisdiction = random.choice(_JURISDICTIONS_CLEAN)

            data["customer_profiles"][decoy_id] = {
                "customer_id": decoy_id,
                "name": decoy_name,
                "type": "business",
                "jurisdiction": decoy_jurisdiction,
                "registered_address": random.choice(_ADDRESSES),
                "notes": f"Legitimate commercial entity. Regular transactions. No suspicious activity.",
            }

            data["watchlist_results"][decoy_name] = {
                "entity": decoy_name, "hit": False,
                "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions"],
                "result": "No matches found",
            }

            data["network_graph"][decoy_id] = {
                "entity_id": decoy_id, "entity_name": decoy_name,
                "connections": [], "note": "No suspicious connections.",
            }

            # Add to excluded_entities in ground truth
            if "excluded_entities" not in data["ground_truth"]:
                data["ground_truth"]["excluded_entities"] = []
            data["ground_truth"]["excluded_entities"].append(decoy_id)

        for _ in range(num_txns):
            tid = self._next_txn_id("N")
            decoy_ent = random.choice(
                [k for k in data["customer_profiles"] if k != existing_subject]
            ) if len(data["customer_profiles"]) > 1 else self._next_ent_id()

            d = _random_date(epoch_start, epoch_end)
            amt = round(random.uniform(500, 15000), 2)

            data["transactions"].append({
                "transaction_id": tid,
                "customer_id": existing_subject,
                "sender_id": decoy_ent,
                "receiver_id": existing_subject,
                "date": d,
                "type": random.choice(["wire_inbound", "ach_inbound", "check_deposit"]),
                "amount": amt,
                "currency": "USD",
                "description": random.choice(_WIRE_DESCRIPTIONS_LEGIT),
                "reference": f"REF-{_uid('',4)}",
                "note": "Routine commercial payment. No concerns.",
            })

            data["source_of_funds"][tid] = {
                "transaction_id": tid,
                "source": "Legitimate commercial payment",
                "documentation": f"Invoice verified",
                "verified": True,
                "notes": "Standard business transaction. No concerns.",
            }

    # ================================================================== #
    # HELPERS                                                              #
    # ================================================================== #

    @staticmethod
    def _split_amount(total: int, n: int) -> List[int]:
        """Split total into n unequal parts that sum to total."""
        if n <= 0:
            return []
        cuts = sorted(random.sample(range(1, total), min(n - 1, total - 1)))
        parts = []
        prev = 0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(total - prev)
        # Ensure exactly n parts
        while len(parts) < n:
            parts.append(0)
        return parts[:n]


# ===========================================================================
# Module-level convenience
# ===========================================================================

_default_generator: Optional[ScenarioGenerator] = None


def generate_scenario(
    difficulty: Optional[str] = None,
    typology: Optional[str] = None,
    seed: Optional[int] = None,
) -> GeneratedScenario:
    """Module-level convenience function to generate a scenario."""
    global _default_generator
    if seed is not None:
        _default_generator = ScenarioGenerator(seed=seed)
    if _default_generator is None:
        _default_generator = ScenarioGenerator()
    return _default_generator.generate(difficulty=difficulty, typology=typology)
