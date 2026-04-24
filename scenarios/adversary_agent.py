"""
Memex OS-Agent Benchmark — Adversarial Scenario Generator ("Money Launderer" Agent).

This standalone module generates complex, evasive AML scenarios designed to
defeat the PPO-trained Defender agent. It uses an LLM (via OpenAI-compatible API)
to dynamically create transaction graphs with advanced evasion techniques.

Typologies supported:
  - mule_ring: Shared device fingerprints, cascading transfers across mule accounts
  - pass_through: Circular fund flows through shell companies
  - phantom_invoice: Trade-based ML with phantom shipments

The output conforms to the GeneratedScenario contract (from procedural_generator.py)
so it can be fed directly to AMLEnvironment.reset().

Usage:
    from scenarios.adversary_agent import AdversaryAgent
    agent = AdversaryAgent(model="gpt-4o-mini")
    scenario_data = agent.generate(typology="mule_ring", difficulty="hard")
"""

from __future__ import annotations

import json
import os
import random
import re
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .procedural_generator import (
    GeneratedScenario,
    _random_name,
    _random_company,
    _random_date,
    _random_ip,
    _random_device_id,
    _random_mac,
    _random_coords,
    _uid,
    _JURISDICTIONS_CLEAN,
    _JURISDICTIONS_RISKY,
    _JURISDICTIONS_FATF,
    _ADDRESSES,
    _BRANCH_NAMES,
    _WIRE_DESCRIPTIONS_SUSPECT,
    _COMMODITIES,
    _HS_CODES,
)


# ---------------------------------------------------------------------------
# LLM-backed Adversarial Agent
# ---------------------------------------------------------------------------

# System prompt for the adversary LLM
_ADVERSARY_SYSTEM_PROMPT = """You are a Red Team scenario generator for an AML (Anti-Money Laundering) investigation benchmark.

Your job is to generate a REALISTIC and EVASIVE money laundering scenario that will be difficult for an AI investigator to detect.

RULES:
1. Generate valid JSON that matches the schema provided.
2. Make the scenario realistic — use plausible names, amounts, and jurisdictions.
3. Use evasion techniques: mix legitimate transactions with suspicious ones, use intermediaries, time delays, jurisdiction hopping.
4. The scenario MUST have a clear ground truth (the correct answer), but it should be HIDDEN behind noise.
5. Include at least 2-3 "red herrings" — entities or transactions that look suspicious but are actually clean.
6. Never use amounts that are obviously round numbers ($1,000,000 exactly). Use realistic amounts ($987,431.22).

You will respond ONLY with valid JSON. No markdown, no explanation."""


_SCENARIO_SCHEMA = """
Generate a scenario with this JSON structure:
{
  "typology": "<mule_ring|pass_through|phantom_invoice>",
  "difficulty": "<easy|medium|hard>",
  "evasion_techniques": ["list of techniques used"],
  "entities": [
    {
      "id": "ENT_XX",
      "name": "Full Name or Company Name",
      "type": "individual|company|shell",
      "jurisdiction": "Country/Region",
      "is_suspicious": true/false,
      "is_red_herring": true/false,
      "role": "subject|intermediary|beneficiary|decoy",
      "pep": true/false,
      "occupation": "string or null"
    }
  ],
  "transactions": [
    {
      "from": "ENT_XX",
      "to": "ENT_YY",
      "amount": 123456.78,
      "currency": "USD",
      "date": "YYYY-MM-DD",
      "type": "wire_transfer|cash_deposit|trade_payment|internal_transfer",
      "description": "string",
      "is_suspicious": true/false
    }
  ],
  "device_overlaps": [
    {
      "entities": ["ENT_XX", "ENT_YY"],
      "shared_device_id": "DEV-XXXX",
      "shared_ip": "IP address"
    }
  ],
  "phantom_invoices": [
    {
      "invoice_id": "INV-XXX",
      "commodity": "string",
      "declared_value": 123456.78,
      "actual_value": 0,
      "origin": "Country",
      "destination": "Country"
    }
  ],
  "ground_truth": {
    "correct_decision": "file_sar|close_alert",
    "typology": "string",
    "key_entities": ["ENT_XX"],
    "excluded_entities": ["ENT_YY"],
    "key_findings": ["finding1", "finding2"],
    "red_flags": ["flag1", "flag2"]
  }
}
"""


class AdversaryAgent:
    """LLM-powered adversarial scenario generator.

    Uses a local Llama-3.1-8B model (via Unsloth) or an OpenAI-compatible API
    to generate complex, evasive AML scenarios that are designed to fool the
    PPO Defender agent.  Falls back to procedural generation when no LLM is
    available.

    Args:
        model: Model name (default: gpt-4o-mini, overridden to local 8B when
               run via train_adversary.py --local).
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        base_url: Base URL for OpenAI-compatible API.
        temperature: Sampling temperature for creativity.
    """

    TYPOLOGIES = ["mule_ring", "pass_through", "phantom_invoice"]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.9,
        is_local: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.temperature = temperature
        self.is_local = is_local
        self.local_model = None
        self.local_tokenizer = None

    def generate(
        self,
        typology: Optional[str] = None,
        difficulty: str = "hard",
    ) -> Dict[str, Any]:
        """Generate an adversarial scenario.

        Args:
            typology: Specific typology or None for random.
            difficulty: easy | medium | hard.

        Returns:
            Dict conforming to GeneratedScenario data contract.
        """
        typo = typology or random.choice(self.TYPOLOGIES)

        # Local LLM generation
        if self.is_local:
            local_scenario = self._generate_via_local_llm(typo, difficulty)
            if local_scenario:
                result = self._normalize_to_scenario(local_scenario, typo, difficulty)
            else:
                result = self._generate_fallback(typo, difficulty)
        else:
            # API LLM generation
            llm_scenario = self._generate_via_llm(typo, difficulty)
            if llm_scenario:
                result = self._normalize_to_scenario(llm_scenario, typo, difficulty)
            else:
                # Fallback: enhanced procedural generation
                result = self._generate_fallback(typo, difficulty)

        # Synthesize missing data fields (watchlist, network, SoF, UBO)
        return self._synthesize_missing_data(result)

    @staticmethod
    def _synthesize_missing_data(scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Populate empty watchlist, network, source_of_funds, and beneficial_ownership
        from existing profiles and transactions so all env tools return useful data."""
        profiles = scenario.get("customer_profiles", {})
        transactions = scenario.get("transactions", [])
        gt = scenario.get("ground_truth", {})
        key_entities = set(gt.get("key_entities", []))
        excluded = set(gt.get("excluded_entities", []))

        # -- Watchlist: key entities get PEP/FATF hits, excluded get clean --
        if not scenario.get("watchlist_results"):
            wl: Dict[str, Any] = {}
            for eid, prof in profiles.items():
                jurisdiction = prof.get("jurisdiction", prof.get("nationality", "Unknown"))
                if eid in key_entities:
                    is_risky = jurisdiction in (
                        "Seychelles", "BVI", "Cayman Islands", "Panama", "Vanuatu",
                        "Myanmar", "Iran", "DPRK", "Syria", "Unknown",
                    )
                    wl[eid] = {
                        "match": True,
                        "lists": ["FATF-Monitored"] if is_risky else ["PEP"],
                        "details": f"Flagged entity in {jurisdiction} — {'FATF-monitored jurisdiction' if is_risky else 'possible PEP connection'}.",
                    }
                else:
                    wl[eid] = {"match": False, "lists": [], "details": "No matches"}
            scenario["watchlist_results"] = wl

        # -- Network graph: connect key entities based on transactions --
        if not scenario.get("network_graph"):
            graph: Dict[str, Any] = {}
            for txn in transactions:
                sender = txn.get("customer_id", txn.get("from", ""))
                receiver = txn.get("counterparty", txn.get("to", ""))
                if not sender or not receiver:
                    continue
                if sender not in graph:
                    graph[sender] = {"connections": []}
                if receiver not in graph:
                    graph[receiver] = {"connections": []}
                # Add bidirectional connections (avoid duplicates)
                existing_targets = {c["entity"] for c in graph[sender]["connections"]}
                if receiver not in existing_targets:
                    strength = "strong" if (sender in key_entities or receiver in key_entities) else "weak"
                    graph[sender]["connections"].append({
                        "entity": receiver,
                        "relationship": txn.get("type", "transfer"),
                        "strength": strength,
                    })
                    graph[receiver]["connections"].append({
                        "entity": sender,
                        "relationship": txn.get("type", "transfer"),
                        "strength": strength,
                    })
            scenario["network_graph"] = graph

        # -- Source of funds: flag suspicious transactions --
        if not scenario.get("source_of_funds"):
            sof: Dict[str, Any] = {}
            for txn in transactions:
                tid = txn.get("transaction_id", "")
                sender = txn.get("customer_id", txn.get("from", ""))
                amount = txn.get("amount", 0)
                if sender in key_entities and amount > 5000:
                    sof[tid] = {
                        "source": profiles.get(sender, {}).get("name", sender),
                        "documentation": "Unverified" if sender in key_entities else "Verified",
                        "risk_flags": ["No independent verification"] if sender in key_entities else [],
                    }
            scenario["source_of_funds"] = sof

        # -- Beneficial ownership: create chain for key entities --
        if not scenario.get("beneficial_ownership"):
            bo: Dict[str, Any] = {}
            key_list = list(key_entities)
            for i, eid in enumerate(key_list):
                prof = profiles.get(eid, {})
                chain = [{
                    "entity_id": eid,
                    "entity_name": prof.get("name", eid),
                    "hop_count": 0,
                    "ownership_pct": 100.0,
                    "is_ubo": (i == 0),  # First key entity is the UBO
                }]
                # Add indirect links
                for j, other_eid in enumerate(key_list):
                    if other_eid != eid:
                        other_prof = profiles.get(other_eid, {})
                        chain.append({
                            "entity_id": other_eid,
                            "entity_name": other_prof.get("name", other_eid),
                            "hop_count": j + 1,
                            "ownership_pct": round(100.0 / (j + 2), 1),
                            "is_ubo": False,
                        })
                bo[eid] = chain
            scenario["beneficial_ownership"] = bo

        return scenario

    def _generate_via_llm(
        self, typology: str, difficulty: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to generate a scenario via the LLM API."""
        if not self.api_key:
            return None

        try:
            import httpx
        except ImportError:
            # If httpx not available, try with urllib
            return self._generate_via_llm_urllib(typology, difficulty)

        prompt = (
            f"Generate a {difficulty.upper()} difficulty {typology.replace('_', ' ')} "
            f"money laundering scenario.\n\n{_SCENARIO_SCHEMA}"
        )

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _ADVERSARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"[AdversaryAgent] LLM generation failed: {e}")
            return None

    def _generate_via_llm_urllib(
        self, typology: str, difficulty: str,
    ) -> Optional[Dict[str, Any]]:
        """Fallback LLM call using urllib (no extra deps)."""
        import urllib.request
        import urllib.error

        prompt = (
            f"Generate a {difficulty.upper()} difficulty {typology.replace('_', ' ')} "
            f"money laundering scenario.\n\n{_SCENARIO_SCHEMA}"
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _ADVERSARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            print(f"[AdversaryAgent] LLM generation (urllib) failed: {e}")
            return None

    def _generate_via_local_llm(
        self, typology: str, difficulty: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to generate a scenario via a local Unsloth model."""
        try:
            from unsloth import FastLanguageModel
            import torch
        except ImportError:
            print("[AdversaryAgent] Local inference requires 'unsloth'. Fallback to procedural.")
            return None

        if self.local_model is None:
            print(f"[AdversaryAgent] Loading local model: {self.model}")
            try:
                self.local_model, self.local_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(self.local_model)
            except Exception as e:
                print(f"[AdversaryAgent] Failed to load local model {self.model}: {e}")
                return None

        prompt = (
            f"Generate a {difficulty.upper()} difficulty {typology.replace('_', ' ')} "
            f"money laundering scenario.\n\n{_SCENARIO_SCHEMA}"
        )

        messages = [
            {"role": "system", "content": _ADVERSARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        input_ids = self.local_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        try:
            outputs = self.local_model.generate(
                input_ids=input_ids,
                max_new_tokens=4096,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )
            
            response = self.local_tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            
            # Extract JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print("[AdversaryAgent] Local LLM failed to output valid JSON.")
                return None
                
        except Exception as e:
            print(f"[AdversaryAgent] Local LLM generation failed: {e}")
            return None

    def _normalize_to_scenario(
        self, raw: Dict[str, Any], typology: str, difficulty: str,
    ) -> Dict[str, Any]:
        """Convert LLM output to GeneratedScenario-compatible dict."""
        epoch_start = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))

        # Build customer profiles from entities
        profiles: Dict[str, Any] = {}
        for ent in raw.get("entities", []):
            eid = ent.get("id", _uid("ENT"))
            profiles[eid] = {
                "customer_id": eid,
                "name": ent.get("name", _random_name()),
                "type": ent.get("type", "individual"),
                "jurisdiction": ent.get("jurisdiction", random.choice(_JURISDICTIONS_CLEAN)),
                "pep_status": ent.get("pep", False),
                "occupation": ent.get("occupation"),
                "notes": f"Role: {ent.get('role', 'unknown')}",
            }

        # Build transactions
        transactions: List[Dict[str, Any]] = []
        for i, txn in enumerate(raw.get("transactions", [])):
            tid = f"TXN-ADV-{i+1:03d}"
            transactions.append({
                "transaction_id": tid,
                "customer_id": txn.get("from", ""),
                "date": txn.get("date", _random_date(epoch_start, epoch_start + timedelta(days=60))),
                "type": txn.get("type", "wire_transfer"),
                "amount": txn.get("amount", 0),
                "currency": txn.get("currency", "USD"),
                "description": txn.get("description", ""),
                "counterparty": txn.get("to", ""),
                "is_suspicious": txn.get("is_suspicious", False),
            })

        # Build alert
        alert_id = f"ALERT-ADV-{random.randint(1000, 9999)}"
        alert = {
            "alert_id": alert_id,
            "alert_date": (epoch_start + timedelta(days=5)).strftime("%Y-%m-%d"),
            "alert_type": f"Adversarial {typology.replace('_', ' ').title()} Pattern",
            "risk_score": random.randint(75, 95),
            "priority": "HIGH",
            "customer_id": raw.get("entities", [{}])[0].get("id", "UNKNOWN"),
            "summary": f"Adversary-generated {typology} scenario (difficulty: {difficulty})",
        }

        # Device fingerprints from overlaps
        device_fingerprints: Dict[str, Any] = {}
        for overlap in raw.get("device_overlaps", []):
            shared_dev = overlap.get("shared_device_id", _random_device_id())
            shared_ip = overlap.get("shared_ip", _random_ip(clean=False))
            for eid in overlap.get("entities", []):
                if eid not in device_fingerprints:
                    device_fingerprints[eid] = []
                device_fingerprints[eid].append({
                    "device_id": shared_dev,
                    "ip_address": shared_ip,
                    "mac_address": _random_mac(),
                    "latitude": _random_coords()[0],
                    "longitude": _random_coords()[1],
                    "jurisdiction": profiles.get(eid, {}).get("jurisdiction", "Unknown"),
                    "entity_id": eid,
                })

        # Customs invoices from phantom invoices
        customs_invoices: Dict[str, Any] = {}
        for inv in raw.get("phantom_invoices", []):
            inv_id = inv.get("invoice_id", f"INV-{_uid('', 4)}")
            customs_invoices[inv_id] = {
                "invoice_id": inv_id,
                "transaction_id": transactions[0]["transaction_id"] if transactions else "",
                "hs_code": _HS_CODES.get("generic_goods", "9999.99"),
                "commodity_description": inv.get("commodity", "Generic Goods"),
                "declared_value_usd": inv.get("declared_value", 0),
                "shipping_weight_kg": random.uniform(100, 5000),
                "origin_country": inv.get("origin", "Unknown"),
                "destination_country": inv.get("destination", "Unknown"),
                "is_phantom": True,
            }

        # Ground truth
        gt = raw.get("ground_truth", {})
        ground_truth = {
            "correct_decision": gt.get("correct_decision", "file_sar"),
            "typology": gt.get("typology", typology),
            "key_entities": gt.get("key_entities", []),
            "excluded_entities": gt.get("excluded_entities", []),
            "key_findings": gt.get("key_findings", []),
            "red_flags": gt.get("red_flags", []),
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": {},
            "network_graph": {},
            "source_of_funds": {},
            "ground_truth": ground_truth,
            "market_data": {},
            "device_fingerprints": device_fingerprints,
            "customs_invoices": customs_invoices,
            "beneficial_ownership": {},
            "_meta": {
                "difficulty": difficulty,
                "typology": typology,
                "generator": "adversary_agent",
                "model": self.model,
                "evasion_techniques": raw.get("evasion_techniques", []),
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def _generate_fallback(
        self, typology: str, difficulty: str,
    ) -> Dict[str, Any]:
        """Procedural fallback when LLM is unavailable.

        Generates a deterministic adversarial scenario using enhanced
        procedural techniques with additional evasion layers.
        """
        epoch_start = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
        epoch_end = epoch_start + timedelta(days=random.randint(30, 90))

        if typology == "mule_ring":
            return self._fallback_mule_ring(difficulty, epoch_start, epoch_end)
        elif typology == "pass_through":
            return self._fallback_pass_through(difficulty, epoch_start, epoch_end)
        elif typology == "phantom_invoice":
            return self._fallback_phantom_invoice(difficulty, epoch_start, epoch_end)
        else:
            return self._fallback_mule_ring(difficulty, epoch_start, epoch_end)

    def _fallback_mule_ring(
        self, difficulty: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Procedural mule ring scenario with shared devices."""
        num_mules = 3 if difficulty == "easy" else 5 if difficulty == "medium" else 7
        shared_device = _random_device_id("MOB")
        shared_ip = _random_ip(clean=False)
        controller_id = _uid("ENT")
        controller_name = _random_name()

        profiles: Dict[str, Any] = {
            controller_id: {
                "customer_id": controller_id,
                "name": controller_name,
                "type": "individual",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN),
                "notes": "Ring controller",
            }
        }

        transactions = []
        device_fps: Dict[str, Any] = {}
        mule_ids = []

        for i in range(num_mules):
            mid = _uid("MULE")
            mname = _random_name()
            mule_ids.append(mid)
            profiles[mid] = {
                "customer_id": mid,
                "name": mname,
                "type": "individual",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN),
                "account_age_days": random.randint(14, 60),
                "notes": f"Account opened recently. Linked to ring via device {shared_device}.",
            }
            device_fps[mid] = [{
                "device_id": shared_device,
                "ip_address": shared_ip,
                "mac_address": _random_mac(),
                "latitude": _random_coords()[0],
                "longitude": _random_coords()[1],
                "jurisdiction": profiles[mid]["jurisdiction"],
                "entity_id": mid,
            }]

            # Cascading transfer: controller → mule → next mule
            amt = round(random.uniform(3000, 9500), 2)
            txn_date = _random_date(epoch_start, epoch_end)
            transactions.append({
                "transaction_id": f"TXN-MULE-{i+1:03d}",
                "customer_id": controller_id if i == 0 else mule_ids[i - 1],
                "counterparty": mid,
                "date": txn_date,
                "type": "wire_transfer",
                "amount": amt,
                "currency": "USD",
                "description": random.choice(_WIRE_DESCRIPTIONS_SUSPECT),
            })

        # Add decoy legitimate entities
        for _ in range(2):
            did = _uid("DEC")
            profiles[did] = {
                "customer_id": did,
                "name": _random_name(),
                "type": "individual",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN),
                "notes": "Legitimate customer",
            }
            transactions.append({
                "transaction_id": f"TXN-DEC-{_uid('', 3)}",
                "customer_id": did,
                "date": _random_date(epoch_start, epoch_end),
                "type": "wire_transfer",
                "amount": round(random.uniform(500, 5000), 2),
                "currency": "USD",
                "description": "Monthly payroll",
                "counterparty": None,
            })

        alert = {
            "alert_id": f"ALERT-ADV-{random.randint(1000, 9999)}",
            "alert_date": (epoch_start + timedelta(days=5)).strftime("%Y-%m-%d"),
            "alert_type": "Mule Ring — Shared Device Fingerprint Cluster",
            "risk_score": random.randint(80, 95),
            "priority": "HIGH",
            "customer_id": controller_id,
            "summary": f"Cluster of {num_mules} accounts sharing device {shared_device}.",
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": {},
            "network_graph": {},
            "source_of_funds": {},
            "ground_truth": {
                "correct_decision": "file_sar",
                "typology": "mule_ring",
                "key_entities": [controller_id] + mule_ids,
                "excluded_entities": [k for k in profiles if k.startswith("DEC")],
                "key_findings": ["shared_device_fingerprint", "cascading_transfers", "new_account_cluster"],
                "red_flags": [f"Shared device: {shared_device}", f"Shared VPN IP: {shared_ip}",
                              f"{num_mules} mule accounts opened within 60 days"],
            },
            "market_data": {},
            "device_fingerprints": device_fps,
            "customs_invoices": {},
            "beneficial_ownership": {},
            "_meta": {
                "difficulty": difficulty,
                "typology": "mule_ring",
                "generator": "adversary_agent_fallback",
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def _fallback_pass_through(
        self, difficulty: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Procedural pass-through / circular fund flow scenario."""
        num_shells = 3 if difficulty == "easy" else 5 if difficulty == "medium" else 7
        shared_address = random.choice(_ADDRESSES)

        profiles: Dict[str, Any] = {}
        shell_ids = []
        transactions = []

        for i in range(num_shells):
            sid = _uid("SHELL")
            shell_ids.append(sid)
            profiles[sid] = {
                "customer_id": sid,
                "name": _random_company(),
                "type": "company" if i > 0 else "shell",
                "jurisdiction": random.choice(_JURISDICTIONS_RISKY),
                "registered_address": shared_address if i < 2 else random.choice(_ADDRESSES),
                "beneficial_owners": "Unknown",
                "notes": "Shell entity, no public business activity.",
            }

        # Circular flow: shell_0 → shell_1 → ... → shell_n → shell_0
        base_amount = random.choice([150_000, 250_000, 500_000])
        for i in range(num_shells):
            src = shell_ids[i]
            tgt = shell_ids[(i + 1) % num_shells]
            amt = round(base_amount * random.uniform(0.8, 1.1), 2)
            txn_date = (epoch_start + timedelta(days=i * 2)).strftime("%Y-%m-%d")
            transactions.append({
                "transaction_id": f"TXN-PT-{i+1:03d}",
                "customer_id": src,
                "counterparty": tgt,
                "date": txn_date,
                "type": "wire_transfer",
                "amount": amt,
                "currency": "USD",
                "description": random.choice(_WIRE_DESCRIPTIONS_SUSPECT),
            })

        alert = {
            "alert_id": f"ALERT-ADV-{random.randint(1000, 9999)}",
            "alert_date": (epoch_start + timedelta(days=num_shells * 2 + 3)).strftime("%Y-%m-%d"),
            "alert_type": "Circular Fund Flow — Pass-Through Pattern",
            "risk_score": random.randint(85, 98),
            "priority": "HIGH",
            "customer_id": shell_ids[0],
            "summary": f"Circular fund flow of ~${base_amount:,.0f} detected across {num_shells} entities.",
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": {},
            "network_graph": {},
            "source_of_funds": {},
            "ground_truth": {
                "correct_decision": "file_sar",
                "typology": "pass_through",
                "key_entities": shell_ids,
                "excluded_entities": [],
                "key_findings": ["circular_fund_flow", "shared_registered_address", "shell_company_cluster"],
                "red_flags": [
                    f"Circular flow of ~${base_amount:,.0f} across {num_shells} entities",
                    f"Shared registered address: {shared_address}",
                    "No legitimate business activity for any entity",
                ],
            },
            "market_data": {},
            "device_fingerprints": {},
            "customs_invoices": {},
            "beneficial_ownership": {},
            "_meta": {
                "difficulty": difficulty,
                "typology": "pass_through",
                "generator": "adversary_agent_fallback",
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def _fallback_phantom_invoice(
        self, difficulty: str, epoch_start: datetime, epoch_end: datetime,
    ) -> Dict[str, Any]:
        """Procedural phantom invoice (TBML) scenario."""
        exporter_id = _uid("EXP")
        importer_id = _uid("IMP")
        exporter_name = _random_company()
        importer_name = _random_company()

        profiles = {
            exporter_id: {
                "customer_id": exporter_id,
                "name": exporter_name,
                "type": "company",
                "jurisdiction": random.choice(_JURISDICTIONS_RISKY),
                "notes": "Exporter — no verifiable trade history.",
            },
            importer_id: {
                "customer_id": importer_id,
                "name": importer_name,
                "type": "company",
                "jurisdiction": random.choice(_JURISDICTIONS_CLEAN),
                "notes": "Importer — recently incorporated.",
            },
        }

        commodity = random.choice(_COMMODITIES)
        num_invoices = 2 if difficulty == "easy" else 4 if difficulty == "medium" else 6
        transactions = []
        customs = {}

        for i in range(num_invoices):
            tid = f"TXN-TBML-{i+1:03d}"
            declared = round(commodity[2] * random.uniform(3, 8), 2)
            actual_weight = round(random.uniform(10, 50), 1)  # Suspiciously low weight

            transactions.append({
                "transaction_id": tid,
                "customer_id": importer_id,
                "counterparty": exporter_id,
                "date": (epoch_start + timedelta(days=i * 7)).strftime("%Y-%m-%d"),
                "type": "trade_payment",
                "amount": declared,
                "currency": "USD",
                "description": f"Payment for {commodity[1]}",
            })

            inv_id = f"INV-{_uid('', 4)}"
            customs[inv_id] = {
                "invoice_id": inv_id,
                "transaction_id": tid,
                "hs_code": _HS_CODES.get(commodity[0], "9999.99"),
                "commodity_description": commodity[1],
                "declared_value_usd": declared,
                "shipping_weight_kg": actual_weight,
                "origin_country": profiles[exporter_id]["jurisdiction"],
                "destination_country": profiles[importer_id]["jurisdiction"],
                "shipper_name": exporter_name,
                "consignee_name": importer_name,
                "bill_of_lading": None,  # Missing — phantom indicator
                "is_phantom": True,
            }

        alert = {
            "alert_id": f"ALERT-ADV-{random.randint(1000, 9999)}",
            "alert_date": (epoch_start + timedelta(days=num_invoices * 7 + 5)).strftime("%Y-%m-%d"),
            "alert_type": "Phantom Invoice — Trade-Based ML",
            "risk_score": random.randint(82, 96),
            "priority": "HIGH",
            "customer_id": importer_id,
            "summary": f"{num_invoices} invoices with no bill of lading. Declared values inconsistent with weights.",
        }

        return {
            "initial_alert": alert,
            "customer_profiles": profiles,
            "transactions": transactions,
            "watchlist_results": {},
            "network_graph": {},
            "source_of_funds": {},
            "ground_truth": {
                "correct_decision": "file_sar",
                "typology": "trade_based_ml",
                "key_entities": [exporter_id, importer_id],
                "excluded_entities": [],
                "key_findings": ["phantom_invoices", "no_bill_of_lading", "value_weight_mismatch"],
                "red_flags": [
                    f"{num_invoices} invoices with no bill of lading",
                    "Declared values inconsistent with shipping weights",
                    f"Exporter in {profiles[exporter_id]['jurisdiction']} — no trade history",
                ],
            },
            "market_data": {},
            "device_fingerprints": {},
            "customs_invoices": customs,
            "beneficial_ownership": {},
            "_meta": {
                "difficulty": difficulty,
                "typology": "phantom_invoice",
                "generator": "adversary_agent_fallback",
                "generated_at": datetime.utcnow().isoformat(),
            },
        }
