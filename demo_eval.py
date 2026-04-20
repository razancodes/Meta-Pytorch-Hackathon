#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — Demo Evaluation (Stage Presentation).

Runs a trained (or base) LLM agent through a hardcoded AML case inspired
by the 1MDB sovereign wealth fund scandal. Captures and saves full AGUI
state payloads per step so the Next.js frontend can replay the simulation.

Output:
  demo_output/
  ├── episode_meta.json         # scenario summary, model info, final score
  ├── step_001.json             # AGUI state + action + observation per step
  ├── step_002.json
  └── ...

Usage:
  python demo_eval.py                                # base model
  python demo_eval.py --model checkpoints/best       # trained checkpoint
  python demo_eval.py --dry-run                      # no model, scripted actions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction, AMLObservation, AMLState
from scenarios.procedural_generator import GeneratedScenario
from server.aml_environment import AMLEnvironment
from state_manager import StateManager


# ═══════════════════════════════════════════════════════════════════════
# Static 1MDB-Inspired Scenario
# ═══════════════════════════════════════════════════════════════════════

def build_1mdb_scenario() -> GeneratedScenario:
    """Construct a hardcoded AML scenario inspired by the 1MDB scandal.

    Entities:
      - Taek Jho Lowe          (primary subject, orchestrator)
      - Golden Star Holdings   (Seychelles shell)
      - Arabella Investments   (BVI shell)
      - PetraStar Energy Fund  (sovereign wealth fund)
      - Sarah Chen             (clean decoy customer)
    """

    subject_id = "CUST-1MDB-001"
    shell_a = "ENT-GSTAR-001"
    shell_b = "ENT-ARBL-001"
    fund_id = "ENT-PETRA-001"
    decoy_id = "CUST-CHEN-002"

    alert = {
        "alert_id": "ALERT-2024-1MDB-7701",
        "customer_id": subject_id,
        "summary": (
            "Suspicious international wire transfers totaling $681M from "
            "PetraStar Energy Fund through offshore shell entities. "
            "Multiple jurisdictions flagged: Malaysia, Seychelles, BVI, Singapore."
        ),
        "type": "Suspicious Wire Transfer",
        "risk_score": 92,
        "date": "2024-03-15",
    }

    profiles = {
        subject_id: {
            "customer_id": subject_id,
            "name": "Taek Jho Lowe",
            "nationality": "Malaysian",
            "occupation": "Investment Consultant",
            "account_open_date": "2023-01-10",
            "risk_rating": "High",
            "pep_status": True,
            "pep_details": "Connected to Minister of Finance (MY)",
            "address": "15 Jalan Ampang, Kuala Lumpur, Malaysia",
            "annual_income": "$120,000",
            "notes": "Multiple passports. Frequent travel to Seychelles, Switzerland.",
        },
        shell_a: {
            "customer_id": shell_a,
            "name": "Golden Star Holdings Ltd",
            "nationality": "Seychelles",
            "occupation": "Investment Holding Company",
            "account_open_date": "2022-06-01",
            "risk_rating": "High",
            "registration": "Seychelles IBC, incorporated 2022-05-14",
            "directors": ["Taek Jho Lowe", "Nominee Director Services (SC) Ltd"],
            "registered_address": "Suite 4, Mahe Business Centre, Victoria, Seychelles",
            "notes": "No employees. No commercial operations. Sole purpose appears to be fund transfers.",
        },
        shell_b: {
            "customer_id": shell_b,
            "name": "Arabella Investments PJS Ltd",
            "nationality": "British Virgin Islands",
            "occupation": "Private Equity Vehicle",
            "account_open_date": "2022-09-20",
            "risk_rating": "High",
            "registration": "BVI Business Company, incorporated 2022-08-30",
            "directors": ["Nominee Director (BVI) Corp"],
            "registered_address": "Suite 4, Mahe Business Centre, Victoria, Seychelles",
            "notes": "SHARES REGISTERED ADDRESS with Golden Star Holdings. Beneficial owner undisclosed.",
        },
        fund_id: {
            "customer_id": fund_id,
            "name": "PetraStar Energy Fund",
            "nationality": "Malaysia",
            "occupation": "Sovereign Wealth Fund",
            "risk_rating": "Medium",
            "notes": "Government-linked investment fund. Board includes Minister of Finance.",
        },
        decoy_id: {
            "customer_id": decoy_id,
            "name": "Sarah Chen",
            "nationality": "Singaporean",
            "occupation": "Software Engineer",
            "account_open_date": "2021-03-15",
            "risk_rating": "Low",
            "annual_income": "$95,000",
            "notes": "Legitimate retail customer. Regular salary deposits.",
        },
    }

    transactions = [
        # The ML chain
        {"transaction_id": "TXN-1MDB-001", "from": fund_id, "to": shell_a,
         "amount": 681000000, "currency": "USD", "date": "2024-01-15",
         "type": "International Wire", "description": "Investment partnership contribution",
         "jurisdiction_from": "Malaysia", "jurisdiction_to": "Seychelles"},
        {"transaction_id": "TXN-1MDB-002", "from": shell_a, "to": shell_b,
         "amount": 260000000, "currency": "USD", "date": "2024-01-22",
         "type": "International Wire", "description": "Intercompany transfer",
         "jurisdiction_from": "Seychelles", "jurisdiction_to": "BVI"},
        {"transaction_id": "TXN-1MDB-003", "from": shell_b, "to": "Real Estate Holdings (NY)",
         "amount": 250000000, "currency": "USD", "date": "2024-02-05",
         "type": "International Wire", "description": "Real estate acquisition",
         "jurisdiction_from": "BVI", "jurisdiction_to": "United States"},
        {"transaction_id": "TXN-1MDB-004", "from": shell_b, "to": "Art dealer (Geneva)",
         "amount": 135000000, "currency": "USD", "date": "2024-02-14",
         "type": "International Wire", "description": "Art collection purchase",
         "jurisdiction_from": "BVI", "jurisdiction_to": "Switzerland"},
        {"transaction_id": "TXN-1MDB-005", "from": shell_a, "to": subject_id,
         "amount": 30000000, "currency": "USD", "date": "2024-02-28",
         "type": "Wire Transfer", "description": "Consulting fees",
         "jurisdiction_from": "Seychelles", "jurisdiction_to": "Malaysia"},
        # Reversal (cover-up attempt)
        {"transaction_id": "TXN-1MDB-006", "from": shell_b, "to": fund_id,
         "amount": 6000000, "currency": "USD", "date": "2024-03-01",
         "type": "Wire Transfer", "description": "Return of overpayment",
         "jurisdiction_from": "BVI", "jurisdiction_to": "Malaysia"},
        # Decoy noise — legitimate transactions
        {"transaction_id": "TXN-LEGIT-001", "from": "Employer (TechCorp SG)", "to": decoy_id,
         "amount": 7900, "currency": "SGD", "date": "2024-01-31",
         "type": "Salary", "description": "Monthly salary"},
        {"transaction_id": "TXN-LEGIT-002", "from": decoy_id, "to": "DBS Savings",
         "amount": 3000, "currency": "SGD", "date": "2024-02-01",
         "type": "Transfer", "description": "Savings transfer"},
    ]

    watchlist = {
        subject_id: {"match": True, "lists": ["PEP"], "details": "Politically Exposed Person — connected to Malaysian government officials"},
        shell_a: {"match": True, "lists": ["FATF-Monitored"], "details": "Seychelles entity — FATF-monitored jurisdiction (grey list)"},
        shell_b: {"match": False, "lists": [], "details": "No direct watchlist match"},
        decoy_id: {"match": False, "lists": [], "details": "No matches"},
    }

    network = {
        subject_id: {
            "connections": [
                {"entity": shell_a, "relationship": "Director", "strength": "strong"},
                {"entity": fund_id, "relationship": "Consultant", "strength": "moderate"},
            ]
        },
        shell_a: {
            "connections": [
                {"entity": shell_b, "relationship": "Shared registered address", "strength": "strong"},
                {"entity": subject_id, "relationship": "Director", "strength": "strong"},
            ]
        },
        shell_b: {
            "connections": [
                {"entity": shell_a, "relationship": "Shared registered address", "strength": "strong"},
            ]
        },
    }

    source_of_funds = {
        "TXN-1MDB-001": {
            "source": "PetraStar Energy Fund (Sovereign Wealth)",
            "documentation": "Investment partnership agreement (unverified)",
            "risk_flags": ["Sovereign fund disbursement without board resolution", "No independent audit"],
        },
        "TXN-1MDB-005": {
            "source": "Golden Star Holdings Ltd",
            "documentation": "Consulting agreement (backdated)",
            "risk_flags": ["Backdated contract", "No deliverables documented"],
        },
    }

    ground_truth = {
        "correct_decision": "file_sar",
        "typology": "layering",
        "key_entities": [subject_id, shell_a, shell_b],
        "excluded_entities": [decoy_id],
        "key_findings": [
            "pep_connection",
            "offshore_source",
            "shared_registered_address",
            "rapid_fan_out",
            "no_source_documentation",
            "reversed_transaction",
        ],
    }

    market_data = {}

    return GeneratedScenario({
        "initial_alert": alert,
        "customer_profiles": profiles,
        "transactions": transactions,
        "watchlist_results": watchlist,
        "network_graph": network,
        "source_of_funds": source_of_funds,
        "ground_truth": ground_truth,
        "market_data": market_data,
    })


# ═══════════════════════════════════════════════════════════════════════
# Demo Environment (injects static scenario)
# ═══════════════════════════════════════════════════════════════════════

class DemoEnvironment(AMLEnvironment):
    """AMLEnvironment variant that uses a pre-built static scenario."""

    def __init__(self, scenario: GeneratedScenario):
        super().__init__()
        self._injected_scenario = scenario

    def reset(self, **kwargs):
        """Override reset to use the injected scenario."""
        import uuid as _uuid
        ep_id = str(_uuid.uuid4())

        self._current_scenario = self._injected_scenario
        self._state = AMLState(episode_id=ep_id, step_count=0, task_id="demo_1mdb")
        self._sm = StateManager()

        alert = self._current_scenario.initial_alert
        alert_summary = (
            f"Alert {alert['alert_id']}: {alert.get('summary', '')} "
            f"Customer: {alert.get('customer_id', 'N/A')}"
        )
        self._sm.push_observation(alert_summary)
        self._sm.sync_to_state(self._state)

        return AMLObservation(
            tool_result={"alert": alert},
            available_tools=self._get_tools(),
            message=(
                f"[1MDB DEMO] Alert {alert['alert_id']} assigned. "
                f"Subject: {alert['customer_id']}. Investigate and decide."
            ),
            done=False,
            reward=None,
            metadata={
                "episode_id": ep_id, "task_id": "demo_1mdb", "step": 0,
                "agui_state": self._sm.build_agui_state(),
            },
        )

    def _get_tools(self):
        from server.aml_environment import AVAILABLE_TOOLS
        return AVAILABLE_TOOLS


# ═══════════════════════════════════════════════════════════════════════
# AGUI Replay Recorder
# ═══════════════════════════════════════════════════════════════════════

class AGUIRecorder:
    """Records AGUI state payloads for frontend replay."""

    def __init__(self, output_dir: str = "demo_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.steps: List[Dict[str, Any]] = []

    def record_step(
        self,
        step_num: int,
        action: Dict[str, Any],
        observation: AMLObservation,
        reasoning: str = "",
    ) -> None:
        """Capture one step for replay."""
        agui = observation.metadata.get("agui_state", {})

        record = {
            "step_number": step_num,
            "timestamp": time.time(),
            "action": action,
            "reasoning": reasoning,
            "observation": {
                "tool_result": observation.tool_result,
                "message": observation.message,
                "done": observation.done,
                "reward": observation.reward,
            },
            "agui_state": agui,
        }
        self.steps.append(record)

        # Save individual step file
        path = os.path.join(self.output_dir, f"step_{step_num:03d}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    def save_meta(self, scenario: GeneratedScenario, final_score: float, model_name: str):
        """Save episode metadata."""
        meta = {
            "scenario": "1MDB-Inspired Sovereign Wealth Fund Investigation",
            "model": model_name,
            "total_steps": len(self.steps),
            "final_score": final_score,
            "ground_truth": scenario.ground_truth,
            "alert": scenario.initial_alert,
            "entity_count": len(scenario.customer_profiles),
            "transaction_count": len(scenario.transactions),
            "steps_summary": [
                {
                    "step": s["step_number"],
                    "tool": s["action"].get("tool", "?"),
                    "reward": s["observation"]["reward"],
                    "done": s["observation"]["done"],
                }
                for s in self.steps
            ],
        }
        path = os.path.join(self.output_dir, "episode_meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  📁 Saved {len(self.steps)} step files + meta → {self.output_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# Scripted Demo (no model needed)
# ═══════════════════════════════════════════════════════════════════════

SCRIPTED_ACTIONS = [
    ("review_alert", {}, "Begin by reviewing the alert details"),
    ("get_customer_profile", {"customer_id": "CUST-1MDB-001"}, "Profile the primary subject"),
    ("write_to_case_file", {"note": "Subject is PEP, connected to Minister of Finance. High risk."}, "Save PEP finding to disk before RAM eviction"),
    ("query_transactions", {"customer_id": "CUST-1MDB-001"}, "Analyze transaction history"),
    ("search_compliance_manual", {"query": "layering wire transfer offshore"}, "Look up compliance rules"),
    ("update_system_prompt", {"directive": "Apply enhanced due diligence for PEP with offshore transfers"}, "Inject compliance rule into kernel"),
    ("trace_network", {"entity_id": "ENT-GSTAR-001"}, "Trace Golden Star Holdings network"),
    ("write_to_case_file", {"note": "Golden Star and Arabella share registered address in Seychelles. Classic layering."}, "Persist network findings"),
    ("check_watchlist", {"entity": "CUST-1MDB-001"}, "Check subject against sanctions lists"),
    ("check_source_of_funds", {"transaction_id": "TXN-1MDB-001"}, "Verify source of the $681M wire"),
    ("request_wire_trace", {"transaction_id": "TXN-1MDB-002"}, "Request async trace on the $260M intercompany transfer"),
    # Wait 2 steps for async result
    ("get_customer_profile", {"customer_id": "ENT-ARBL-001"}, "Profile the second shell company while waiting"),
    ("check_source_of_funds", {"transaction_id": "TXN-1MDB-005"}, "Check the $30M consulting fee source"),
    # Now retrieve async
    ("retrieve_async_result", {"job_id": "REQ-001"}, "Retrieve the completed wire trace"),
    ("file_sar", {
        "typology": "layering",
        "entities_involved": ["CUST-1MDB-001", "ENT-GSTAR-001", "ENT-ARBL-001"],
        "findings": [
            "pep_connection", "offshore_source", "shared_registered_address",
            "rapid_fan_out", "no_source_documentation", "reversed_transaction",
        ],
    }, "File SAR with complete evidence chain"),
]


# ═══════════════════════════════════════════════════════════════════════
# LLM-Driven Demo
# ═══════════════════════════════════════════════════════════════════════

def run_llm_demo(model_path: str, output_dir: str) -> None:
    """Run the demo using a trained or base LLM."""
    import torch
    from unsloth import FastLanguageModel

    print(f"  Loading model: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=2048, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    # Import prompt formatting and parsing from train_ppo
    from train_ppo import format_prompt, parse_action

    scenario = build_1mdb_scenario()
    env = DemoEnvironment(scenario)
    recorder = AGUIRecorder(output_dir)

    obs = env.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for step_num in range(1, 26):
        ram = list(env._sm.ram_queue) if env._sm else []
        disk = env._sm.disk_contents if env._sm else []
        kernel = env._sm.kernel_directives if env._sm else []

        prompt = format_prompt(obs, step_num, kernel, disk, ram)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1856).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=192, temperature=0.3,
                top_p=0.9, do_sample=True, repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tool, params = parse_action(response)
        reasoning = ""
        try:
            d = json.loads(response.strip())
            reasoning = d.get("reasoning", "")
        except Exception:
            pass

        action = {"tool": tool, "parameters": params}
        obs = env.step(AMLAction(tool=tool, parameters=params))

        recorder.record_step(step_num, action, obs, reasoning)
        reward = obs.reward if obs.reward is not None else 0.0
        print(f"  Step {step_num:>2} | {tool:<30} | R={reward:+.4f} | {'DONE' if obs.done else 'ok'}")
        if reasoning:
            print(f"           └─ {reasoning[:80]}")

        if obs.done:
            break

    final_score = env._state.accumulated_reward
    recorder.save_meta(scenario, final_score, model_path)
    print(f"\n  🏁 FINAL SCORE: {final_score:+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Scripted Demo (no GPU needed)
# ═══════════════════════════════════════════════════════════════════════

def run_scripted_demo(output_dir: str) -> None:
    """Run demo with scripted actions — no model needed."""
    scenario = build_1mdb_scenario()
    env = DemoEnvironment(scenario)
    recorder = AGUIRecorder(output_dir)

    obs = env.reset()
    print(f"  Alert: {obs.tool_result['alert']['alert_id']}")
    print(f"  Subject: {obs.tool_result['alert']['customer_id']}\n")

    for step_num, (tool, params, reasoning) in enumerate(SCRIPTED_ACTIONS, 1):
        action = {"tool": tool, "parameters": params}
        obs = env.step(AMLAction(tool=tool, parameters=params))

        recorder.record_step(step_num, action, obs, reasoning)
        reward = obs.reward if obs.reward is not None else 0.0

        print(f"  Step {step_num:>2} | {tool:<30} | R={reward:+.4f} | {'DONE' if obs.done else 'ok'}")
        print(f"           └─ {reasoning}")

        if obs.done:
            break

    final_score = env._state.accumulated_reward
    recorder.save_meta(scenario, final_score, "scripted")

    print(f"\n{'═'*60}")
    print(f"  🏁 1MDB DEMO COMPLETE  |  FINAL SCORE: {final_score:+.4f}")
    print(f"  📁 Replay data: {output_dir}/")
    print(f"     {len(recorder.steps)} step files + episode_meta.json")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex Demo Evaluation — 1MDB Case")
    p.add_argument("--model", type=str, default=None,
                   help="Path to trained checkpoint (omit for scripted demo)")
    p.add_argument("--output-dir", default="demo_output")
    p.add_argument("--dry-run", action="store_true",
                   help="Run scripted demo without model")
    args = p.parse_args()

    print(f"\n{'═'*60}")
    print(f"  MEMEX DEMO — 1MDB Sovereign Wealth Fund Investigation")
    print(f"{'═'*60}\n")

    if args.dry_run or args.model is None:
        run_scripted_demo(args.output_dir)
    else:
        run_llm_demo(args.model, args.output_dir)


if __name__ == "__main__":
    main()
