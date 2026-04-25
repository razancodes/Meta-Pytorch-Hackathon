#!/usr/bin/env python3
"""
Memex Evaluation Harness — Multi-Typology Benchmark Suite.

Runs scripted evaluation scenarios across all supported typologies and
collects per-scenario metrics. Used to compare checkpoints, verify
grader consistency, and measure OS-mechanic proficiency.

Scenarios:
  1. 1MDB Layering (existing demo)
  2. FinCEN Mule Ring (device overlap, shared IPs)
  3. Phantom Invoice TBML (customs, market data)
  4. Shell Company Passthrough (BO queries, network)
  5. Cash-Intensive Structuring (transaction patterns)
  6. Sanctions Evasion (watchlist, PEP, compliance)

Usage:
  python eval_harness.py                    # Run all 6 scenarios (dry-run)
  python eval_harness.py --scenario mule_ring
  python eval_harness.py --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction
from scenarios.procedural_generator import ScenarioGenerator, GeneratedScenario
from server.aml_environment import AMLEnvironment
from state_manager import StateManager


# ═══════════════════════════════════════════════════════════════════════
# Scenario Definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvalScenario:
    """A scripted evaluation scenario with actions and expected outcomes."""
    name: str
    typology: str
    difficulty: str
    description: str
    actions: List[Dict[str, Any]]
    expected_typology: str
    expected_entities: List[str]


def _build_1mdb_scenario() -> EvalScenario:
    """1MDB Layering — the original demo scenario."""
    return EvalScenario(
        name="1mdb_layering",
        typology="layering",
        difficulty="hard",
        description="$681M sovereign wealth fund diversion through offshore shells",
        expected_typology="layering",
        expected_entities=["CUST-1MDB-001", "ENT-GSTAR-001", "ENT-ARBL-001"],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-1MDB-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Subject is PEP, connected to Minister of Finance. High risk."}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-1MDB-001"}},
            {"tool": "search_compliance_manual", "parameters": {"query": "layering wire transfer offshore"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Apply enhanced due diligence for PEP with offshore transfers"}},
            {"tool": "trace_network", "parameters": {"entity_id": "ENT-GSTAR-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Golden Star and Arabella share registered address. Classic layering."}},
            {"tool": "check_watchlist", "parameters": {"entity": "CUST-1MDB-001"}},
            {"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-1MDB-001"}},
            {"tool": "request_wire_trace", "parameters": {"transaction_id": "TXN-1MDB-002"}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "ENT-ARBL-001"}},
            {"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-1MDB-005"}},
            {"tool": "retrieve_async_result", "parameters": {"job_id": "__LAST_JOB_ID__"}},
            {"tool": "file_sar", "parameters": {
                "typology": "layering",
                "entities_involved": ["CUST-1MDB-001", "ENT-GSTAR-001", "ENT-ARBL-001"],
                "findings": ["pep_connection", "offshore_source", "shared_registered_address",
                             "rapid_fan_out", "no_source_documentation"],
                "ubo_identified": "CUST-1MDB-001",
                "evidence_chain": "Multi-layer offshore diversion via Golden Star and Arabella shells",
            }},
        ],
    )


def _build_mule_ring_scenario() -> EvalScenario:
    """FinCEN Mule Ring — device overlap and shared IP detection."""
    return EvalScenario(
        name="mule_ring",
        typology="mule_ring",
        difficulty="hard",
        description="Ring of mule accounts sharing devices, rapid round-trip transfers",
        expected_typology="mule_ring",
        expected_entities=[],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Rapid in/out pattern detected. Checking device overlap."}},
            {"tool": "check_device_overlap", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "trace_network", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "search_compliance_manual", "parameters": {"query": "mule ring device sharing rapid transfers"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Flag accounts sharing device fingerprints as potential mules"}},
            {"tool": "check_watchlist", "parameters": {"entity": "CUST-001"}},
            {"tool": "assess_risk", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "file_sar", "parameters": {
                "typology": "mule_ring",
                "entities_involved": ["CUST-001"],
                "findings": ["device_sharing", "rapid_round_trips", "new_account"],
                "evidence_chain": "Device fingerprint overlap + rapid in/out transfer pattern",
            }},
        ],
    )


def _build_phantom_invoice_scenario() -> EvalScenario:
    """Trade-based ML with phantom invoices."""
    return EvalScenario(
        name="phantom_invoice",
        typology="trade_based_ml",
        difficulty="hard",
        description="Over-invoiced goods with no bill of lading, price manipulation",
        expected_typology="trade_based_ml",
        expected_entities=[],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "verify_customs_invoice", "parameters": {"invoice_id": "INV-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Missing bill of lading on $2M electronics shipment."}},
            {"tool": "check_market_price", "parameters": {"commodity": "electronics", "declared_value": 2000000}},
            {"tool": "search_compliance_manual", "parameters": {"query": "trade based money laundering phantom invoice"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Flag invoices without BoL and price >200% market rate"}},
            {"tool": "query_beneficial_ownership", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "trace_network", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "file_sar", "parameters": {
                "typology": "trade_based_ml",
                "entities_involved": ["CUST-001"],
                "findings": ["phantom_invoice", "no_bill_of_lading", "over_invoicing"],
                "evidence_chain": "Phantom invoice with no BoL, price 3x market rate",
            }},
        ],
    )


def _build_pass_through_scenario() -> EvalScenario:
    """Shell company passthrough with circular flows."""
    return EvalScenario(
        name="pass_through",
        typology="pass_through",
        difficulty="medium",
        description="Funds routed through shell companies in a circular pattern",
        expected_typology="pass_through",
        expected_entities=[],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "trace_network", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Circular flow detected: A -> B -> C -> A."}},
            {"tool": "query_beneficial_ownership", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-001"}},
            {"tool": "search_compliance_manual", "parameters": {"query": "shell company pass through circular"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Flag circular fund flows through entities sharing UBOs"}},
            {"tool": "assess_risk", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "file_sar", "parameters": {
                "typology": "pass_through",
                "entities_involved": ["CUST-001"],
                "findings": ["circular_flow", "shell_company", "shared_ubo"],
                "evidence_chain": "Circular fund movement through shells with same UBO",
            }},
        ],
    )


def _build_structuring_scenario() -> EvalScenario:
    """Cash-intensive structuring under CTR threshold."""
    return EvalScenario(
        name="structuring",
        typology="structuring",
        difficulty="easy",
        description="Sub-$10K cash deposits across multiple branches",
        expected_typology="structuring",
        expected_entities=[],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Multiple sub-$10K cash deposits across 3 branches in 2 days."}},
            {"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-001"}},
            {"tool": "search_compliance_manual", "parameters": {"query": "structuring cash deposits under CTR threshold"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Flag multiple sub-10K deposits within 48 hours"}},
            {"tool": "assess_risk", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "file_sar", "parameters": {
                "typology": "structuring",
                "entities_involved": ["CUST-001"],
                "findings": ["sub_threshold_deposits", "multiple_branches", "rapid_frequency"],
                "evidence_chain": "8 cash deposits under $10K across 3 branches in 48 hours",
            }},
        ],
    )


def _build_sanctions_scenario() -> EvalScenario:
    """Sanctions evasion via layered transfers."""
    return EvalScenario(
        name="sanctions_evasion",
        typology="layering",
        difficulty="hard",
        description="PEP-linked entity on sanctions list, layered transfers to evade screening",
        expected_typology="layering",
        expected_entities=[],
        actions=[
            {"tool": "review_alert", "parameters": {}},
            {"tool": "get_customer_profile", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "check_watchlist", "parameters": {"entity": "CUST-001"}},
            {"tool": "write_to_case_file", "parameters": {"note": "Subject appears on OFAC SDN list. Immediate escalation required."}},
            {"tool": "query_transactions", "parameters": {"customer_id": "CUST-001"}},
            {"tool": "trace_network", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "search_compliance_manual", "parameters": {"query": "sanctions OFAC SDN PEP screening"}},
            {"tool": "update_system_prompt", "parameters": {"rule": "Freeze all transactions for sanctioned entities pending review"}},
            {"tool": "check_source_of_funds", "parameters": {"transaction_id": "TXN-001"}},
            {"tool": "assess_risk", "parameters": {"entity_id": "CUST-001"}},
            {"tool": "file_sar", "parameters": {
                "typology": "layering",
                "entities_involved": ["CUST-001"],
                "findings": ["sanctions_hit", "pep_connection", "layered_transfers"],
                "evidence_chain": "OFAC SDN match + PEP connection + multi-hop transfers",
            }},
        ],
    )


SCENARIO_BUILDERS = {
    "1mdb_layering": _build_1mdb_scenario,
    "mule_ring": _build_mule_ring_scenario,
    "phantom_invoice": _build_phantom_invoice_scenario,
    "pass_through": _build_pass_through_scenario,
    "structuring": _build_structuring_scenario,
    "sanctions_evasion": _build_sanctions_scenario,
}


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    """Metrics for a single scenario evaluation."""
    scenario_name: str
    typology: str
    difficulty: str
    final_score: float
    total_steps: int
    page_faults: int = 0
    async_timeouts: int = 0
    disk_writes: int = 0
    kernel_injections: int = 0
    tool_calls: List[str] = field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class HarnessResult:
    """Aggregate metrics across all scenarios."""
    scenarios: List[ScenarioResult] = field(default_factory=list)
    mean_score: float = 0.0
    per_typology: Dict[str, float] = field(default_factory=dict)
    total_page_faults: int = 0
    total_async_timeouts: int = 0
    total_disk_writes: int = 0
    total_kernel_injections: int = 0
    ram_capacity: int = 2


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def run_scenario(eval_scenario: EvalScenario, verbose: bool = True, model: Any = None, tokenizer: Any = None) -> ScenarioResult:
    """Execute a single evaluation scenario and collect metrics."""
    result = ScenarioResult(
        scenario_name=eval_scenario.name,
        typology=eval_scenario.typology,
        difficulty=eval_scenario.difficulty,
        final_score=0.0,
        total_steps=0,
    )

    t0 = time.time()

    try:
        # Generate a procedural scenario matching the typology
        generator = ScenarioGenerator()
        scenario = generator.generate(
            typology=eval_scenario.typology,
            difficulty=eval_scenario.difficulty,
        )

        # Create environment
        env = AMLEnvironment(scenario)
        obs = env.reset()

        last_job_id = None

        if model is not None and tokenizer is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from train_grpo import DEFENDER_SYSTEM_PROMPT, parse_tool_call
            
            for step_idx in range(1, 26):
                ram = list(env._sm.ram_queue) if env._sm else []
                disk = env._sm.disk_contents if env._sm else []
                kernel = env._sm.kernel_directives if env._sm else []

                alert = obs.tool_result.get("alert", {}) if step_idx == 1 else {}
                if step_idx == 1:
                    alert_text = (
                        f"New AML Alert Assigned:\n"
                        f"- Alert ID: {alert.get('alert_id', 'N/A')}\n"
                        f"- Summary: {alert.get('summary', 'No summary')}\n"
                        f"- Customer: {alert.get('customer_id', 'N/A')}\n"
                        f"- Risk Level: {alert.get('risk_level', 'N/A')}\n"
                        f"- Total Amount: ${alert.get('total_amount', 'N/A')}\n"
                        f"- Alert Type: {alert.get('alert_type', eval_scenario.typology)}\n\n"
                        f"Available tools: {obs.available_tools}\n\n"
                        f"Investigate this alert. Use the available tools to gather evidence, "
                        f"then make your decision: file_sar or close_alert."
                    )
                else:
                    alert_text = (
                        f"Observation:\n{obs.message}\n"
                        f"Tool result: {obs.tool_result}\n\n"
                        f"Memory: {ram}\nDisk: {disk}\nKernel: {kernel}\n\n"
                        f"Action JSON:"
                    )
                    
                messages = [
                    {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
                    {"role": "user", "content": alert_text},
                ]
                
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1856).to(device)
                
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=192, temperature=0.3,
                        top_p=0.9, do_sample=True, repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                action_def = parse_tool_call(response)
                if not action_def:
                    action_def = {"tool": "invalid_action", "parameters": {}}
                    
                tool = action_def.get("tool", "invalid_action")
                params = action_def.get("parameters", {})
                
                if "__LAST_JOB_ID__" in str(params):
                    if last_job_id is None:
                        last_job_id = "REQ-001"
                    params = {k: (last_job_id if v == "__LAST_JOB_ID__" else v) for k, v in params.items()}
                
                obs = env.step(AMLAction(tool=tool, parameters=params))
                
                if tool == "request_wire_trace" and isinstance(obs.tool_result, dict):
                    last_job_id = obs.tool_result.get("job_id", last_job_id)

                result.tool_calls.append(tool)
                result.total_steps += 1
                reward = obs.reward if obs.reward is not None else 0.0
                if reward == -0.05 or (isinstance(obs.tool_result, dict) and obs.tool_result.get("page_fault")):
                    result.page_faults += 1
                if tool == "write_to_case_file" and reward > 0:
                    result.disk_writes += 1
                if tool == "update_system_prompt" and reward > 0:
                    result.kernel_injections += 1
                if tool == "retrieve_async_result" and reward < 0:
                    result.async_timeouts += 1

                if verbose:
                    print(f"    Step {step_idx:>2} | {tool:<30} | R={reward:+.4f} | {'DONE' if obs.done else 'ok'}")
                    
                if obs.done:
                    break
        else:
            for step_idx, action_def in enumerate(eval_scenario.actions):
                tool = action_def["tool"]
                params = dict(action_def["parameters"])
    
                # Dynamic job ID substitution
                if "__LAST_JOB_ID__" in str(params):
                    if last_job_id is None:
                        last_job_id = "REQ-001"
                    params = {k: (last_job_id if v == "__LAST_JOB_ID__" else v) for k, v in params.items()}
    
                obs = env.step(AMLAction(tool=tool, parameters=params))
    
                # Capture async job IDs
                if tool == "request_wire_trace" and isinstance(obs.tool_result, dict):
                    last_job_id = obs.tool_result.get("job_id", last_job_id)
    
                result.tool_calls.append(tool)
                result.total_steps += 1
    
                # Track OS mechanics from observations
                reward = obs.reward if obs.reward is not None else 0.0
                if reward == -0.05 or (isinstance(obs.tool_result, dict) and obs.tool_result.get("page_fault")):
                    result.page_faults += 1
                if tool == "write_to_case_file" and reward > 0:
                    result.disk_writes += 1
                if tool == "update_system_prompt" and reward > 0:
                    result.kernel_injections += 1
                if tool == "retrieve_async_result" and reward < 0:
                    result.async_timeouts += 1
    
                if verbose:
                    r_str = f"R={reward:+.4f}"
                    done_str = "DONE" if obs.done else "ok"
                    print(f"    Step {step_idx+1:>2} | {tool:<30} | {r_str} | {done_str}")
    
                if obs.done:
                    break

        result.final_score = env._state.accumulated_reward
        result.completed = True

    except Exception as e:
        result.error = str(e)
        if verbose:
            print(f"    ERROR: {e}")

    result.elapsed_seconds = time.time() - t0
    return result


def run_harness(
    scenario_names: Optional[List[str]] = None,
    ram_capacity: int = 2,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
) -> HarnessResult:
    """Run the full evaluation harness."""
    from state_manager import RAM_CAPACITY as default_ram

    if scenario_names is None:
        scenario_names = list(SCENARIO_BUILDERS.keys())

    harness = HarnessResult(ram_capacity=ram_capacity)

    model, tokenizer = None, None
    if checkpoint_path:
        from unsloth import FastLanguageModel
        if verbose:
            print(f"  [+] Loading checkpoint: {checkpoint_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path, max_seq_length=2048, load_in_4bit=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        FastLanguageModel.for_inference(model)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  MEMEX EVALUATION HARNESS")
        print(f"  Scenarios: {len(scenario_names)}")
        print(f"  RAM Capacity: {ram_capacity}")
        if checkpoint_path:
            print(f"  Model: {checkpoint_path}")
        print("=" * 60 + "\n")

    for name in scenario_names:
        if name not in SCENARIO_BUILDERS:
            print(f"  [!] Unknown scenario: {name}, skipping")
            continue

        eval_scenario = SCENARIO_BUILDERS[name]()

        if verbose:
            print(f"  [{name.upper()}] {eval_scenario.description}")
            print(f"  Typology: {eval_scenario.typology} | Difficulty: {eval_scenario.difficulty}")

        result = run_scenario(eval_scenario, verbose=verbose, model=model, tokenizer=tokenizer)
        harness.scenarios.append(result)

        if verbose:
            status = "PASS" if result.completed else "FAIL"
            print(f"  [{status}] Score: {result.final_score:+.4f} | "
                  f"Steps: {result.total_steps} | "
                  f"PF: {result.page_faults} | "
                  f"Disk: {result.disk_writes} | "
                  f"Kernel: {result.kernel_injections} | "
                  f"Time: {result.elapsed_seconds:.1f}s")
            print()

    # Aggregate
    scores = [r.final_score for r in harness.scenarios if r.completed]
    harness.mean_score = sum(scores) / max(len(scores), 1)

    # Per-typology breakdown
    typo_scores: Dict[str, List[float]] = {}
    for r in harness.scenarios:
        if r.completed:
            typo_scores.setdefault(r.typology, []).append(r.final_score)
    harness.per_typology = {t: sum(s)/len(s) for t, s in typo_scores.items()}

    harness.total_page_faults = sum(r.page_faults for r in harness.scenarios)
    harness.total_async_timeouts = sum(r.async_timeouts for r in harness.scenarios)
    harness.total_disk_writes = sum(r.disk_writes for r in harness.scenarios)
    harness.total_kernel_injections = sum(r.kernel_injections for r in harness.scenarios)

    if verbose:
        print("=" * 60)
        print(f"  HARNESS COMPLETE")
        print(f"  Mean Score: {harness.mean_score:+.4f}")
        print(f"  Per-Typology: {harness.per_typology}")
        print(f"  OS Mechanics: PF={harness.total_page_faults} "
              f"Disk={harness.total_disk_writes} "
              f"Kernel={harness.total_kernel_injections} "
              f"Async-TO={harness.total_async_timeouts}")
        print("=" * 60 + "\n")

    return harness


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Memex Evaluation Harness — Multi-Typology Benchmark Suite",
    )
    p.add_argument("--scenario", type=str, default=None,
                   choices=list(SCENARIO_BUILDERS.keys()),
                   help="Run a single scenario (default: all)")
    p.add_argument("--checkpoint", "--model", type=str, default=None,
                   dest="checkpoint_path",
                   help="Path to trained checkpoint to evaluate (LLM-driven)")
    p.add_argument("--output", type=str, default="eval_results.json",
                   help="Output JSON file for results")
    p.add_argument("--ram-capacity", type=int, default=2,
                   help="RAM capacity for evaluation (default: 2)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose output")
    args = p.parse_args()

    scenarios = [args.scenario] if args.scenario else None

    harness_result = run_harness(
        scenario_names=scenarios,
        ram_capacity=args.ram_capacity,
        verbose=not args.quiet,
        checkpoint_path=args.checkpoint_path,
    )

    # Write JSON output
    output = {
        "mean_score": harness_result.mean_score,
        "per_typology": harness_result.per_typology,
        "ram_capacity": harness_result.ram_capacity,
        "os_mechanics": {
            "page_faults": harness_result.total_page_faults,
            "async_timeouts": harness_result.total_async_timeouts,
            "disk_writes": harness_result.total_disk_writes,
            "kernel_injections": harness_result.total_kernel_injections,
        },
        "scenarios": [asdict(r) for r in harness_result.scenarios],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if not args.quiet:
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
