"""
Memex OS-Agent Benchmark — GRPO Training Script.

Trains the Defender agent using TRL's GRPOTrainer + Unsloth 4-bit quantization.
This is the PRIMARY training entrypoint for the OpenEnv hackathon submission.

Architecture:
    - Model: Meta-Llama-3.1-8B-Instruct (4-bit via Unsloth) with LoRA adapters
    - Algorithm: GRPO (Group Relative Policy Optimization)
    - Reward: OpenEnv reward function — runs each completion through AMLEnvironment
    - Dataset: Procedurally generated AML investigation prompts

Usage:
    # Colab / HF Jobs
    python train_grpo.py

    # Dry-run (2 prompts, 1 epoch, no WandB)
    python train_grpo.py --dry-run

    # Custom settings
    python train_grpo.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
        --num-prompts 200 --num-generations 4 --lr 5e-6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOTrainConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 4096
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO
    num_generations: int = 4        # G — group size per prompt
    max_completion_length: int = 2048
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    beta: float = 0.04              # KL penalty
    loss_type: str = "grpo"         # "grpo", "dapo", "dr_grpo"
    scale_rewards: bool = True

    # Dataset
    num_prompts: int = 100          # Number of unique scenario prompts
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])

    # Infrastructure
    output_dir: str = "checkpoints/defender-grpo"
    wandb_project: str = "memex-grpo"
    logging_steps: int = 1
    save_steps: int = 25
    dry_run: bool = False

    # Environment
    env_base_url: str = ""          # If set, use remote HF Space; else local


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

DEFENDER_SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) investigator operating within the Memex OS-Agent environment.

Your job: Investigate financial alerts and decide whether to file a SAR (Suspicious Activity Report) or close the alert as a false positive.

## Available Tools
You have 18 tools organized in three categories:

### Domain Investigation Tools
- review_alert: Review the current alert details
- get_customer_profile(customer_id): Get KYC profile for a customer
- query_transactions(customer_id): Query transaction records
- check_watchlist(entity_name): Screen entity against sanctions/PEP lists
- trace_network(entity_id, depth): Map entity relationships
- check_source_of_funds(transaction_id): Verify fund origins
- check_market_price(commodity): Check commodity prices (trade-based ML)
- assess_risk(customer_id): Get computed risk assessment
- check_device_overlap(entity_id): Check shared device fingerprints
- verify_customs_invoice(invoice_id): Verify trade documentation
- query_beneficial_ownership(entity_id): Trace UBO through shell layers

### OS-Mechanic Tools (Memory Management)
- write_to_case_file(content): Save findings to persistent disk (prevents data loss from RAM eviction)
- request_wire_trace(entity_id): Start async background trace (returns job_id, result available after ETA)
- retrieve_async_result(job_id): Retrieve completed async job
- search_compliance_manual(query): Look up AML regulations
- update_system_prompt(rule): Inject compliance rule into your active directives

### Terminal Actions (end the episode)
- file_sar(typology, entities_involved, findings, evidence_chain): File a Suspicious Activity Report
- close_alert(reason): Close alert as false positive

## Investigation Protocol
1. Start with review_alert to understand the case
2. Gather evidence: customer profiles, transactions, watchlists, network
3. Use OS tools: write important findings to case file, request async traces
4. Assess risk when you have enough evidence
5. Make your decision: file_sar or close_alert

## Important
- RAM only holds 2 items — data gets evicted! Use write_to_case_file to persist critical findings.
- Async wire traces take 2-4 steps. Don't retrieve early (timeout penalty).
- You can inject compliance rules via update_system_prompt for better decisions.

Respond with a JSON tool call: {"tool": "<tool_name>", "parameters": {...}}"""


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_prompt_dataset(num_prompts: int, difficulties: List[str]) -> list:
    """Generate AML investigation prompts from the procedural scenario engine.

    Each prompt = system prompt + initial alert observation from a fresh
    AMLEnvironment.reset() call. This creates diverse, unique scenarios.
    """
    from server.aml_environment import AMLEnvironment

    prompts = []
    for i in range(num_prompts):
        try:
            env = AMLEnvironment()
            task_id = difficulties[i % len(difficulties)]
            obs = env.reset(task_id=task_id)
            alert = obs.tool_result.get("alert", {})

            # Build the user message from the alert observation
            alert_text = (
                f"New AML Alert Assigned:\n"
                f"- Alert ID: {alert.get('alert_id', 'N/A')}\n"
                f"- Summary: {alert.get('summary', 'No summary')}\n"
                f"- Customer: {alert.get('customer_id', 'N/A')}\n"
                f"- Risk Level: {alert.get('risk_level', 'N/A')}\n"
                f"- Total Amount: ${alert.get('total_amount', 'N/A')}\n"
                f"- Alert Type: {alert.get('alert_type', task_id)}\n\n"
                f"Available tools: {obs.available_tools}\n\n"
                f"Investigate this alert. Use the available tools to gather evidence, "
                f"then make your decision: file_sar or close_alert."
            )

            prompts.append({
                "prompt": [
                    {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
                    {"role": "user", "content": alert_text},
                ],
                "task_id": task_id,
            })
        except Exception as e:
            print(f"  ⚠ Failed to generate prompt {i}: {e}")
            continue

    print(f"  ✓ Generated {len(prompts)} scenario prompts")
    return prompts


# ═══════════════════════════════════════════════════════════════════════════
# Tool Call Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON tool call from model completion text.

    Handles various formats:
    - Raw JSON: {"tool": "review_alert", "parameters": {...}}
    - Markdown code blocks: ```json {...} ```
    - Embedded in reasoning text
    """
    if not text or not isinstance(text, str):
        return None

    # Pattern 1: ```json {...} ``` blocks
    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Pattern 2: Raw JSON object with "tool" key
    json_pattern = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', text)
    if json_pattern:
        try:
            data = json.loads(json_pattern.group(0))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Pattern 3: Try the entire text as JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "tool" in data:
            return data
    except json.JSONDecodeError:
        pass

    return None


def _extract_completion_text(completion) -> str:
    """Normalize TRL completion formats to plain text."""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    elif isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


# Valid tool names — the 18 tools from AMLEnvironment
_VALID_TOOLS = {
    # Domain investigation (12)
    "review_alert", "get_customer_profile", "query_transactions",
    "check_watchlist", "trace_network", "check_source_of_funds",
    "check_market_price", "assess_risk", "file_sar", "close_alert",
    "check_device_overlap", "verify_customs_invoice",
    "query_beneficial_ownership",
    # OS-mechanic tools (5)
    "write_to_case_file", "request_wire_trace", "retrieve_async_result",
    "search_compliance_manual", "update_system_prompt",
}

# Investigation tools that show the agent is actually investigating
_INVESTIGATION_TOOLS = {
    "review_alert", "get_customer_profile", "query_transactions",
    "check_watchlist", "trace_network", "check_source_of_funds",
    "check_market_price", "assess_risk", "check_device_overlap",
    "verify_customs_invoice", "query_beneficial_ownership",
}

# OS-mechanic tools (the innovative features)
_OS_TOOLS = {
    "write_to_case_file", "request_wire_trace", "retrieve_async_result",
    "search_compliance_manual", "update_system_prompt",
}


# ═══════════════════════════════════════════════════════════════════════════
# Decomposed Reward Functions (Anti-Gaming Design)
#
# We pass MULTIPLE reward functions to GRPOTrainer. Each one scores a
# different dimension. TRL sums them for the final reward. This makes
# reward hacking much harder — gaming one signal doesn't help if the
# others penalize the degenerate behavior.
#
# R_total = R_format + R_investigation + R_execution + R_os_mechanics
# ═══════════════════════════════════════════════════════════════════════════

def reward_format_compliance(completions, **kwargs) -> List[float]:
    """R1: Format Compliance — Is the output a valid, well-formed tool call?

    Anti-gaming target: Prevents gibberish, repetitive text, or
    non-JSON output from getting any positive reward.

    Scoring:
        +0.2   Valid JSON with "tool" key pointing to a known tool
        +0.1   Valid JSON with "tool" key but unknown tool name
        -0.5   No valid JSON tool call found (gibberish/off-format)
        -1.0   Empty or degenerate output (< 5 chars or >80% repeated tokens)
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)

        # Check for degenerate output
        if not text or len(text.strip()) < 5:
            rewards.append(-1.0)
            continue

        # Detect degenerate repetition (> 80% of tokens are the same)
        tokens = text.split()
        if tokens and len(tokens) > 5:
            from collections import Counter
            most_common_count = Counter(tokens).most_common(1)[0][1]
            if most_common_count / len(tokens) > 0.8:
                rewards.append(-1.0)
                continue

        # Parse tool call
        tool_call = parse_tool_call(text)
        if tool_call is None:
            rewards.append(-0.5)
            continue

        tool = tool_call.get("tool", "").strip().lower()
        if tool in _VALID_TOOLS:
            rewards.append(0.2)
        elif tool:
            rewards.append(0.1)  # Valid JSON, unknown tool
        else:
            rewards.append(-0.5)

    return rewards


def reward_investigation_quality(completions, **kwargs) -> List[float]:
    """R2: Investigation Quality — Is the agent choosing appropriate tools?

    Anti-gaming target: Prevents the agent from always calling the same
    tool or only calling terminal actions without investigating.

    Scoring:
        +0.3   Used an investigation tool (evidence gathering)
        +0.2   Used an OS-mechanic tool (memory management/async/kernel)
        +0.1   Used a terminal tool (file_sar/close_alert) — lower because
               we don't want premature terminal actions
        -0.3   Provided parameters that are clearly empty/dummy
         0.0   No valid tool call
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        tool_call = parse_tool_call(text)

        if tool_call is None:
            rewards.append(0.0)
            continue

        tool = tool_call.get("tool", "").strip().lower()
        params = tool_call.get("parameters", {})

        # Check for dummy/empty parameters on tools that require them
        tools_needing_params = {
            "get_customer_profile", "query_transactions", "check_watchlist",
            "trace_network", "check_source_of_funds", "write_to_case_file",
            "file_sar", "check_device_overlap",
        }
        if tool in tools_needing_params and (not params or all(
            v == "" or v is None for v in params.values()
        )):
            rewards.append(-0.3)
            continue

        if tool in _INVESTIGATION_TOOLS:
            rewards.append(0.3)
        elif tool in _OS_TOOLS:
            rewards.append(0.2)
        elif tool in {"file_sar", "close_alert"}:
            rewards.append(0.1)
        else:
            rewards.append(0.0)

    return rewards


def reward_environment_execution(completions, **kwargs) -> List[float]:
    """R3: Environment Execution — The ground-truth reward from the environment.

    This is the core OpenEnv pattern: execute the tool call against a fresh
    AMLEnvironment instance and return the environment's reward signal.

    Anti-gaming target: This is the HARDEST reward to game because it
    requires actually interacting correctly with the environment. The
    environment's dense reward system includes:
    - Action cost (-0.02 per step)
    - Redundancy penalty (-0.03 for duplicate calls)
    - Page fault penalty (-0.05 for accessing evicted data)
    - Async timeout penalty (-0.10 for premature retrieval)
    - Investigation bonuses (+0.02-0.05 for first use of each tool)
    - Terminal TP/TN/FP/FN scoring
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        tool_call = parse_tool_call(text)

        if tool_call is None:
            rewards.append(-0.5)
            continue

        tool = tool_call.get("tool", "").strip().lower()
        params = tool_call.get("parameters", {})
        if isinstance(params, str):
            params = {}

        try:
            from server.aml_environment import AMLEnvironment
            from models import AMLAction

            env = AMLEnvironment()
            task_id = kwargs.get("task_id", ["easy"])
            if isinstance(task_id, list):
                task_id = task_id[0] if task_id else "easy"
            env.reset(task_id=task_id)

            action = AMLAction(tool=tool, parameters=params)
            obs = env.step(action)

            if obs.done:
                # Terminal action — full episode score
                rewards.append(float(obs.reward or 0.0))
            else:
                # Non-terminal — step reward from grader
                rewards.append(float(obs.reward or 0.0))

        except Exception:
            rewards.append(-0.3)

    return rewards


def reward_os_mechanics(completions, **kwargs) -> List[float]:
    """R4: OS Mechanics — Does the agent leverage the OS-inspired features?

    Anti-gaming target: Ensures the agent doesn't ignore the innovative
    OS mechanics. An agent that never writes to disk, never uses async,
    and never updates its system prompt gets penalized.

    Scoring:
        +0.3   Used write_to_case_file (demonstrates memory management)
        +0.3   Used search_compliance_manual (knowledge retrieval)
        +0.2   Used update_system_prompt (kernel-level meta-prompting)
        +0.2   Used request_wire_trace (async job scheduling)
        +0.1   Used retrieve_async_result (interrupt handling)
         0.0   Used a non-OS tool (neutral)
        -0.1   Provided invalid content to write_to_case_file (empty string)
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        tool_call = parse_tool_call(text)

        if tool_call is None:
            rewards.append(0.0)
            continue

        tool = tool_call.get("tool", "").strip().lower()
        params = tool_call.get("parameters", {})

        if tool == "write_to_case_file":
            content = params.get("content", params.get("note", ""))
            if content and len(str(content).strip()) > 3:
                rewards.append(0.3)  # Good: persisting evidence to disk
            else:
                rewards.append(-0.1)  # Empty write — gaming attempt
        elif tool == "search_compliance_manual":
            query = params.get("query", "")
            if query and len(str(query).strip()) > 2:
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        elif tool == "update_system_prompt":
            rule = params.get("rule", "")
            if rule and len(str(rule).strip()) > 5:
                rewards.append(0.2)  # Kernel-level meta-prompting
            else:
                rewards.append(-0.1)  # Empty injection — gaming
        elif tool == "request_wire_trace":
            rewards.append(0.2)  # Async job scheduling
        elif tool == "retrieve_async_result":
            rewards.append(0.1)  # Interrupt handling
        else:
            rewards.append(0.0)  # Non-OS tool — neutral for this reward

    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: GRPOTrainConfig) -> None:
    """Main GRPO training loop using TRL + Unsloth."""

    print("\n" + "═" * 70)
    print("  MEMEX DEFENDER — GRPO TRAINING (TRL + Unsloth)")
    print("═" * 70)
    print(f"  Model:       {cfg.model_name}")
    print(f"  LoRA r:      {cfg.lora_r}")
    print(f"  Group size:  G={cfg.num_generations}")
    print(f"  Prompts:     {cfg.num_prompts}")
    print(f"  LR:          {cfg.learning_rate}")
    print(f"  Dry run:     {cfg.dry_run}")
    print("═" * 70 + "\n")

    # Override for dry-run
    if cfg.dry_run:
        cfg.num_prompts = 4
        cfg.num_train_epochs = 1
        cfg.num_generations = 2
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.save_steps = 999
        cfg.max_completion_length = 512

    # ── 1. Load Model with Unsloth ──────────────────────────────────────
    print("▸ Loading model with Unsloth (4-bit quantization)...")
    from unsloth import FastLanguageModel

    import torch

    # Explicitly set dtype to bfloat16 to match GRPOConfig(bf16=True).
    # Using dtype=None (auto-detect) can pick float16 on some GPUs,
    # which causes "self and mat2 must have the same dtype, but got
    # Half and BFloat16" in Unsloth's LoRA matmul kernels.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.bfloat16,
    )

    print("▸ Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"  ✓ Model loaded. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 2. Generate Prompt Dataset ──────────────────────────────────────
    print(f"\n▸ Generating {cfg.num_prompts} scenario prompts...")
    prompt_data = generate_prompt_dataset(cfg.num_prompts, cfg.difficulties)

    from datasets import Dataset
    dataset = Dataset.from_list(prompt_data)
    print(f"  ✓ Dataset: {len(dataset)} prompts")

    # ── 3. Create Decomposed Reward Functions ───────────────────────────
    # We use MULTIPLE reward functions to prevent reward hacking.
    # Each scores a different dimension. TRL sums them for the total.
    # R_total = R_format + R_investigation + R_execution + R_os_mechanics
    print("\n▸ Setting up decomposed reward functions (anti-gaming design)...")
    reward_fns = [
        reward_format_compliance,        # R1: Valid JSON tool call?
        reward_investigation_quality,    # R2: Appropriate tool choice?
        reward_environment_execution,    # R3: Ground-truth env reward
        reward_os_mechanics,             # R4: OS feature utilization?
    ]
    print(f"  ✓ {len(reward_fns)} reward functions registered:")
    print("    R1: Format Compliance (prevents gibberish)")
    print("    R2: Investigation Quality (prevents lazy tool choice)")
    print("    R3: Environment Execution (ground-truth env reward)")
    print("    R4: OS Mechanics (rewards memory/async/kernel usage)")

    # ── 4. Configure GRPOTrainer ────────────────────────────────────────
    print("\n▸ Configuring GRPOTrainer...")
    from trl import GRPOTrainer, GRPOConfig

    report_to = "none" if cfg.dry_run else "wandb"

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=report_to,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        seed=42,
        # Log completions for debugging
        log_completions=True,
    )

    if not cfg.dry_run:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=f"defender-grpo-G{cfg.num_generations}-{time.strftime('%m%d-%H%M')}",
            config={
                "model": cfg.model_name,
                "lora_r": cfg.lora_r,
                "num_generations": cfg.num_generations,
                "learning_rate": cfg.learning_rate,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "num_prompts": cfg.num_prompts,
                "reward_functions": [
                    "format_compliance", "investigation_quality",
                    "environment_execution", "os_mechanics",
                ],
            },
        )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,  # List of 4 decomposed reward functions
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 5. Train ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  STARTING GRPO TRAINING")
    print("═" * 70 + "\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\n  ✓ Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── 6. Save ─────────────────────────────────────────────────────────
    print(f"\n▸ Saving to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"  ✓ Model saved to {cfg.output_dir}")

    # ── 7. Plot Training Curves ─────────────────────────────────────────
    try:
        _plot_training_curves(trainer, cfg.output_dir)
    except Exception as e:
        print(f"  ⚠ Could not generate plots: {e}")

    # ── 8. Cleanup ──────────────────────────────────────────────────────
    if not cfg.dry_run:
        import wandb
        wandb.finish()

    print(f"\n{'═' * 70}")
    print(f"  DEFENDER GRPO TRAINING COMPLETE")
    print(f"  Output:   {cfg.output_dir}")
    print(f"  Duration: {elapsed:.0f}s")
    print(f"{'═' * 70}\n")


def _plot_training_curves(trainer, output_dir: str) -> None:
    """Extract and save training curves from trainer log history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history
    if not log_history:
        return

    # Extract metrics
    steps, rewards, losses = [], [], []
    for entry in log_history:
        step = entry.get("step", 0)
        if "reward" in entry:
            steps.append(step)
            rewards.append(entry["reward"])
        if "loss" in entry:
            losses.append((step, entry["loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Memex Defender GRPO Training", fontsize=14, fontweight="bold")

    # Reward plot
    if rewards:
        axes[0].plot(steps[:len(rewards)], rewards, "b-", alpha=0.7, linewidth=1.5)
        axes[0].set_title("Mean Reward per Step")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Loss plot
    if losses:
        loss_steps, loss_vals = zip(*losses)
        axes[1].plot(loss_steps, loss_vals, "r-", alpha=0.7, linewidth=1.5)
        axes[1].set_title("GRPO Loss")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved to {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Memex Defender GRPO Training (TRL + Unsloth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=GRPOTrainConfig.model_name,
                    help="HuggingFace model ID (Unsloth 4-bit recommended)")
    p.add_argument("--lr", type=float, default=GRPOTrainConfig.learning_rate)
    p.add_argument("--lora-r", type=int, default=GRPOTrainConfig.lora_r)
    p.add_argument("--num-prompts", type=int, default=GRPOTrainConfig.num_prompts,
                    help="Number of unique scenario prompts to generate")
    p.add_argument("--num-generations", type=int, default=GRPOTrainConfig.num_generations,
                    help="G — number of completions per prompt (group size)")
    p.add_argument("--max-completion-length", type=int,
                    default=GRPOTrainConfig.max_completion_length)
    p.add_argument("--epochs", type=int, default=GRPOTrainConfig.num_train_epochs)
    p.add_argument("--batch-size", type=int,
                    default=GRPOTrainConfig.per_device_train_batch_size)
    p.add_argument("--grad-accum", type=int,
                    default=GRPOTrainConfig.gradient_accumulation_steps)
    p.add_argument("--beta", type=float, default=GRPOTrainConfig.beta,
                    help="KL penalty coefficient (0.0 = no KL)")
    p.add_argument("--loss-type", choices=["grpo", "dapo", "dr_grpo"],
                    default=GRPOTrainConfig.loss_type)
    p.add_argument("--wandb-project", default=GRPOTrainConfig.wandb_project)
    p.add_argument("--output-dir", default=GRPOTrainConfig.output_dir)
    p.add_argument("--dry-run", action="store_true",
                    help="Quick test: 4 prompts, 1 epoch, no WandB")
    args = p.parse_args()

    train(GRPOTrainConfig(
        model_name=args.model,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        num_prompts=args.num_prompts,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        loss_type=args.loss_type,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
