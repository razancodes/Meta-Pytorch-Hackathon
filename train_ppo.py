#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — PPO Training Script.

Trains an 8B LLM (Llama-3.1-8B-Instruct) on our AML investigation
environment using Proximal Policy Optimization.

Stack:
  - Unsloth:  4-bit quantized model + LoRA adapters (low VRAM)
  - TRL:      PPOTrainer for RLHF-style policy gradient updates
  - WandB:    Experiment tracking and OS-mechanic dashboards

Architecture:
  Each training iteration collects N episodes of multi-step trajectories.
  Every (prompt, response, reward) step is fed to TRL's PPOTrainer.
  The model learns to maximize dense per-step rewards AND the terminal
  composite score from our AMLGrader.

Usage:
  python train_ppo.py                           # defaults
  python train_ppo.py --episodes 200 --lr 5e-6  # custom
  python train_ppo.py --dry-run                  # 2 episodes, no WandB

See TRAINING.md for full setup instructions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction, AMLObservation
from server.aml_environment import AMLEnvironment


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 4096
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # PPO
    learning_rate: float = 5e-6
    ppo_epochs: int = 4
    mini_batch_size: int = 2
    batch_size: int = 4        # steps collected before PPO update
    gamma: float = 0.99        # discount factor
    lam: float = 0.95          # GAE lambda
    cliprange: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    kl_penalty: str = "kl"     # KL penalty type
    init_kl_coef: float = 0.05 # initial KL coefficient
    adap_kl_ctrl: bool = True  # adaptive KL control
    target_kl: float = 6.0     # target KL divergence

    # Generation (strict JSON mode)
    gen_max_new_tokens: int = 256
    gen_temperature: float = 0.3    # low = deterministic tool calls
    gen_top_p: float = 0.9
    gen_do_sample: bool = True
    gen_repetition_penalty: float = 1.1

    # Environment
    episodes_per_iter: int = 4   # episodes per PPO iteration
    total_iterations: int = 50   # total PPO iterations
    max_steps_per_episode: int = 25
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])
    typologies: list = field(default_factory=lambda: [
        "structuring", "layering", "trade_based_ml",
    ])

    # Logging
    wandb_project: str = "memex-ppo"
    wandb_run_name: str = ""
    log_every_episode: bool = True

    # Checkpointing
    save_every: int = 10        # save every N iterations
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")

    # Debug
    dry_run: bool = False


# ===========================================================================
# System Prompt (reused from inference.py)
# ===========================================================================

SYSTEM_PROMPT = """You are a Senior AML Compliance Investigator operating within a Memex OS-Agent Environment.

## OUTPUT FORMAT (STRICTLY ENFORCED)
Respond with EXACTLY ONE raw JSON object. No markdown. No explanation outside JSON.
{"tool": "<tool_name>", "parameters": {<params>}, "reasoning": "<one sentence>"}

## OS MECHANICS
- RAM holds LAST 2 observations. Use write_to_case_file to save important data to disk.
- request_wire_trace returns async job. Wait for ETA before retrieve_async_result.
- search_compliance_manual + update_system_prompt injects rules into your directives.

## TOOLS: review_alert, get_customer_profile, query_transactions, check_watchlist,
trace_network, check_source_of_funds, assess_risk, check_market_price,
write_to_case_file, request_wire_trace, retrieve_async_result,
search_compliance_manual, update_system_prompt, file_sar, close_alert

## TERMINAL: file_sar(typology, entities_involved, findings) | close_alert(reason)
Typologies: "structuring" | "layering" | "trade_based_ml" | "false_positive"
"""


# ===========================================================================
# Prompt Formatter
# ===========================================================================

def format_prompt(
    observation: AMLObservation,
    step: int,
    kernel_directives: List[str],
    disk_contents: List[str],
    ram_contents: List[str],
) -> str:
    """Format the current state into a chat-style prompt string for the model.

    Uses Llama-3.1 chat template tokens for proper formatting.
    """
    # System message with kernel directives
    sys_parts = [SYSTEM_PROMPT]
    if kernel_directives and len(kernel_directives) > 1:
        sys_parts.append("\n## ACTIVE KERNEL DIRECTIVES")
        for d in kernel_directives:
            sys_parts.append(f"- {d}")
    system_msg = "\n".join(sys_parts)

    # Build user context
    user_parts = []

    # Disk (persistent memory)
    if disk_contents:
        user_parts.append("## CASE FILE (Disk — persistent)")
        for i, entry in enumerate(disk_contents, 1):
            user_parts.append(f"  {i}. {entry}")

    # RAM (last 2 observations)
    if ram_contents:
        user_parts.append("## RAM (recent observations)")
        for obs_text in ram_contents[-2:]:
            user_parts.append(f"  [RAM] {obs_text[:300]}")

    # Current observation
    user_parts.append(f"## STEP {step} OBSERVATION")
    obs_data = observation.tool_result if observation.tool_result else {}
    user_parts.append(json.dumps(obs_data, indent=1, default=str)[:1500])

    if observation.message:
        user_parts.append(f"Message: {observation.message[:300]}")

    user_msg = "\n".join(user_parts)

    # Format as Llama-3.1 chat template
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


# ===========================================================================
# Response Parser
# ===========================================================================

def parse_model_output(text: str) -> Tuple[str, Dict[str, Any]]:
    """Parse model output into (tool_name, parameters).

    Robust to markdown fences, extra text, and partial JSON.
    """
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")

    # Try full JSON parse
    try:
        parsed = json.loads(text)
        return parsed.get("tool", "review_alert"), parsed.get("parameters", {})
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed.get("tool", "review_alert"), parsed.get("parameters", {})
        except json.JSONDecodeError:
            pass

    # Last resort: extract tool name
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    tool = tool_match.group(1) if tool_match else "review_alert"
    return tool, {}


# ===========================================================================
# Episode Rollout
# ===========================================================================

@dataclass
class StepRecord:
    """One step of an episode trajectory."""
    prompt: str            # tokenized prompt fed to the model
    response: str          # raw text output from the model
    reward: float          # step reward from grader
    query_tensor: Any = None   # tokenized prompt tensor
    response_tensor: Any = None  # tokenized response tensor


@dataclass
class EpisodeResult:
    """Complete episode trajectory + summary stats."""
    steps: List[StepRecord]
    total_reward: float
    final_score: float
    difficulty: str
    typology: str
    page_faults: int = 0
    async_timeouts: int = 0
    successful_pages: int = 0
    meta_injections: int = 0
    step_count: int = 0
    done: bool = False


def rollout_episode(
    model: Any,
    tokenizer: Any,
    env: AMLEnvironment,
    config: TrainConfig,
    difficulty: str | None = None,
    typology: str | None = None,
    device: str = "cuda",
) -> EpisodeResult:
    """Collect a full episode trajectory.

    1. Reset the environment
    2. Loop: format prompt → generate → parse → step environment
    3. Collect (prompt, response, reward) tuples for PPO
    """
    diff = difficulty or random.choice(config.difficulties)
    typo = typology or random.choice(config.typologies)
    task_id = diff  # get_scenario handles the rest

    # Reset environment
    init_obs = env.reset(task_id=task_id)
    subject_id = init_obs.tool_result.get("alert", {}).get("customer_id", "")

    steps: List[StepRecord] = []
    total_reward = 0.0
    done = False

    for step_num in range(1, config.max_steps_per_episode + 1):
        if done:
            break

        # Get OS state from the StateManager
        ram_contents = env._sm.ram_queue if env._sm else []
        disk_contents = env._sm.disk_contents if env._sm else []
        kernel_directives = env._sm.kernel_directives if env._sm else []

        # Format the observation as the current obs on first step
        obs = init_obs if step_num == 1 else obs  # noqa: F821

        # Build prompt
        prompt_str = format_prompt(
            observation=obs,
            step=step_num,
            kernel_directives=kernel_directives,
            disk_contents=disk_contents,
            ram_contents=list(ram_contents),
        )

        # Tokenize
        inputs = tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length - config.gen_max_new_tokens,
        ).to(device)

        query_tensor = inputs["input_ids"].squeeze(0)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.gen_max_new_tokens,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                do_sample=config.gen_do_sample,
                repetition_penalty=config.gen_repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Extract only the generated tokens (exclude prompt)
        response_ids = outputs[0][query_tensor.shape[0]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Parse action
        tool_name, params = parse_model_output(response_text)

        # Step the environment
        action = AMLAction(tool=tool_name, parameters=params)
        obs = env.step(action)

        reward = obs.reward if obs.reward is not None else 0.0
        done = obs.done

        steps.append(StepRecord(
            prompt=prompt_str,
            response=response_text,
            reward=reward,
            query_tensor=query_tensor,
            response_tensor=response_ids,
        ))

        total_reward += reward

    # Extract OS mechanic stats from the environment state
    state = env._state
    final_score = state.accumulated_reward if state else total_reward

    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        final_score=final_score,
        difficulty=diff,
        typology=typo,
        page_faults=state.page_fault_count if state else 0,
        async_timeouts=state.async_timeout_count if state else 0,
        successful_pages=state.successful_pages if state else 0,
        meta_injections=state.meta_injections if state else 0,
        step_count=len(steps),
        done=done,
    )


# ===========================================================================
# Main Training Loop
# ===========================================================================

def train(config: TrainConfig) -> None:
    """Main PPO training function."""

    print("=" * 70)
    print("  MEMEX OS-AGENT BENCHMARK — PPO TRAINING")
    print("=" * 70)
    print(f"  Model:      {config.model_name}")
    print(f"  LoRA r:     {config.lora_r}")
    print(f"  LR:         {config.learning_rate}")
    print(f"  Episodes:   {config.episodes_per_iter} per iteration × {config.total_iterations} iters")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Dry run:    {config.dry_run}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load model with Unsloth                                          #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading model with Unsloth (4-bit quantized)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,  # auto-detect
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  ✓ Model loaded: {config.model_name}")
    print(f"  ✓ Vocab size: {len(tokenizer)}")

    # ------------------------------------------------------------------ #
    # 2. Attach LoRA adapters                                              #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"  ✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------ #
    # 3. Initialize PPOTrainer                                             #
    # ------------------------------------------------------------------ #
    print("\n[3/5] Initializing TRL PPOTrainer...")
    from trl import PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        ppo_epochs=config.ppo_epochs,
        mini_batch_size=config.mini_batch_size,
        batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        log_with="wandb" if not config.dry_run else None,
        seed=42,
        kl_penalty=config.kl_penalty,
        init_kl_coef=config.init_kl_coef,
        adap_kl_ctrl=config.adap_kl_ctrl,
        target=config.target_kl,
        cliprange=config.cliprange,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
    )

    # Create a frozen reference model for KL divergence
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    print("  ✓ PPOTrainer initialized")

    # ------------------------------------------------------------------ #
    # 4. Initialize WandB                                                  #
    # ------------------------------------------------------------------ #
    if not config.dry_run:
        print("\n[4/5] Initializing WandB...")
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"memex-ppo-{int(time.time())}",
            config={
                "model": config.model_name,
                "lora_r": config.lora_r,
                "lr": config.learning_rate,
                "batch_size": config.batch_size,
                "ppo_epochs": config.ppo_epochs,
                "episodes_per_iter": config.episodes_per_iter,
                "total_iterations": config.total_iterations,
                "gen_temperature": config.gen_temperature,
            },
        )
        print("  ✓ WandB initialized")
    else:
        print("\n[4/5] WandB SKIPPED (dry-run mode)")

    # ------------------------------------------------------------------ #
    # 5. Training Loop                                                     #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Starting PPO training loop...\n")

    env = AMLEnvironment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_iters = 2 if config.dry_run else config.total_iterations

    # Enable generation mode for rollouts
    FastLanguageModel.for_inference(model)

    global_episode = 0
    best_mean_reward = -float("inf")

    for iteration in range(1, total_iters + 1):
        iter_start = time.time()

        # ---- Collect trajectories ----
        all_queries: List[torch.Tensor] = []
        all_responses: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []

        iter_total_reward = 0.0
        iter_page_faults = 0
        iter_async_timeouts = 0
        iter_successful_pages = 0
        iter_meta_injections = 0
        iter_steps = 0
        iter_episodes = 0
        episode_scores: List[float] = []

        eps_per_iter = 1 if config.dry_run else config.episodes_per_iter

        for ep in range(eps_per_iter):
            global_episode += 1

            # Randomize difficulty and typology
            diff = random.choice(config.difficulties)
            typo = random.choice(config.typologies)

            episode = rollout_episode(
                model=model,
                tokenizer=tokenizer,
                env=env,
                config=config,
                difficulty=diff,
                typology=typo,
                device=device,
            )

            # Collect step-level (query, response, reward) for PPO
            for step_rec in episode.steps:
                if step_rec.query_tensor is not None and step_rec.response_tensor is not None:
                    all_queries.append(step_rec.query_tensor)
                    all_responses.append(step_rec.response_tensor)
                    all_rewards.append(torch.tensor(step_rec.reward, dtype=torch.float32))

            # Accumulate stats
            iter_total_reward += episode.final_score
            iter_page_faults += episode.page_faults
            iter_async_timeouts += episode.async_timeouts
            iter_successful_pages += episode.successful_pages
            iter_meta_injections += episode.meta_injections
            iter_steps += episode.step_count
            iter_episodes += 1
            episode_scores.append(episode.final_score)

            if config.log_every_episode:
                print(
                    f"  Iter {iteration:>3} | Ep {ep+1:>2} | "
                    f"{diff}/{typo} | "
                    f"steps={episode.step_count:>2} | "
                    f"score={episode.final_score:+.4f} | "
                    f"PF={episode.page_faults} AT={episode.async_timeouts} "
                    f"SP={episode.successful_pages} MI={episode.meta_injections}"
                )

        # ---- PPO Update ----
        if len(all_queries) >= config.mini_batch_size:
            # Switch model to training mode
            FastLanguageModel.for_training(model)

            try:
                stats = ppo_trainer.step(all_queries, all_responses, all_rewards)
                ppo_loss = stats.get("ppo/loss/total", 0.0)
                ppo_kl = stats.get("objective/kl", 0.0)
            except Exception as e:
                print(f"  ⚠ PPO step failed: {e}. Skipping update.")
                ppo_loss = 0.0
                ppo_kl = 0.0
                stats = {}

            # Switch back to inference mode for next rollout
            FastLanguageModel.for_inference(model)
        else:
            ppo_loss = 0.0
            ppo_kl = 0.0
            stats = {}
            print(f"  ⚠ Only {len(all_queries)} steps collected, skipping PPO update")

        # ---- Logging ----
        mean_reward = iter_total_reward / max(iter_episodes, 1)
        iter_time = time.time() - iter_start

        print(
            f"\n  === Iter {iteration}/{total_iters} Summary ===\n"
            f"    Mean reward:       {mean_reward:+.4f}\n"
            f"    Episode scores:    {[f'{s:+.3f}' for s in episode_scores]}\n"
            f"    PPO loss:          {ppo_loss:.6f}\n"
            f"    KL divergence:     {ppo_kl:.6f}\n"
            f"    Total steps:       {iter_steps}\n"
            f"    OS: PF={iter_page_faults} AT={iter_async_timeouts} "
            f"SP={iter_successful_pages} MI={iter_meta_injections}\n"
            f"    Time:              {iter_time:.1f}s\n"
        )

        if not config.dry_run:
            import wandb
            wandb.log({
                "iteration": iteration,
                "ppo/returns/mean": mean_reward,
                "ppo/loss/total": ppo_loss,
                "ppo/kl_divergence": ppo_kl,
                "env/mean_episode_score": mean_reward,
                "env/total_steps": iter_steps,
                "env/episodes": iter_episodes,
                "os/page_faults": iter_page_faults,
                "os/async_timeouts": iter_async_timeouts,
                "os/successful_pages": iter_successful_pages,
                "os/meta_injections": iter_meta_injections,
            })

        # ---- Checkpointing ----
        if iteration % config.save_every == 0 or mean_reward > best_mean_reward:
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                tag = "best"
            else:
                tag = f"iter-{iteration}"

            save_path = os.path.join(config.output_dir, tag)
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  💾 Checkpoint saved: {save_path}")

        # GC
        del all_queries, all_responses, all_rewards
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Finish ----
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Best mean reward: {best_mean_reward:+.4f}")
    print("=" * 70)

    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Final model saved: {final_path}")

    if not config.dry_run:
        import wandb
        wandb.finish()


# ===========================================================================
# Evaluation Helper
# ===========================================================================

def evaluate(
    model_path: str,
    num_episodes: int = 9,
    seed: int = 42,
) -> None:
    """Evaluate a trained model on all difficulty/typology combinations."""
    from unsloth import FastLanguageModel

    print(f"\n{'='*70}")
    print(f"  EVALUATION: {model_path}")
    print(f"{'='*70}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = TrainConfig()
    env = AMLEnvironment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    random.seed(seed)

    for diff in ["easy", "medium", "hard"]:
        for typo in ["structuring", "layering", "trade_based_ml"]:
            episode = rollout_episode(
                model, tokenizer, env, config,
                difficulty=diff, typology=typo, device=device,
            )
            results.append((diff, typo, episode))
            print(
                f"  {diff:>6}/{typo:<15} | "
                f"steps={episode.step_count:>2} | "
                f"score={episode.final_score:+.4f} | "
                f"done={episode.done} | "
                f"PF={episode.page_faults} AT={episode.async_timeouts} "
                f"SP={episode.successful_pages} MI={episode.meta_injections}"
            )

    scores = [r[2].final_score for r in results]
    print(f"\n  Mean: {sum(scores)/len(scores):+.4f}  |  "
          f"Min: {min(scores):+.4f}  |  Max: {max(scores):+.4f}")
    print("=" * 70)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memex PPO Training")

    p.add_argument("--model", type=str, default=TrainConfig.model_name,
                   help="HuggingFace model name or path")
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--episodes", type=int, default=TrainConfig.episodes_per_iter,
                   help="Episodes per PPO iteration")
    p.add_argument("--iterations", type=int, default=TrainConfig.total_iterations)
    p.add_argument("--temperature", type=float, default=TrainConfig.gen_temperature)
    p.add_argument("--wandb-project", type=str, default=TrainConfig.wandb_project)
    p.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    p.add_argument("--dry-run", action="store_true",
                   help="Quick 2-iteration test run without WandB")
    p.add_argument("--eval", type=str, default=None, metavar="MODEL_PATH",
                   help="Evaluate a trained checkpoint instead of training")

    return p.parse_args()


def main():
    args = parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    config = TrainConfig(
        model_name=args.model,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        episodes_per_iter=args.episodes,
        total_iterations=args.iterations,
        gen_temperature=args.temperature,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    train(config)


if __name__ == "__main__":
    main()
