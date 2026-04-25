#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — GRPO Training (L4-Optimized).

⚠️  EXPERIMENTAL / ABLATION ONLY — NOT PRODUCTION-READY  ⚠️

The production training path is train_ppo.py. This GRPO trainer is provided
for research comparison and ablation studies only.

Known deficiencies vs. PPO:
  1. No importance-ratio clipping — uses raw REINFORCE -(log_prob * advantage),
     making training vulnerable to large policy updates.
  2. KL penalty uses |KL| (absolute value) instead of proper forward KL,
     which penalizes the policy equally for moving toward or away from the
     reference — incorrect directionality.
  3. Loss is divided by total_steps, coupling gradient magnitude to group size
     instead of using a fixed effective batch size.

Group Relative Policy Optimization: eliminates the critic network entirely.
Instead of a value function V(s), GRPO averages rewards across G sampled
responses and uses that as the baseline:

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

Usage:
  python train_grpo.py --experimental --dry-run
  python train_grpo.py --experimental --iterations 150 --use-plr
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*attention mask API.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_return_dict.*")

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction, AMLObservation
from server.aml_environment import AMLEnvironment

# Import shared utilities from train_ppo.py
from train_ppo import (
    PPOConfig,
    SYSTEM_PROMPT,
    format_prompt,
    parse_action,
    StepData,
    EpisodeStats,
    vram_status,
)


# ═══════════════════════════════════════════════════════════════════════
# GRPO Configuration (extends PPOConfig)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GRPOConfig(PPOConfig):
    """GRPO-specific configuration extending PPOConfig."""

    # ── GRPO ──
    group_size: int = 4          # G: responses sampled per scenario for baseline
    kl_coef: float = 0.1        # KL penalty coefficient (higher than PPO default)
    entropy_coef: float = 0.05  # entropy bonus

    # ── Overrides ──
    wandb_project: str = "memex-grpo"


# ═══════════════════════════════════════════════════════════════════════
# GRPO Trainer
# ═══════════════════════════════════════════════════════════════════════

class MemexGRPO:
    """Group Relative Policy Optimization trainer for Memex.

    Key difference from PPO:
      - No critic/value function
      - Collects G episodes per scenario, normalizes rewards within the group
      - Advantage = (r_i - mean(r_group)) / std(r_group)
    """

    def __init__(self, model, tokenizer, config: GRPOConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device

        # Optimizer over LoRA parameters only
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_iterations, eta_min=config.lr * 0.1,
        )

    # ── Log Probability (response-only tokens) ──────────────────────

    def _compute_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
        """Compute mean per-token log probability for response tokens ONLY.

        Critical: we slice to response-only tokens to avoid polluting
        the gradient signal with prompt tokens.
        """
        full_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        resp_start = len(query_ids)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=full_ids).logits[0]
            # logits[t] predicts token t+1; take response positions only
            resp_logits = logits[resp_start - 1 : -1, :]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            token_lp = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
            return token_lp.mean()

    def _compute_ref_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor,
    ) -> float:
        """Compute reference log prob using the frozen base (LoRA disabled)."""
        self.model.eval()
        try:
            with self.model.disable_adapter():
                lp = self._compute_log_prob(query_ids, response_ids, with_grad=False)
        except AttributeError:
            lp = self._compute_log_prob(query_ids, response_ids, with_grad=False)
        self.model.train()
        return lp.item()

    # ── Generation ───────────────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate a response and return (text, query_ids, response_ids)."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
        ).to(self.device)

        query_ids = inputs["input_ids"].squeeze(0)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                max_length=None,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = out[0][len(query_ids):]

        if len(response_ids) == 0:
            response_ids = torch.tensor([self.tokenizer.eos_token_id])

        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return text, query_ids.cpu(), response_ids.cpu()

    # ── Episode Rollout ──────────────────────────────────────────────

    def rollout(
        self, env: AMLEnvironment, difficulty: str, typology: str,
        seed: Optional[int] = None,
    ) -> Tuple[List[StepData], EpisodeStats]:
        """Collect one full episode trajectory (no gradients).

        Args:
            seed: If provided, seeds the environment RNG so G rollouts
                  in the same group see the same generated scenario.
        """
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        obs = env.reset(task_id=difficulty, seed=seed)
        steps: List[StepData] = []

        for step_num in range(1, self.cfg.max_steps + 1):
            ram = env._sm.ram_contents if env._sm else []
            disk = env._sm.disk_contents if env._sm else []
            kernel = env._sm.kernel_directives if env._sm else []

            prompt = format_prompt(obs, step_num, kernel, disk, ram)
            resp_text, q_ids, r_ids = self.generate(prompt)

            # Degenerate response detection
            if len(r_ids) > 4:
                unique_ratio = len(set(r_ids.tolist())) / len(r_ids)
                if unique_ratio < 0.20:
                    reward = -0.15
                    steps.append(StepData(
                        query_ids=q_ids, response_ids=r_ids,
                        response_start=len(q_ids),
                        reward=reward,
                        old_log_prob=self._compute_log_prob(q_ids, r_ids, with_grad=False).item(),
                        ref_log_prob=self._compute_ref_log_prob(q_ids, r_ids),
                        response_text=resp_text,
                    ))
                    break

            old_lp = self._compute_log_prob(q_ids, r_ids, with_grad=False).item()
            ref_lp = self._compute_ref_log_prob(q_ids, r_ids)

            tool, params = parse_action(resp_text)
            try:
                obs = env.step(AMLAction(tool=tool, parameters=params))
                reward = obs.reward if obs.reward is not None else 0.0
            except Exception as e:
                reward = -0.10
                obs = AMLObservation(
                    tool_result={"error": f"Invalid action: {str(e)[:100]}"},
                    available_tools=[],
                    message=f"Action failed: {str(e)[:100]}. Produce valid JSON.",
                    done=True, reward=reward,
                    metadata={"step": step_num, "error": "malformed_action"},
                )

            steps.append(StepData(
                query_ids=q_ids, response_ids=r_ids,
                response_start=len(q_ids),
                reward=reward, old_log_prob=old_lp, ref_log_prob=ref_lp,
                response_text=resp_text,
            ))

            if obs.done:
                # De-duplicate: terminal obs.reward from file_sar/close_alert
                # includes accumulated per-step rewards (added by the grader).
                # Subtract prior steps to avoid double-counting in returns.
                if len(steps) > 1:
                    prior_step_reward_sum = sum(s.reward for s in steps[:-1])
                    steps[-1].reward = reward - prior_step_reward_sum
                break

        st = env._state
        stats = EpisodeStats(
            score=st.accumulated_reward, steps=len(steps),
            difficulty=difficulty, typology=typology,
            page_faults=st.page_fault_count, async_timeouts=st.async_timeout_count,
            successful_pages=st.successful_pages, meta_injections=st.meta_injections,
        )
        return steps, stats

    # ── GRPO Update ──────────────────────────────────────────────────

    def grpo_update(self, group_trajectories: List[Tuple[List[StepData], float]]) -> Dict[str, float]:
        """Run GRPO update on a group of G episode trajectories.

        Args:
            group_trajectories: List of (steps, episode_score) tuples, length G.

        Returns:
            Dict of training metrics.
        """
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(self.model)

        G = len(group_trajectories)
        if G == 0:
            return {"grpo/loss": 0.0, "grpo/kl": 0.0, "grpo/entropy": 0.0, "grpo/steps_trained": 0}

        # ── Per-step discounted returns, normalized across group ──
        scores = torch.tensor([s for _, s in group_trajectories], dtype=torch.float32)
        for g_idx, (steps, _) in enumerate(group_trajectories):
            T = len(steps)
            G_val = 0.0
            for t in reversed(range(T)):
                G_val = steps[t].reward + 0.99 * G_val
                steps[t].advantage = G_val  # temporarily store return

        # Normalize returns across all steps from all G trajectories
        all_returns = []
        for steps, _ in group_trajectories:
            all_returns.extend([s.advantage for s in steps])
        ret_mean = sum(all_returns) / max(len(all_returns), 1)
        ret_std = max((sum((r - ret_mean)**2 for r in all_returns) / max(len(all_returns), 1))**0.5, 1e-8)

        for steps, _ in group_trajectories:
            for s in steps:
                s.advantage = (s.advantage - ret_mean) / ret_std

        total_loss_val = 0.0
        total_kl_val = 0.0
        total_entropy_val = 0.0
        n_updates = 0

        self.optimizer.zero_grad()

        for g_idx, (steps, _) in enumerate(group_trajectories):
            for i, step in enumerate(steps):
                try:
                    # Current policy log prob (WITH gradients)
                    full_ids = torch.cat([step.query_ids, step.response_ids]).unsqueeze(0).to(self.device)
                    resp_start = len(step.query_ids)

                    logits = self.model(input_ids=full_ids).logits[0]
                    resp_logits = logits[resp_start - 1 : -1, :]
                    log_probs = F.log_softmax(resp_logits, dim=-1)
                    token_lp = log_probs.gather(
                        1, step.response_ids.to(self.device).unsqueeze(1),
                    ).squeeze(1)
                    new_lp = token_lp.mean()

                    # Entropy bonus
                    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

                    # DEFICIENCY-1: Raw REINFORCE — no importance ratio clipping.
                    # PPO uses clamp(ratio, 1-eps, 1+eps) * advantage for trust-region safety.
                    # This is vulnerable to large, destabilizing policy updates.
                    adv_tensor = torch.tensor(step.advantage, device=self.device)
                    policy_loss = -(new_lp * adv_tensor)

                    # DEFICIENCY-2: Uses |KL| instead of proper forward KL.
                    # This penalizes the policy equally for moving toward or away from
                    # the reference distribution — incorrect directionality.
                    kl = (new_lp - step.ref_log_prob).clamp(-10, 10)
                    kl_loss = self.cfg.kl_coef * kl.abs()

                    # DEFICIENCY-3: Loss divided by total_steps couples gradient magnitude
                    # to group size. A group of 8 gets half the gradient of a group of 4.
                    total_steps = sum(len(s) for s, _ in group_trajectories)
                    loss = (policy_loss + kl_loss - self.cfg.entropy_coef * entropy) / max(total_steps, 1)
                    loss.backward()

                    total_loss_val += policy_loss.item()
                    total_kl_val += kl.item()
                    total_entropy_val += entropy.item()
                    n_updates += 1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ⚠ OOM on group {g_idx} step {i}, skipping")
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        n = max(n_updates, 1)
        return {
            "grpo/loss": total_loss_val / n,
            "grpo/kl": total_kl_val / n,
            "grpo/entropy": total_entropy_val / n,
            "grpo/steps_trained": n_updates,
            "grpo/lr": self.scheduler.get_last_lr()[0],
            "grpo/group_score_std": scores.std().item(),
        }

    # ── Checkpoint ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 Saved: {path}")

    # ── Stable Checkpoint (Auto-Revert) ─────────────────────────────────

    def save_stable_state(self, path: str) -> None:
        """Save LoRA weights + optimizer state as 'last known good'."""
        os.makedirs(path, exist_ok=True)
        lora_state = {
            k: v.cpu().clone()
            for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(lora_state, os.path.join(path, "lora_state.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optim_state.pt"))
        torch.save({
            "entropy_coef": self.cfg.entropy_coef,
            "temperature": self.cfg.temperature,
            "lr": self.cfg.lr,
        }, os.path.join(path, "hyperparams.pt"))

    def load_stable_state(self, path: str) -> bool:
        """Reload LoRA weights from stable checkpoint. Returns True on success."""
        lora_path = os.path.join(path, "lora_state.pt")
        if not os.path.exists(lora_path):
            return False
        state = torch.load(lora_path, map_location=self.device, weights_only=True)
        for name, param in self.model.named_parameters():
            if name in state:
                param.data.copy_(state[name].to(self.device))
        return True

    def rebuild_optimizer(self, remaining_iters: int = 0) -> None:
        """Recreate optimizer + scheduler with current config (after revert)."""
        t_max = remaining_iters if remaining_iters > 0 else self.cfg.total_iterations
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=self.cfg.lr * 0.1,
        )


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: GRPOConfig) -> None:
    banner = f"""
{'═'*60}
  MEMEX GRPO TRAINER  —  L4-Optimized
  Model:  {cfg.model_name}
  LoRA:   r={cfg.lora_r}  α={cfg.lora_alpha}
  LR:     {cfg.lr}  |  KL: {cfg.kl_coef}  |  Group: {cfg.group_size}
  Iters:  {cfg.total_iterations}  ×  {cfg.episodes_per_iter} ep/iter
  PLR:    {cfg.use_plr}
  Dry:    {cfg.dry_run}
{'═'*60}"""
    print(banner)

    # ── 1. Load Model ──
    print("[1/4] Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Loaded  |  VRAM: {vram_status()}")

    # ── 2. Attach LoRA ──
    print("[2/4] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets,
        bias="none", use_gradient_checkpointing="unsloth", random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  ✓ LoRA: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)")
    print(f"  VRAM: {vram_status()}")

    # ── 3. WandB ──
    if not cfg.dry_run:
        print("[3/4] Initializing WandB...")
        import wandb
        wandb.init(project=cfg.wandb_project, name=f"memex-grpo-{int(time.time())}",
                   config=vars(cfg))
        print("  ✓ WandB ready")
    else:
        print("[3/4] WandB SKIPPED (dry-run)")

    # ── 4. Train ──
    print("[4/4] Starting GRPO training loop...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grpo = MemexGRPO(model, tokenizer, cfg, device)
    env = AMLEnvironment()

    # ── PLR Curriculum Engine ──
    plr = None
    if cfg.use_plr:
        from curriculum.plr_engine import PLREngine
        plr = PLREngine(buffer_size=cfg.plr_buffer_size)
        print(f"  ✓ PLR Curriculum enabled (buffer={cfg.plr_buffer_size})")

    iters = 2 if cfg.dry_run else cfg.total_iterations
    eps_per = 1 if cfg.dry_run else cfg.episodes_per_iter
    best_score = -float("inf")

    # ── Auto-Revert State ──
    stable_path = os.path.join(cfg.output_dir, "_stable_grpo")
    stable_saved = False
    low_entropy_streak = 0
    low_score_streak = 0
    revert_count = 0
    max_reverts = 5

    for it in range(1, iters + 1):
        t0 = time.time()
        iter_scores = []

        # ── Collect trajectories for each episode ──
        for ep in range(eps_per):
            # PLR curriculum: adaptive scenario selection
            if plr is not None:
                diff, typo = plr.sample_scenario(cfg.difficulties, cfg.typologies)
            else:
                diff = random.choice(cfg.difficulties)
                typo = random.choice(cfg.typologies)

            # ── Collect G rollouts for GRPO group (same scenario via seed) ──
            group_trajectories: List[Tuple[List[StepData], float]] = []
            group_scores = []
            scenario_seed = random.randint(0, 2**31)  # shared seed for all G rollouts

            for g in range(cfg.group_size):
                try:
                    steps, stats = grpo.rollout(env, diff, typo, seed=scenario_seed)
                    group_trajectories.append((steps, stats.score))
                    group_scores.append(stats.score)
                except Exception as e:
                    print(f"  ⚠ Group rollout {g} failed: {e}")
                    continue

            if not group_trajectories:
                print(f"  ⚠ All group rollouts failed for ep {ep}, skipping")
                continue

            # ── GRPO Update on this group ──
            grpo_stats = grpo.grpo_update(group_trajectories)

            # ── PLR update ──
            best_group_score = max(group_scores)
            if plr is not None:
                scenario_id = f"iter{it}_ep{ep}_{diff}_{typo}"
                plr.update(scenario_id, diff, typo, best_group_score)

            iter_scores.append(best_group_score)

            plr_tag = " [PLR]" if plr is not None else ""
            print(
                f"  It {it:>3} Ep {ep+1} | {diff}/{typo}{plr_tag} | "
                f"G={len(group_trajectories)} best={best_group_score:+.3f} | "
                f"loss={grpo_stats['grpo/loss']:.4f} kl={grpo_stats['grpo/kl']:.4f}"
            )

        if not iter_scores:
            print(f"  ⚠ No scores collected, skipping iteration {it}")
            continue

        # ── Stats ──
        mean_score = sum(iter_scores) / len(iter_scores)
        elapsed = time.time() - t0

        print(
            f"\n  ═══ Iter {it}/{iters} ═══\n"
            f"    Mean score:  {mean_score:+.4f}\n"
            f"    GRPO loss:   {grpo_stats['grpo/loss']:.6f}\n"
            f"    KL:          {grpo_stats['grpo/kl']:.6f}\n"
            f"    Entropy:     {grpo_stats['grpo/entropy']:.4f}\n"
            f"    VRAM: {vram_status()}  |  {elapsed:.0f}s\n"
        )

        # ── Entropy Heartbeat Monitor (Auto-Revert) ──────────────────
        entropy = grpo_stats["grpo/entropy"]
        kl = grpo_stats["grpo/kl"]

        if entropy < 0.01:
            low_entropy_streak += 1
        else:
            low_entropy_streak = 0

        if mean_score <= 0.0:
            low_score_streak += 1
        else:
            low_score_streak = 0

        # Save stable checkpoint when training is healthy
        if entropy > 0.05 and mean_score > 0.3:
            grpo.save_stable_state(stable_path)
            stable_saved = True

        # Check for collapse
        is_collapse = (
            low_entropy_streak >= 2
            or low_score_streak >= 2
            or abs(kl) > 10.0
        )

        if is_collapse and stable_saved and revert_count < max_reverts:
            revert_count += 1
            print(
                f"\n  ⚠️  COLLAPSE DETECTED (revert #{revert_count})  ⚠️\n"
                f"    Trigger: entropy={entropy:.4f} | KL={kl:.4f} | "
                f"score={mean_score:+.4f}\n"
                f"    Purging VRAM and reloading stable checkpoint..."
            )

            torch.cuda.empty_cache()
            gc.collect()

            success = grpo.load_stable_state(stable_path)
            if not success:
                print("    ❌ Failed to load stable state, continuing...")
            else:
                old_ec = cfg.entropy_coef
                old_temp = cfg.temperature
                old_lr = cfg.lr
                cfg.entropy_coef = min(cfg.entropy_coef * 1.5, 0.20)
                cfg.temperature = min(cfg.temperature + 0.1, 0.9)
                cfg.lr = cfg.lr * 0.7
                grpo.cfg = cfg

                del grpo.optimizer, grpo.scheduler
                gc.collect()
                torch.cuda.empty_cache()

                grpo.rebuild_optimizer(remaining_iters=iters - it)

                print(
                    f"    ✅ Reverted successfully.\n"
                    f"    entropy_coef: {old_ec:.4f} → {cfg.entropy_coef:.4f}\n"
                    f"    temperature:  {old_temp:.2f} → {cfg.temperature:.2f}\n"
                    f"    lr:           {old_lr:.2e} → {cfg.lr:.2e}\n"
                )

            low_entropy_streak = 0
            low_score_streak = 0

            if not cfg.dry_run:
                import wandb
                wandb.log({
                    "revert/count": revert_count,
                    "revert/entropy_coef": cfg.entropy_coef,
                    "revert/temperature": cfg.temperature,
                    "revert/lr": cfg.lr,
                })

            gc.collect()
            torch.cuda.empty_cache()
            continue

        # ── WandB ──
        if not cfg.dry_run:
            import wandb
            log_dict = {
                "iteration": it,
                "grpo/returns/mean": mean_score,
                **grpo_stats,
                "perf/iter_seconds": elapsed,
                "perf/vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "stability/entropy_coef": cfg.entropy_coef,
                "stability/temperature": cfg.temperature,
                "stability/revert_count": revert_count,
            }
            if plr is not None:
                log_dict.update(plr.get_wandb_metrics())
            wandb.log(log_dict)

        # ── Checkpoint ──
        if mean_score > best_score:
            best_score = mean_score
            grpo.save(os.path.join(cfg.output_dir, "best"))
            if plr is not None:
                plr.save(os.path.join(cfg.output_dir, "best", "plr_buffer.json"))
        if it % cfg.save_every == 0:
            grpo.save(os.path.join(cfg.output_dir, f"iter-{it}"))

        # ── GC ──
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final save ──
    grpo.save(os.path.join(cfg.output_dir, "final"))
    print(f"\n{'═'*60}")
    print(f"  GRPO TRAINING COMPLETE  |  Best: {best_score:+.4f}")
    print(f"{'═'*60}\n")

    if not cfg.dry_run:
        import wandb
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Memex GRPO Trainer (EXPERIMENTAL — use train_ppo.py for production)",
    )
    p.add_argument("--experimental", action="store_true",
                   help="Required flag to acknowledge GRPO is experimental")
    p.add_argument("--model", default=PPOConfig.model_name)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=PPOConfig.lora_r)
    p.add_argument("--episodes", type=int, default=PPOConfig.episodes_per_iter)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=PPOConfig.temperature)
    p.add_argument("--wandb-project", default="memex-grpo")
    p.add_argument("--output-dir", default=PPOConfig.output_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-plr", action="store_true", default=False,
                   help="Enable PLR curriculum (adaptive scenario sampling)")
    args = p.parse_args()

    if not args.experimental:
        print("\n" + "=" * 60)
        print("  WARNING: GRPO IS EXPERIMENTAL — NOT PRODUCTION-READY")
        print("  Use train_ppo.py for production training.")
        print("  To proceed anyway, re-run with: --experimental")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  GRPO EXPERIMENTAL MODE")
    print("  Known deficiencies: no clipping, |KL|, gradient coupling.")
    print("  Results are for ablation/comparison only.")
    print("=" * 60 + "\n")

    train(GRPOConfig(
        model_name=args.model, lr=args.lr, lora_r=args.lora_r,
        episodes_per_iter=args.episodes, total_iterations=args.iterations,
        temperature=args.temperature, group_size=args.group_size,
        wandb_project=args.wandb_project, output_dir=args.output_dir,
        dry_run=args.dry_run, use_plr=args.use_plr,
    ))


if __name__ == "__main__":
    main()
