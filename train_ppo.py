#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — PPO Training (T4-Optimized).

Custom step-level PPO with clipped surrogate objective and KL penalty
against the frozen base model. Designed for NVIDIA T4 (15 GB VRAM).

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Unsloth 4-bit LLM + LoRA adapters  (~5 GB)            │
  │  ┌─────────────────────┐  ┌────────────────────────┐   │
  │  │ Trajectory Rollout  │→ │ PPO Clipped Surrogate  │   │
  │  │ (inference, no grad)│  │ + KL vs frozen base    │   │
  │  └─────────────────────┘  └────────────────────────┘   │
  │              ↑                       ↓                  │
  │       AMLEnvironment            LoRA weight update      │
  │    (procedural scenarios)    (AdamW, grad accum)        │
  └─────────────────────────────────────────────────────────┘

Usage:
  python train_ppo.py --dry-run              # quick 2-iter test
  python train_ppo.py --iterations 50        # real training
  python train_ppo.py --eval checkpoints/best
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


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    # ── Model ──
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048       # keep low for T4
    load_in_4bit: bool = True

    # ── LoRA ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── PPO ──
    lr: float = 5e-6
    ppo_epochs: int = 4              # epochs per PPO update
    clip_eps: float = 0.2            # clipping epsilon
    kl_coef: float = 0.05            # KL penalty coefficient
    entropy_coef: float = 0.01       # entropy bonus (prevents mode collapse)
    gamma: float = 0.99              # discount factor
    reward_clip: float = 2.0         # clip returns to [-clip, +clip]
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 4        # gradient accumulation

    # ── Generation ──
    max_new_tokens: int = 192        # JSON tool calls are short
    temperature: float = 0.3         # low for deterministic JSON
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # ── Environment ──
    episodes_per_iter: int = 4
    total_iterations: int = 50
    max_steps: int = 25
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])
    typologies: list = field(default_factory=lambda: [
        "structuring", "layering", "trade_based_ml",
    ])

    # ── Logging / Checkpointing ──
    wandb_project: str = "memex-ppo"
    save_every: int = 10
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    dry_run: bool = False


# ═══════════════════════════════════════════════════════════════════════
# System Prompt (condensed for token efficiency)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Senior AML Compliance Investigator in a Memex OS-Agent Environment.

OUTPUT: Respond with EXACTLY ONE raw JSON object. No markdown.
{"tool": "<name>", "parameters": {<params>}, "reasoning": "<1 sentence>"}

OS RULES:
- RAM holds LAST 2 observations. Use write_to_case_file to persist data.
- request_wire_trace is ASYNC (wait for ETA). retrieve_async_result when ready.
- search_compliance_manual + update_system_prompt injects rules (+0.15 reward).

TOOLS: review_alert, get_customer_profile, query_transactions, check_watchlist,
trace_network, check_source_of_funds, assess_risk, check_market_price,
write_to_case_file, request_wire_trace, retrieve_async_result,
search_compliance_manual, update_system_prompt, file_sar, close_alert

TERMINAL: file_sar(typology, entities_involved, findings) | close_alert(reason)
Typologies: structuring | layering | trade_based_ml | false_positive"""


# ═══════════════════════════════════════════════════════════════════════
# Prompt Formatting & JSON Parsing
# ═══════════════════════════════════════════════════════════════════════

def format_prompt(
    obs: AMLObservation,
    step: int,
    kernel: List[str],
    disk: List[str],
    ram: List[str],
) -> str:
    """Build Llama-3.1 chat-template prompt from environment state."""
    sys_msg = SYSTEM_PROMPT
    if kernel and len(kernel) > 1:
        sys_msg += "\n\nKERNEL DIRECTIVES:\n" + "\n".join(f"- {d}" for d in kernel)

    user_parts = []
    if disk:
        user_parts.append("CASE FILE (Disk):\n" + "\n".join(f"  {i}. {e}" for i, e in enumerate(disk, 1)))
    if ram:
        for r in ram[-2:]:
            user_parts.append(f"[RAM] {r[:300]}")

    obs_data = obs.tool_result if obs.tool_result else {}
    user_parts.append(f"STEP {step}:\n{json.dumps(obs_data, indent=1, default=str)[:1200]}")
    if obs.message:
        user_parts.append(obs.message[:200])

    user_msg = "\n".join(user_parts)

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_action(text: str) -> Tuple[str, Dict[str, Any]]:
    """Robust JSON tool-call parser with multi-tier fallback."""
    text = re.sub(r"```(?:json)?\s*", "", text.strip()).strip("` \n")
    # Tier 1: full parse
    try:
        d = json.loads(text)
        return d.get("tool", "review_alert"), d.get("parameters", {})
    except json.JSONDecodeError:
        pass
    # Tier 2: regex extract
    m = re.search(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return d.get("tool", "review_alert"), d.get("parameters", {})
        except json.JSONDecodeError:
            pass
    # Tier 3: tool name only
    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    return (tm.group(1) if tm else "review_alert"), {}


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StepData:
    """Single step in an episode trajectory."""
    query_ids: torch.Tensor        # tokenized prompt
    response_ids: torch.Tensor     # tokenized response only
    response_start: int            # index where response begins in full seq
    reward: float
    old_log_prob: float            # log π_old(response | query)
    ref_log_prob: float            # log π_ref(response | query)  [frozen base]
    advantage: float = 0.0
    response_text: str = ""


@dataclass
class EpisodeStats:
    """Summary statistics for one episode."""
    score: float
    steps: int
    difficulty: str
    typology: str
    page_faults: int = 0
    async_timeouts: int = 0
    successful_pages: int = 0
    meta_injections: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Core PPO Trainer
# ═══════════════════════════════════════════════════════════════════════

class MemexPPO:
    """Custom step-level PPO trainer for multi-step LLM environments.

    Key design choices for T4 (15 GB VRAM):
      - Process steps ONE AT A TIME during PPO update (no batching)
      - KL reference via model.disable_adapter() — no second model copy
      - Precompute ref_log_prob during collection to save a fwd pass
      - Gradient accumulation to simulate larger effective batch
    """

    def __init__(self, model, tokenizer, config: PPOConfig, device: str = "cuda"):
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

        # Running reward baseline (exponential moving average)
        self._reward_ema = 0.0
        self._reward_ema_alpha = 0.1

    # ── Log Probability ──────────────────────────────────────────────

    def _compute_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
        """Compute mean per-token log probability for response tokens."""
        full_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        resp_start = len(query_ids)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=full_ids).logits[0]
            # logits[t] predicts token t+1; take response positions
            resp_logits = logits[resp_start - 1 : -1, :]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            token_lp = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
            return token_lp.mean()  # mean, NOT sum — keeps KL scale-invariant across response lengths

    def _compute_ref_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor,
    ) -> float:
        """Compute reference log prob using the frozen base (LoRA disabled)."""
        self.model.eval()
        try:
            with self.model.disable_adapter():
                lp = self._compute_log_prob(query_ids, response_ids, with_grad=False)
        except AttributeError:
            # Fallback if disable_adapter unavailable
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
                max_length=None,  # suppress max_length vs max_new_tokens warning
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = out[0][len(query_ids):]

        # Guard: if model emits 0 new tokens, return a dummy to avoid NaN
        if len(response_ids) == 0:
            response_ids = torch.tensor([self.tokenizer.eos_token_id])

        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return text, query_ids.cpu(), response_ids.cpu()

    # ── Episode Rollout ──────────────────────────────────────────────

    def rollout(
        self, env: AMLEnvironment, difficulty: str, typology: str,
    ) -> Tuple[List[StepData], EpisodeStats]:
        """Collect one full episode trajectory (no gradients)."""
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        obs = env.reset(task_id=difficulty)
        steps: List[StepData] = []

        for step_num in range(1, self.cfg.max_steps + 1):
            ram = env._sm.ram_contents if env._sm else []
            disk = env._sm.disk_contents if env._sm else []
            kernel = env._sm.kernel_directives if env._sm else []

            prompt = format_prompt(obs, step_num, kernel, disk, ram)
            resp_text, q_ids, r_ids = self.generate(prompt)

            # Compute old & ref log probs during collection (saves fwd pass later)
            old_lp = self._compute_log_prob(q_ids, r_ids, with_grad=False).item()
            ref_lp = self._compute_ref_log_prob(q_ids, r_ids)

            # Parse and step environment
            tool, params = parse_action(resp_text)
            obs = env.step(AMLAction(tool=tool, parameters=params))
            reward = obs.reward if obs.reward is not None else 0.0

            steps.append(StepData(
                query_ids=q_ids, response_ids=r_ids,
                response_start=len(q_ids),
                reward=reward, old_log_prob=old_lp, ref_log_prob=ref_lp,
                response_text=resp_text,
            ))

            if obs.done:
                break

        # Compute discounted returns and advantages
        self._compute_advantages(steps)

        st = env._state
        stats = EpisodeStats(
            score=st.accumulated_reward, steps=len(steps),
            difficulty=difficulty, typology=typology,
            page_faults=st.page_fault_count, async_timeouts=st.async_timeout_count,
            successful_pages=st.successful_pages, meta_injections=st.meta_injections,
        )
        return steps, stats

    def _compute_advantages(self, steps: List[StepData]) -> None:
        """Compute discounted returns and normalized advantages."""
        T = len(steps)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = steps[t].reward + self.cfg.gamma * G
            returns[t] = G

        # Clip returns to prevent outlier gradient signals
        clip = self.cfg.reward_clip
        returns = [max(-clip, min(clip, r)) for r in returns]

        # Update EMA baseline
        mean_return = sum(returns) / T
        self._reward_ema = (
            self._reward_ema_alpha * mean_return
            + (1 - self._reward_ema_alpha) * self._reward_ema
        )

        # Advantages = returns - baseline, then normalize
        advs = [r - self._reward_ema for r in returns]
        std = max((sum(a**2 for a in advs) / T) ** 0.5, 1e-8)
        for t in range(T):
            steps[t].advantage = advs[t] / std

    # ── PPO Update ───────────────────────────────────────────────────

    def ppo_update(self, all_steps: List[StepData]) -> Dict[str, float]:
        """Run PPO epochs on collected trajectory steps.

        Processes steps ONE AT A TIME for minimal VRAM usage.
        Uses gradient accumulation to simulate larger batch.
        """
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(self.model)

        total_policy_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.ppo_epochs):
            random.shuffle(all_steps)
            self.optimizer.zero_grad()

            for i, step in enumerate(all_steps):
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

                    # Entropy bonus (prevents mode collapse)
                    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

                    # PPO clipped surrogate
                    log_ratio = new_lp - step.old_log_prob
                    log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # prevent exp overflow
                    ratio = torch.exp(log_ratio)
                    adv = torch.tensor(step.advantage, device=self.device)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # KL penalty against frozen base
                    kl = new_lp - step.ref_log_prob
                    kl_loss = self.cfg.kl_coef * kl

                    loss = (policy_loss + kl_loss - self.cfg.entropy_coef * entropy) / self.cfg.grad_accum_steps
                    loss.backward()

                    total_policy_loss += policy_loss.item()
                    total_kl += kl.item()
                    total_entropy += entropy.item()
                    n_updates += 1

                    # Gradient accumulation step
                    if (i + 1) % self.cfg.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.cfg.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ⚠ OOM on step {i}, skipping")
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise

            # Flush remaining gradients
            if len(all_steps) % self.cfg.grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

        n = max(n_updates, 1)
        return {
            "ppo/loss/policy": total_policy_loss / n,
            "ppo/kl": total_kl / n,
            "ppo/entropy": total_entropy / n,
            "ppo/steps_trained": n_updates,
            "ppo/lr": self.scheduler.get_last_lr()[0],
        }

    # ── Checkpoint ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# VRAM Monitoring
# ═══════════════════════════════════════════════════════════════════════

def vram_status() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{alloc:.1f}/{total:.1f} GB"


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: PPOConfig) -> None:
    banner = f"""
{'═'*60}
  MEMEX PPO TRAINER  —  T4-Optimized
  Model:  {cfg.model_name}
  LoRA:   r={cfg.lora_r}  α={cfg.lora_alpha}
  LR:     {cfg.lr}  |  Clip: {cfg.clip_eps}  |  KL: {cfg.kl_coef}
  Iters:  {cfg.total_iterations}  ×  {cfg.episodes_per_iter} ep/iter
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
        wandb.init(project=cfg.wandb_project, name=f"memex-{int(time.time())}",
                   config=vars(cfg))
        print("  ✓ WandB ready")
    else:
        print("[3/4] WandB SKIPPED (dry-run)")

    # ── 4. Train ──
    print("[4/4] Starting training loop...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo = MemexPPO(model, tokenizer, cfg, device)
    env = AMLEnvironment()

    iters = 2 if cfg.dry_run else cfg.total_iterations
    eps_per = 1 if cfg.dry_run else cfg.episodes_per_iter
    best_score = -float("inf")

    for it in range(1, iters + 1):
        t0 = time.time()
        all_steps: List[StepData] = []
        ep_stats: List[EpisodeStats] = []

        # ── Collect trajectories ──
        for ep in range(eps_per):
            diff = random.choice(cfg.difficulties)
            typo = random.choice(cfg.typologies)

            try:
                steps, stats = ppo.rollout(env, diff, typo)
                all_steps.extend(steps)
                ep_stats.append(stats)

                print(
                    f"  It {it:>3} Ep {ep+1} | {diff}/{typo} | "
                    f"steps={stats.steps:>2} score={stats.score:+.3f} | "
                    f"PF={stats.page_faults} AT={stats.async_timeouts} "
                    f"SP={stats.successful_pages} MI={stats.meta_injections}"
                )
                # Show a sample response from the first step
                if ep == 0 and steps:
                    print(f"    [sample] {steps[0].response_text[:120]}...")

            except Exception as e:
                print(f"  ⚠ Episode failed: {e}")
                continue

        if not all_steps:
            print(f"  ⚠ No steps collected, skipping iteration {it}")
            continue

        # ── PPO Update ──
        ppo_stats = ppo.ppo_update(all_steps)

        # ── Stats ──
        mean_score = sum(s.score for s in ep_stats) / len(ep_stats)
        total_pf = sum(s.page_faults for s in ep_stats)
        total_at = sum(s.async_timeouts for s in ep_stats)
        total_sp = sum(s.successful_pages for s in ep_stats)
        total_mi = sum(s.meta_injections for s in ep_stats)
        elapsed = time.time() - t0

        print(
            f"\n  ═══ Iter {it}/{iters} ═══\n"
            f"    Mean score:  {mean_score:+.4f}\n"
            f"    PPO loss:    {ppo_stats['ppo/loss/policy']:.6f}\n"
            f"    KL:          {ppo_stats['ppo/kl']:.6f}\n"
            f"    Entropy:     {ppo_stats['ppo/entropy']:.4f}\n"
            f"    OS: PF={total_pf} AT={total_at} SP={total_sp} MI={total_mi}\n"
            f"    VRAM: {vram_status()}  |  {elapsed:.0f}s\n"
        )

        # ── WandB ──
        if not cfg.dry_run:
            import wandb
            wandb.log({
                "iteration": it,
                "ppo/returns/mean": mean_score,
                **ppo_stats,
                "os/page_faults": total_pf,
                "os/async_timeouts": total_at,
                "os/successful_pages": total_sp,
                "os/meta_injections": total_mi,
                "perf/iter_seconds": elapsed,
                "perf/vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            })

        # ── Checkpoint ──
        if mean_score > best_score:
            best_score = mean_score
            ppo.save(os.path.join(cfg.output_dir, "best"))
        if it % cfg.save_every == 0:
            ppo.save(os.path.join(cfg.output_dir, f"iter-{it}"))

        # ── GC ──
        del all_steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final save ──
    ppo.save(os.path.join(cfg.output_dir, "final"))
    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE  |  Best: {best_score:+.4f}")
    print(f"{'═'*60}\n")

    if not cfg.dry_run:
        import wandb
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model_path: str):
    """Evaluate a checkpoint across all 9 difficulty×typology combos."""
    from unsloth import FastLanguageModel

    print(f"\n{'═'*60}\n  EVALUATION: {model_path}\n{'═'*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=2048, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    cfg = PPOConfig()
    ppo = MemexPPO(model, tokenizer, cfg)
    env = AMLEnvironment()

    scores = []
    for diff in ["easy", "medium", "hard"]:
        for typo in ["structuring", "layering", "trade_based_ml"]:
            _, stats = ppo.rollout(env, diff, typo)
            scores.append(stats.score)
            print(
                f"  {diff:>6}/{typo:<15} | steps={stats.steps:>2} | "
                f"score={stats.score:+.4f} | "
                f"PF={stats.page_faults} SP={stats.successful_pages} MI={stats.meta_injections}"
            )

    print(f"\n  Mean: {sum(scores)/len(scores):+.4f}  "
          f"Min: {min(scores):+.4f}  Max: {max(scores):+.4f}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex PPO Trainer")
    p.add_argument("--model", default=PPOConfig.model_name)
    p.add_argument("--lr", type=float, default=PPOConfig.lr)
    p.add_argument("--lora-r", type=int, default=PPOConfig.lora_r)
    p.add_argument("--episodes", type=int, default=PPOConfig.episodes_per_iter)
    p.add_argument("--iterations", type=int, default=PPOConfig.total_iterations)
    p.add_argument("--temperature", type=float, default=PPOConfig.temperature)
    p.add_argument("--wandb-project", default=PPOConfig.wandb_project)
    p.add_argument("--output-dir", default=PPOConfig.output_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval", type=str, default=None, metavar="PATH")
    args = p.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(PPOConfig(
            model_name=args.model, lr=args.lr, lora_r=args.lora_r,
            episodes_per_iter=args.episodes, total_iterations=args.iterations,
            temperature=args.temperature,
            wandb_project=args.wandb_project, output_dir=args.output_dir,
            dry_run=args.dry_run,
        ))


if __name__ == "__main__":
    main()
