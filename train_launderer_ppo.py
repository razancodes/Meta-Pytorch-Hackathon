"""
Memex OS-Agent Benchmark — Launderer PPO Trainer (Single-Step).

Trains the Launderer-8B agent using PPO to generate evasive AML scenarios.
Each episode is a single step: the Launderer generates a scenario JSON,
which is validated and scored against a frozen Defender checkpoint.

Design principles:
  - Single-step PPO: advantage = reward - batch_mean_reward
  - Token-level log-ratio clipped surrogate
  - KL penalty against frozen base model
  - Same VRAM-optimized patterns as train_ppo.py (one step at a time)
  - Compatible with Unsloth 4-bit LoRA on L4/T4

Usage:
    python train_launderer_ppo.py --dry-run
    python train_launderer_ppo.py --defender-checkpoint checkpoints/defender/best
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.launderer_env import LaundererEnv, LaundererObs
from models import AMLObservation


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LaundererPPOConfig:
    """Configuration for Launderer PPO training."""

    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 8192    # A100: 8192 (L4: use 4096)
    max_new_tokens: int = 2048    # Launderer generates full scenario JSON
    load_in_4bit: bool = False    # A100: BF16 is faster (L4: use True)

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # PPO
    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    entropy_coef: float = 0.02
    ppo_epochs: int = 4           # A100: 4 passes per batch (L4: use 2)
    grad_accum_steps: int = 8     # A100: 8 (L4: use 4)
    max_grad_norm: float = 1.0

    # Training loop
    total_iterations: int = 50
    episodes_per_iter: int = 8    # A100: 8 (L4: use 4)
    temperature: float = 0.95     # Higher than Defender: Launderer MUST explore diverse scenarios
    top_p: float = 0.90           # Slightly tighter nucleus to avoid gibberish at high temp
    repetition_penalty: float = 1.15  # Increased to prevent template repetition

    # Defender
    defender_checkpoint: str = ""  # Path to frozen Defender LoRA checkpoint
    defender_max_steps: int = 15

    # Scenario generation
    typologies: list = field(default_factory=lambda: [
        "structuring", "layering", "trade_based_ml",
    ])
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])

    # Infrastructure
    output_dir: str = "checkpoints/launderer"
    wandb_project: str = "memex-launderer"
    save_every: int = 10
    dry_run: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Data
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LaundererStepData:
    """Single trajectory step (one per episode for this one-step MDP)."""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    reward: float
    old_log_prob: float
    ref_log_prob: float
    advantage: float = 0.0
    response_text: str = ""
    typology: str = ""
    difficulty: str = ""
    is_valid: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Launderer PPO Trainer
# ═══════════════════════════════════════════════════════════════════════

class LaundererPPO:
    """PPO trainer for the single-step Launderer agent.

    Unlike the Defender's multi-step PPO, each episode produces exactly
    one (prompt, response) pair. Advantage is batch-normalized:
        advantage_i = reward_i - mean(rewards_in_batch)
    """

    def __init__(self, model, tokenizer, config: LaundererPPOConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_iterations, eta_min=config.lr * 0.1,
        )

    def reconnect_model(self, model, tokenizer, scheduler_step: int = 0):
        """Replace model/tokenizer after VRAM swap, preserving scheduler position.

        During Defender scoring, the Launderer model is unloaded and reloaded.
        This method reconnects without resetting the cosine LR schedule to
        step 0, which previously caused LR to collapse to eta_min every
        iteration.
        """
        self.model = model
        self.tokenizer = tokenizer

        # Must rebuild optimizer — parameter objects are new after model reload.
        # AdamW momentum state is lost, but scheduler position is preserved.
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.total_iterations, eta_min=self.cfg.lr * 0.1,
        )
        # Fast-forward scheduler to pre-swap position
        for _ in range(scheduler_step):
            self.scheduler.step()

    # ── Log Probability ──────────────────────────────────────────────

    def _compute_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
        """Mean per-token log probability for the response."""
        full_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        resp_start = len(query_ids)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=full_ids).logits[0]
            resp_logits = logits[resp_start - 1 : -1, :]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            token_lp = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
            return token_lp.mean()

    def _compute_ref_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor,
    ) -> float:
        """Reference log prob with LoRA disabled."""
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
        """Generate a scenario and return (text, query_ids, response_ids)."""
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

    # ── Rollout (Single-Step) ────────────────────────────────────────

    def rollout(
        self, env: LaundererEnv, typology: str, difficulty: str,
    ) -> LaundererStepData:
        """One episode = one generation step."""
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        obs = env.reset(typology=typology, difficulty=difficulty)

        # Format prompt with Llama chat template
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You generate AML investigation scenarios as JSON objects."
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{obs.prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        resp_text, q_ids, r_ids = self.generate(prompt)

        # Compute log probs during collection
        old_lp = self._compute_log_prob(q_ids, r_ids, with_grad=False).item()
        ref_lp = self._compute_ref_log_prob(q_ids, r_ids)

        # Step environment (validates scenario, runs Defender)
        result = env.step(resp_text)

        # Check if scenario was valid (reward > -2.0)
        is_valid = result.reward > -1.5

        return LaundererStepData(
            query_ids=q_ids,
            response_ids=r_ids,
            reward=result.reward,
            old_log_prob=old_lp,
            ref_log_prob=ref_lp,
            response_text=resp_text,
            typology=typology,
            difficulty=difficulty,
            is_valid=is_valid,
        )

    # ── Compute Batch Advantages ─────────────────────────────────────

    @staticmethod
    def compute_batch_advantages(steps: List[LaundererStepData]) -> None:
        """Batch normalize advantages for single-step episodes."""
        rewards = [s.reward for s in steps]
        mean_r = sum(rewards) / max(len(rewards), 1)
        raw_advs = [r - mean_r for r in rewards]
        std = max((sum(a**2 for a in raw_advs) / max(len(raw_advs), 1)) ** 0.5, 1e-8)
        for i, step in enumerate(steps):
            step.advantage = raw_advs[i] / std

    # ── PPO Update ───────────────────────────────────────────────────

    def ppo_update(self, all_steps: List[LaundererStepData]) -> Dict[str, float]:
        """Run PPO epochs. Processes steps one at a time."""
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
                    full_ids = torch.cat([step.query_ids, step.response_ids]).unsqueeze(0).to(self.device)
                    resp_start = len(step.query_ids)

                    logits = self.model(input_ids=full_ids).logits[0]
                    resp_logits = logits[resp_start - 1 : -1, :]
                    log_probs = F.log_softmax(resp_logits, dim=-1)
                    token_lp = log_probs.gather(
                        1, step.response_ids.to(self.device).unsqueeze(1),
                    ).squeeze(1)
                    new_lp = token_lp.mean()

                    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

                    log_ratio = torch.clamp(new_lp - step.old_log_prob, -10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    adv = torch.tensor(step.advantage, device=self.device)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    kl = new_lp - step.ref_log_prob
                    kl_loss = self.cfg.kl_coef * kl.clamp(min=0)

                    loss = (policy_loss + kl_loss - self.cfg.entropy_coef * entropy) / self.cfg.grad_accum_steps
                    loss.backward()

                    total_policy_loss += policy_loss.item()
                    total_kl += abs(kl.item())
                    total_entropy += entropy.item()
                    n_updates += 1

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
            "launderer_ppo/loss/policy": total_policy_loss / n,
            "launderer_ppo/kl": total_kl / n,
            "launderer_ppo/entropy": total_entropy / n,
            "launderer_ppo/steps_trained": n_updates,
            "launderer_ppo/lr": self.scheduler.get_last_lr()[0],
        }

    # ── Checkpoint ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 Saved Launderer: {path}")


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

def train(cfg: LaundererPPOConfig) -> None:
    banner = f"""
{'═'*60}
  MEMEX LAUNDERER PPO TRAINER  —  Single-Step MDP
  Model:  {cfg.model_name}
  LoRA:   r={cfg.lora_r}  α={cfg.lora_alpha}
  LR:     {cfg.lr}  |  Clip: {cfg.clip_eps}  |  KL: {cfg.kl_coef}
  Iters:  {cfg.total_iterations}  ×  {cfg.episodes_per_iter} ep/iter
  Defender: {cfg.defender_checkpoint or 'NONE (dummy scoring)'}
  Dry:    {cfg.dry_run}
{'═'*60}"""
    print(banner)

    # ── 1. Load Model ──
    print("[1/4] Loading Launderer model with Unsloth...")
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

    # ── 3. WandB ──
    if not cfg.dry_run:
        print("[3/4] Initializing WandB...")
        import wandb
        wandb.init(project=cfg.wandb_project, name=f"launderer-{int(time.time())}",
                   config=vars(cfg))
    else:
        print("[3/4] WandB SKIPPED (dry-run)")

    # ── 4. Setup Environment ──
    # VRAM-SAFE MODEL SWAPPING STRATEGY: Full unload/reload.
    #
    # Why: On a 22 GB L4, we cannot have both agents loaded simultaneously.
    # LoRA adapter swapping was considered but rejected because:
    #   (a) Both agents need different LoRA adapters on the same base model.
    #   (b) PEFT adapter swapping can leak gradient state between agents.
    #   (c) Full unload/reload is simpler and guarantees zero shared state.
    #
    # How it works per iteration:
    #   1. Launderer generates batch of scenarios (model loaded)
    #   2. Save Launderer LoRA weights to temp checkpoint
    #   3. Unload Launderer entirely
    #   4. Load frozen Defender checkpoint
    #   5. Score all valid scenarios against Defender
    #   6. Unload Defender
    #   7. Reload Launderer from temp checkpoint
    #   8. Run PPO update on Launderer with real rewards
    #
    # For dry-run or when no Defender checkpoint exists, skip steps 2-7
    # and use dummy scoring (defender_score=0.0).
    print("[4/4] Initializing LaundererEnv...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo = LaundererPPO(model, tokenizer, cfg, device)

    has_defender = bool(cfg.defender_checkpoint and os.path.exists(cfg.defender_checkpoint))
    if has_defender:
        print(f"  ✓ Frozen Defender checkpoint found: {cfg.defender_checkpoint}")
        print(f"    Will use full unload/reload for VRAM-safe scoring")
    else:
        print(f"  ⚠ No Defender checkpoint — using dummy scoring (defender_score=0.0)")

    # Use dummy scoring initially — will be replaced per-iteration when Defender is available
    env = LaundererEnv(defender_rollout_fn=None)
    print(f"  ✓ LaundererEnv ready  |  VRAM: {vram_status()}\n")

    # ── Training Loop ──
    iters = 2 if cfg.dry_run else cfg.total_iterations
    eps_per = 1 if cfg.dry_run else cfg.episodes_per_iter
    best_reward = -float("inf")

    # KL auto-revert: save weights when mean return improves, revert if it worsens
    # by more than 30% over 3 consecutive iterations.
    # KL threshold: 0.3 per update is a reasonable bound for small batches.
    # Too low (e.g., 0.05) would stall learning; too high (e.g., 10+) allows policy collapse.
    KL_WARN_THRESHOLD = 0.3  # Log warning if KL exceeds this
    REVERT_DECLINE_THRESHOLD = 0.30  # 30% decline triggers revert
    REVERT_PATIENCE = 3  # Must decline for 3 consecutive iterations
    best_checkpoint_path = os.path.join(cfg.output_dir, "_best_revert")
    revert_decline_count = 0
    prev_mean_reward = None

    for it in range(1, iters + 1):
        t0 = time.time()
        batch_steps: List[LaundererStepData] = []

        for ep in range(eps_per):
            typo = random.choice(cfg.typologies)
            diff = random.choice(cfg.difficulties)

            try:
                step_data = ppo.rollout(env, typo, diff)
                batch_steps.append(step_data)

                valid_tag = "✓" if step_data.is_valid else "✗"
                print(
                    f"  It {it:>3} Ep {ep+1} | {diff}/{typo} | "
                    f"reward={step_data.reward:+.3f} [{valid_tag}] | "
                    f"resp_len={len(step_data.response_ids)}"
                )
                if ep == 0:
                    print(f"    [sample] {step_data.response_text[:120]}...")

            except Exception as e:
                print(f"  ⚠ Episode failed: {e}")
                continue

        if not batch_steps:
            print(f"  ⚠ No steps collected, skipping iteration {it}")
            continue

        # ── Frozen Defender Scoring (if available) ──
        # Dual-model loading: keep Launderer in VRAM, load Defender alongside.
        # On A100 (80GB) both models fit comfortably (~32 GB BF16, ~12 GB 4-bit).
        # This eliminates the ~90s save/unload/reload swap cycle per iteration.
        valid_steps = [s for s in batch_steps if s.is_valid]
        if has_defender and valid_steps and not cfg.dry_run:
            print(f"  🔄 Scoring {len(valid_steps)} valid scenarios against frozen Defender...")
            try:
                # Load Defender alongside Launderer (no unload needed)
                def_model, def_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=cfg.defender_checkpoint,
                    max_seq_length=4096,
                    load_in_4bit=cfg.load_in_4bit,
                    dtype=None,
                )
                FastLanguageModel.for_inference(def_model)

                # Score each valid scenario
                from server.aml_environment import AMLEnvironment
                from server.launderer_env import extract_json, validate_scenario
                from scenarios.procedural_generator import GeneratedScenario
                from train_defender_ppo import format_prompt, parse_action

                def_env = AMLEnvironment()
                for step in valid_steps:
                    try:
                        data = extract_json(step.response_text)
                        if data is None:
                            continue
                        is_valid, _ = validate_scenario(data)
                        if not is_valid:
                            continue
                        scenario = GeneratedScenario(data)
                        obs = def_env.reset(scenario=scenario)

                        # Run Defender for up to max_steps
                        for step_num in range(1, cfg.defender_max_steps + 1):
                            ram = def_env._sm.ram_contents if def_env._sm else []
                            disk = def_env._sm.disk_contents if def_env._sm else []
                            kernel = def_env._sm.kernel_directives if def_env._sm else []
                            prompt = format_prompt(obs, step_num, kernel, disk, ram)

                            inputs = def_tokenizer(
                                prompt, return_tensors="pt", truncation=True,
                                max_length=3840,
                            ).to(device)
                            with torch.no_grad():
                                out = def_model.generate(
                                    **inputs, max_new_tokens=192,
                                    temperature=0.5, top_p=0.9, do_sample=True,
                                    pad_token_id=def_tokenizer.eos_token_id,
                                )
                            resp = def_tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                            tool, params = parse_action(resp)

                            from models import AMLAction
                            try:
                                obs = def_env.step(AMLAction(tool=tool, parameters=params))
                            except Exception:
                                obs = AMLObservation(done=True, reward=-0.1)
                            if obs.done:
                                break

                        defender_score = def_env._state.accumulated_reward

                        # Recompute Launderer reward with real Defender score
                        from server.launderer_env import REWARD_VALID_BASELINE
                        clamped = max(-1.0, min(1.0, defender_score))
                        step.reward = round(
                            REWARD_VALID_BASELINE + (1.0 - REWARD_VALID_BASELINE) * (1.0 - clamped) / 2.0,
                            4
                        )
                        print(f"    Def score={defender_score:+.3f} → Laund reward={step.reward:+.3f}")

                    except Exception as e:
                        print(f"    ⚠ Defender scoring failed for scenario: {e}")

                # Unload Defender only — Launderer stays loaded
                del def_model, def_tokenizer, def_env
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  ✓ Defender unloaded, Launderer intact  |  VRAM: {vram_status()}")

            except Exception as e:
                print(f"  ⚠ Defender scoring failed: {e}. Using dummy scores for this iteration.")

        # Inject small noise to break ties when all rewards are identical.
        # This prevents zero advantage variance from stalling training.
        # σ=0.02 is small enough not to corrupt the signal but large
        # enough to create non-zero advantages for PPO to act on.
        reward_std = max(
            (sum((s.reward - sum(r.reward for r in batch_steps) / len(batch_steps)) ** 2
                 for s in batch_steps) / len(batch_steps)) ** 0.5,
            0.0,
        )
        if reward_std < 1e-6:
            for s in batch_steps:
                s.reward += random.gauss(0, 0.02)

        # Compute batch advantages
        LaundererPPO.compute_batch_advantages(batch_steps)

        # ── PPO Stability Guard: Advantage Variance Check ──
        # If all rewards are identical (e.g., all -2.0), advantages are all 0.0
        # and PPO update produces zero gradients. Skip to avoid wasted compute.
        adv_var = sum(s.advantage ** 2 for s in batch_steps) / max(len(batch_steps), 1)
        if adv_var < 1e-10:
            print(f"  ⚠ Skipping PPO update: advantage variance too low ({adv_var:.2e})")
            ppo_stats = {
                "launderer_ppo/loss/policy": 0.0,
                "launderer_ppo/kl": 0.0,
                "launderer_ppo/entropy": 0.0,
                "launderer_ppo/steps_trained": 0,
                "launderer_ppo/lr": ppo.scheduler.get_last_lr()[0],
            }
        else:
            ppo_stats = ppo.ppo_update(batch_steps)

        # Stats
        mean_reward = sum(s.reward for s in batch_steps) / len(batch_steps)
        valid_count = sum(1 for s in batch_steps if s.is_valid)
        mean_kl = ppo_stats.get("launderer_ppo/kl", 0.0)
        elapsed = time.time() - t0

        # KL warning
        if abs(mean_kl) > KL_WARN_THRESHOLD:
            print(f"  ⚠ KL={mean_kl:.4f} exceeds threshold {KL_WARN_THRESHOLD}")

        # KL auto-revert: track consecutive reward decline
        if prev_mean_reward is not None and prev_mean_reward > -1.5:
            decline = (prev_mean_reward - mean_reward) / max(abs(prev_mean_reward), 1e-6)
            if decline > REVERT_DECLINE_THRESHOLD:
                revert_decline_count += 1
                if revert_decline_count >= REVERT_PATIENCE and os.path.exists(best_checkpoint_path):
                    print(f"  🔄 Auto-reverting weights: reward declined {revert_decline_count}x in a row")
                    # Reload from best checkpoint
                    try:
                        old_model = ppo.model
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            model_name=best_checkpoint_path,
                            max_seq_length=cfg.max_seq_length,
                            load_in_4bit=cfg.load_in_4bit,
                            dtype=None,
                        )
                        model = FastLanguageModel.get_peft_model(
                            model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                            lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets,
                            bias="none", use_gradient_checkpointing="unsloth", random_state=42,
                        )
                        del old_model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        ppo = LaundererPPO(model, tokenizer, cfg, device)
                        revert_decline_count = 0
                    except Exception as e:
                        print(f"  ⚠ Auto-revert failed: {e}")
            else:
                revert_decline_count = 0
        prev_mean_reward = mean_reward

        print(
            f"\n  ═══ Iter {it}/{iters} ═══\n"
            f"    Mean reward:  {mean_reward:+.4f}\n"
            f"    Valid rate:    {valid_count}/{len(batch_steps)}\n"
            f"    PPO loss:     {ppo_stats['launderer_ppo/loss/policy']:.6f}\n"
            f"    KL:           {ppo_stats['launderer_ppo/kl']:.6f}\n"
            f"    Entropy:      {ppo_stats['launderer_ppo/entropy']:.4f}\n"
            f"    VRAM: {vram_status()}  |  {elapsed:.0f}s\n"
        )

        # WandB logging
        if not cfg.dry_run:
            import wandb
            wandb.log({
                "iteration": it,
                "launderer/returns/mean": mean_reward,
                "launderer/valid_rate": valid_count / max(len(batch_steps), 1),
                "launderer/adv_variance": adv_var,
                **ppo_stats,
                "perf/iter_seconds": elapsed,
            })

        # Checkpoint
        if mean_reward > best_reward:
            best_reward = mean_reward
            ppo.save(os.path.join(cfg.output_dir, "best"))
            ppo.save(best_checkpoint_path)  # For auto-revert
        if it % cfg.save_every == 0:
            ppo.save(os.path.join(cfg.output_dir, f"iter-{it}"))

        # GC
        del batch_steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    ppo.save(os.path.join(cfg.output_dir, "final"))
    print(f"\n{'═'*60}")
    print(f"  LAUNDERER TRAINING COMPLETE  |  Best reward: {best_reward:+.4f}")
    print(f"{'═'*60}\n")

    if not cfg.dry_run:
        import wandb
        wandb.finish()

    return best_reward


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex Launderer PPO Trainer")
    p.add_argument("--model", default=LaundererPPOConfig.model_name)
    p.add_argument("--lr", type=float, default=LaundererPPOConfig.lr)
    p.add_argument("--lora-r", type=int, default=LaundererPPOConfig.lora_r)
    p.add_argument("--episodes", type=int, default=LaundererPPOConfig.episodes_per_iter)
    p.add_argument("--iterations", type=int, default=LaundererPPOConfig.total_iterations)
    p.add_argument("--temperature", type=float, default=LaundererPPOConfig.temperature)
    p.add_argument("--defender-checkpoint", type=str, default="")
    p.add_argument("--output-dir", default=LaundererPPOConfig.output_dir)
    p.add_argument("--wandb-project", default=LaundererPPOConfig.wandb_project)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    train(LaundererPPOConfig(
        model_name=args.model, lr=args.lr, lora_r=args.lora_r,
        episodes_per_iter=args.episodes, total_iterations=args.iterations,
        temperature=args.temperature,
        defender_checkpoint=args.defender_checkpoint,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
