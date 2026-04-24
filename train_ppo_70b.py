#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — 70B Multi-GPU PPO Trainer.

Distributed step-level PPO for Meta-Llama-3.1-70B-Instruct on multi-node
A100 (80GB) clusters via DeepSpeed ZeRO-3. Preserves the core Memex training
contract: per-step intermediate rewards, disable_adapter() KL divergence,
and OS mechanic tracking (page faults, async timeouts, disk writes, kernel
injections).

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  DeepSpeed ZeRO-3: Model + Optimizer + Gradients sharded across    │
  │  N × A100-80GB GPUs                                                │
  │                                                                    │
  │  ┌──────────────────────┐   ┌─────────────────────────────────┐    │
  │  │  Trajectory Rollout  │──▶│  PPO Clipped Surrogate Update   │    │
  │  │  (rank 0 → env,      │   │  + KL via disable_adapter()     │    │
  │  │   broadcast actions)  │   │  + gradient accumulation        │    │
  │  └──────────────────────┘   └─────────────────────────────────┘    │
  │           ↑                              ↓                         │
  │    AMLEnvironment                  LoRA weight update               │
  │  (procedural scenarios)         (DeepSpeed fused optimizer)        │
  └──────────────────────────────────────────────────────────────────────┘

Launch:
  deepspeed --num_gpus 4 train_ppo_70b.py --iterations 50 --episodes 2
  deepspeed --num_gpus 8 --num_nodes 2 train_ppo_70b.py --hostfile hosts.txt
  python train_ppo_70b.py --dry-run                    # single-GPU dry-run
  python train_ppo_70b.py --eval checkpoints_70b/best  # evaluation sweep
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
import torch.distributed as dist
import torch.nn.functional as F

try:
    import deepspeed
except ImportError:
    deepspeed = None  # Allows dry-run on machines without DeepSpeed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction, AMLObservation
from server.aml_environment import AMLEnvironment


# ═══════════════════════════════════════════════════════════════════════════
# Distributed Utilities
# ═══════════════════════════════════════════════════════════════════════════

def is_distributed() -> bool:
    return dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def log(msg: str) -> None:
    """Only rank 0 prints to avoid log spam."""
    if is_main_process():
        print(msg, flush=True)

def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from src to all ranks."""
    if not is_distributed():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PPOConfig70B:
    # ── Model ──
    model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True        # NF4 quantization via bitsandbytes

    # ── LoRA ──
    lora_r: int = 32                 # higher rank for 70B capacity
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── PPO ──
    lr: float = 2e-6                 # lower LR for 70B stability
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.03           # lower KL for larger model
    entropy_coef: float = 0.05      # entropy bonus (prevents mode collapse)
    gamma: float = 0.99
    reward_clip: float = 2.0        # clip returns to [-clip, +clip]
    max_grad_norm: float = 0.5       # tighter clipping for 70B
    grad_accum_steps: int = 8        # larger accumulation for multi-GPU

    # ── Generation ──
    max_new_tokens: int = 256
    temperature: float = 0.5         # moderate for RL exploration
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # ── Environment ──
    episodes_per_iter: int = 2       # fewer per iter (each episode is expensive)
    total_iterations: int = 50
    max_steps: int = 25
    difficulties: list = field(default_factory=lambda: ["easy", "medium", "hard"])
    typologies: list = field(default_factory=lambda: [
        "structuring", "layering", "trade_based_ml",
    ])

    # ── DeepSpeed ──
    ds_stage: int = 3                # ZeRO-3 for 70B
    offload_optimizer: bool = False  # set True if VRAM-constrained
    offload_params: bool = False     # parameter offloading (CPU)

    # ── Logging / Checkpointing ──
    wandb_project: str = "memex-ppo-70b"
    save_every: int = 10
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints_70b")
    dry_run: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# DeepSpeed Configuration Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_ds_config(cfg: PPOConfig70B) -> Dict[str, Any]:
    """Build a DeepSpeed JSON config for ZeRO-3 with optional offloading."""
    ds = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": cfg.ds_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "gradient_clipping": cfg.max_grad_norm,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    if cfg.offload_optimizer:
        ds["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    if cfg.offload_params:
        ds["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return ds


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt (identical to T4 trainer for consistency)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Senior AML Compliance Investigator in a Memex OS-Agent Environment.
You must produce FinCEN-grade Suspicious Activity Reports following strict investigative protocol.

OUTPUT: Respond with EXACTLY ONE raw JSON object. No markdown.
{"tool": "<name>", "parameters": {<params>}, "reasoning": "<1 sentence>"}

OS RULES:
- RAM holds LAST 2 observations. Use write_to_case_file to persist data.
- request_wire_trace is ASYNC (wait for ETA). retrieve_async_result when ready.
- search_compliance_manual + update_system_prompt injects rules (+0.15 reward).

INVESTIGATION TOOLS:
  review_alert, get_customer_profile, query_transactions, check_watchlist,
  trace_network, check_source_of_funds, assess_risk, check_market_price,
  check_device_overlap, verify_customs_invoice, query_beneficial_ownership,
  write_to_case_file, request_wire_trace, retrieve_async_result,
  search_compliance_manual, update_system_prompt, file_sar, close_alert

FINCEN 4-PILLAR PROTOCOL (follow this order):
1. DEVICE FINGERPRINTS: check_device_overlap(entity_id) — detect mule rings via shared devices/IPs.
2. TRADE VERIFICATION: verify_customs_invoice(invoice_id) — detect phantom shipments & over-invoicing.
3. BENEFICIAL OWNERSHIP: query_beneficial_ownership(entity_id) — trace UBOs through shell layers.
4. VELOCITY ANALYSIS: query_transactions + timestamps — detect pass-through patterns.

TERMINAL:
  file_sar(typology, entities_involved, findings, ubo_identified, evidence_chain)
  close_alert(reason, findings)
Typologies: structuring | layering | trade_based_ml | false_positive

SAR QUALITY: Your score depends on correct decision, typology, entity F1, findings coverage,
UBO identification accuracy, and use of all 4 investigation pillars. Missing UBOs or wrong
typologies incur heavy penalties."""


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Formatting & JSON Parsing
# ═══════════════════════════════════════════════════════════════════════════

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
            user_parts.append(f"[RAM] {r[:500]}")

    obs_data = obs.tool_result if obs.tool_result else {}
    user_parts.append(f"STEP {step}:\n{json.dumps(obs_data, indent=1, default=str)[:2000]}")
    if obs.message:
        user_parts.append(obs.message[:300])

    user_msg = "\n".join(user_parts)

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_action(text: str) -> Tuple[str, Dict[str, Any]]:
    """Robust JSON tool-call parser with multi-tier fallback.

    Always returns (tool_name: str, params: dict). Never crashes.
    """
    text = re.sub(r"```(?:json)?\s*", "", text.strip()).strip("` \n")
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            tool = str(d.get("tool", "review_alert"))
            params = d.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            return tool, params
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = re.search(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                tool = str(d.get("tool", "review_alert"))
                params = d.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                return tool, params
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    return (tm.group(1) if tm else "review_alert"), {}


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepData:
    """Single step in an episode trajectory."""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    response_start: int
    reward: float
    old_log_prob: float
    ref_log_prob: float
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


# ═══════════════════════════════════════════════════════════════════════════
# Core 70B PPO Trainer
# ═══════════════════════════════════════════════════════════════════════════

class MemexPPO70B:
    """Distributed step-level PPO for 70B models.

    Key differences from T4 trainer:
      - DeepSpeed ZeRO-3 engine wraps model + optimizer
      - Forward passes use engine.module for inference
      - KL reference via model.disable_adapter() preserved (LoRA-level, not ZeRO-level)
      - Only rank 0 interacts with the environment; trajectories broadcast to all ranks
      - Gradient accumulation handled by DeepSpeed engine
    """

    def __init__(self, engine, model, tokenizer, config: PPOConfig70B):
        self.engine = engine            # DeepSpeed engine (wraps model + optimizer)
        self.model = model              # raw model reference for generation
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = engine.device if engine else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Running reward baseline
        self._reward_ema = 0.0
        self._reward_ema_alpha = 0.1

    # ── Log Probability ─────────────────────────────────────────────

    def _compute_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor,
        with_grad: bool = False, use_engine: bool = False,
    ) -> torch.Tensor:
        """Compute mean per-token log probability for response tokens."""
        full_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(self.device)
        resp_start = len(query_ids)

        fwd_model = self.engine.module if (use_engine and self.engine) else self.model

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = fwd_model(input_ids=full_ids).logits[0]
            resp_logits = logits[resp_start - 1 : -1, :]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            token_lp = log_probs.gather(1, response_ids.to(self.device).unsqueeze(1)).squeeze(1)
            return token_lp.mean()  # mean, NOT sum — keeps KL scale-invariant across response lengths

    def _compute_ref_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor,
    ) -> float:
        """Compute reference log prob using the frozen base (LoRA disabled).

        The disable_adapter() trick works at the PEFT/LoRA module level,
        independent of DeepSpeed sharding. ZeRO-3 gathers parameters on
        demand for the forward pass — disabling adapters simply means
        the gathered parameters skip LoRA additions.
        """
        base_model = self.engine.module if self.engine else self.model
        base_model.eval()
        try:
            with base_model.disable_adapter():
                lp = self._compute_log_prob(query_ids, response_ids, with_grad=False)
        except AttributeError:
            lp = self._compute_log_prob(query_ids, response_ids, with_grad=False)
        base_model.train()
        return lp.item()

    # ── Generation ──────────────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate a response (rank 0 only during rollout)."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_seq_length - self.cfg.max_new_tokens,
        ).to(self.device)

        query_ids = inputs["input_ids"].squeeze(0)

        gen_model = self.engine.module if self.engine else self.model

        with torch.no_grad():
            out = gen_model.generate(
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

    # ── Episode Rollout ─────────────────────────────────────────────

    def rollout(
        self, env: AMLEnvironment, difficulty: str, typology: str,
    ) -> Tuple[List[StepData], EpisodeStats]:
        """Collect one full episode trajectory.

        In distributed mode, only rank 0 interacts with the environment.
        Trajectory data is broadcast to all ranks for the PPO update.
        """
        gen_model = self.engine.module if self.engine else self.model
        gen_model.eval()

        # Only rank 0 runs the environment
        if is_main_process():
            obs = env.reset(task_id=difficulty)
            steps_data: List[Dict[str, Any]] = []

            for step_num in range(1, self.cfg.max_steps + 1):
                ram = env._sm.ram_contents if env._sm else []
                disk = env._sm.disk_contents if env._sm else []
                kernel = env._sm.kernel_directives if env._sm else []

                prompt = format_prompt(obs, step_num, kernel, disk, ram)
                resp_text, q_ids, r_ids = self.generate(prompt)

                # Degenerate response detection: >80% repeated tokens = gibberish
                if len(r_ids) > 4:
                    unique_ratio = len(set(r_ids.tolist())) / len(r_ids)
                    if unique_ratio < 0.20:
                        steps_data.append({
                            "q_ids": q_ids, "r_ids": r_ids,
                            "reward": -0.15, "old_lp": self._compute_log_prob(q_ids, r_ids, with_grad=False).item(),
                            "ref_lp": self._compute_ref_log_prob(q_ids, r_ids),
                            "resp_text": resp_text,
                        })
                        break  # End episode — degenerate model output

                old_lp = self._compute_log_prob(q_ids, r_ids, with_grad=False).item()
                ref_lp = self._compute_ref_log_prob(q_ids, r_ids)

                # Parse and step environment (fault-tolerant)
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
                        done=True,
                        reward=reward,
                        metadata={"step": step_num, "error": "malformed_action"},
                    )

                steps_data.append({
                    "q_ids": q_ids, "r_ids": r_ids,
                    "reward": reward, "old_lp": old_lp, "ref_lp": ref_lp,
                    "resp_text": resp_text,
                })

                if obs.done:
                    break

            st = env._state
            stats_dict = {
                "score": st.accumulated_reward, "steps": len(steps_data),
                "difficulty": difficulty, "typology": typology,
                "page_faults": st.page_fault_count,
                "async_timeouts": st.async_timeout_count,
                "successful_pages": st.successful_pages,
                "meta_injections": st.meta_injections,
            }
        else:
            steps_data = []
            stats_dict = {}

        # Broadcast trajectory and stats to all ranks
        steps_data = broadcast_object(steps_data, src=0)
        stats_dict = broadcast_object(stats_dict, src=0)

        # Reconstruct StepData objects on all ranks
        steps: List[StepData] = []
        for sd in steps_data:
            steps.append(StepData(
                query_ids=sd["q_ids"], response_ids=sd["r_ids"],
                response_start=len(sd["q_ids"]),
                reward=sd["reward"], old_log_prob=sd["old_lp"],
                ref_log_prob=sd["ref_lp"], response_text=sd["resp_text"],
            ))

        self._compute_advantages(steps)

        stats = EpisodeStats(**stats_dict)
        return steps, stats

    def _compute_advantages(self, steps: List[StepData]) -> None:
        """Discounted returns with EMA baseline normalization."""
        T = len(steps)
        if T == 0:
            return
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = steps[t].reward + self.cfg.gamma * G
            returns[t] = G

        # Clip returns to prevent outlier gradient signals
        clip = self.cfg.reward_clip
        returns = [max(-clip, min(clip, r)) for r in returns]

        # Per-episode normalization: no cross-difficulty contamination
        mean_return = sum(returns) / T
        advs = [r - mean_return for r in returns]
        std = max((sum(a**2 for a in advs) / T) ** 0.5, 1e-8)
        for t in range(T):
            steps[t].advantage = advs[t] / std

    # ── PPO Update ──────────────────────────────────────────────────

    def ppo_update(self, all_steps: List[StepData]) -> Dict[str, float]:
        """PPO clipped surrogate update using DeepSpeed engine.

        Processes steps one at a time for memory efficiency.
        DeepSpeed engine handles gradient accumulation and all-reduce internally.
        """
        if self.engine:
            self.engine.module.train()

        total_policy_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.ppo_epochs):
            random.shuffle(all_steps)
            epoch_kl = 0.0
            epoch_steps = 0

            for i, step in enumerate(all_steps):
                try:
                    # Current policy log prob (WITH gradients, through engine)
                    fwd_model = self.engine.module if self.engine else self.model
                    full_ids = torch.cat([step.query_ids, step.response_ids]).unsqueeze(0).to(self.device)
                    resp_start = len(step.query_ids)

                    logits = fwd_model(input_ids=full_ids).logits[0]
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
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps,
                    ) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # KL penalty against frozen base (abs prevents negative KL rewarding divergence)
                    kl = new_lp - step.ref_log_prob
                    kl_loss = self.cfg.kl_coef * kl.abs()

                    loss = policy_loss + kl_loss - self.cfg.entropy_coef * entropy

                    # DeepSpeed handles backward + gradient accumulation + all-reduce
                    if self.engine:
                        self.engine.backward(loss)
                        self.engine.step()
                    else:
                        loss.backward()

                    total_policy_loss += policy_loss.item()
                    total_kl += kl.item()
                    total_entropy += entropy.item()
                    n_updates += 1
                    epoch_kl += kl.item()
                    epoch_steps += 1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log(f"  ⚠ OOM on step {i}, clearing cache and skipping")
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise

            # KL early stopping: if mean KL too large, remaining epochs are wasted
            if epoch_steps > 0:
                mean_epoch_kl = abs(epoch_kl / epoch_steps)
                if mean_epoch_kl > 15.0:
                    break  # policy drifted too far, stop PPO epochs

        n = max(n_updates, 1)
        return {
            "ppo/loss/policy": total_policy_loss / n,
            "ppo/kl": total_kl / n,
            "ppo/entropy": total_entropy / n,
            "ppo/steps_trained": n_updates,
        }

    # ── Checkpoint ──────────────────────────────────────────────────

    def save(self, path: str, iteration: int = 0) -> None:
        """Save checkpoint using DeepSpeed's distributed saver."""
        if self.engine:
            self.engine.save_checkpoint(path, tag=f"iter-{iteration}")
            if is_main_process():
                self.tokenizer.save_pretrained(path)
        elif is_main_process():
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        log(f"  💾 Checkpoint: {path}")

    def load(self, path: str) -> None:
        """Resume from a DeepSpeed checkpoint."""
        if self.engine:
            _, client_state = self.engine.load_checkpoint(path)
            log(f"  📂 Resumed from: {path}")
        else:
            log(f"  ⚠ No DeepSpeed engine — skipping checkpoint load")

    # ── Stable Checkpoint (Auto-Revert) ─────────────────────────────────

    def save_stable_state(self, path: str) -> None:
        """Save LoRA weights as 'last known good'.

        CRITICAL: On ZeRO-3, parameters are sharded across ranks.
        We must use GatheredParameters to materialize full tensors
        before saving, otherwise rank 0 saves its own 1/N shard.
        """
        base = self.engine.module if self.engine else self.model
        trainable = [p for p in base.parameters() if p.requires_grad]

        # GatheredParameters materializes full tensors from ZeRO-3 shards
        # on all ranks. Only rank 0 does I/O.
        gather_ctx = (
            deepspeed.zero.GatheredParameters(trainable, modifier_rank=0)
            if self.engine and deepspeed is not None and hasattr(deepspeed, 'zero') and hasattr(deepspeed.zero, 'GatheredParameters')
            else torch.no_grad()  # no-op context for non-ZeRO / no DeepSpeed
        )
        with gather_ctx:
            if not is_main_process():
                return
            os.makedirs(path, exist_ok=True)
            lora_state = {
                k: v.cpu().clone()
                for k, v in base.named_parameters() if v.requires_grad
            }
            torch.save(lora_state, os.path.join(path, "lora_state.pt"))
            torch.save({
                "entropy_coef": self.cfg.entropy_coef,
                "temperature": self.cfg.temperature,
                "lr": self.cfg.lr,
                "reward_ema": self._reward_ema,
            }, os.path.join(path, "hyperparams.pt"))

    def load_stable_state(self, path: str) -> bool:
        """Reload LoRA weights from stable checkpoint across all ranks."""
        lora_path = os.path.join(path, "lora_state.pt")
        if is_main_process() and not os.path.exists(lora_path):
            exists = False
        else:
            exists = True
        exists = broadcast_object(exists, src=0)
        if not exists:
            return False

        # Rank 0 loads, then broadcasts
        if is_main_process():
            state = torch.load(lora_path, map_location="cpu", weights_only=True)
            hp = torch.load(
                os.path.join(path, "hyperparams.pt"),
                map_location="cpu", weights_only=True,
            )
        else:
            state = None
            hp = None
        state = broadcast_object(state, src=0)
        hp = broadcast_object(hp, src=0)

        base = self.engine.module if self.engine else self.model
        for name, param in base.named_parameters():
            if name in state:
                param.data.copy_(state[name].to(self.device))
        self._reward_ema = hp.get("reward_ema", 0.0)
        return True


# ═══════════════════════════════════════════════════════════════════════════
# VRAM Monitoring
# ═══════════════════════════════════════════════════════════════════════════

def vram_status() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 1e9
    total = torch.cuda.get_device_properties(dev).total_memory / 1e9
    return f"{alloc:.1f}/{total:.1f} GB (GPU {dev})"


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: PPOConfig70B) -> None:
    banner = f"""
{'═'*70}
  MEMEX PPO TRAINER  —  70B Multi-GPU (DeepSpeed ZeRO-{cfg.ds_stage})
  Model:      {cfg.model_name}
  LoRA:       r={cfg.lora_r}  α={cfg.lora_alpha}
  LR:         {cfg.lr}  |  Clip: {cfg.clip_eps}  |  KL: {cfg.kl_coef}
  Iterations: {cfg.total_iterations}  ×  {cfg.episodes_per_iter} ep/iter
  GPUs:       {get_world_size()}  |  Dry: {cfg.dry_run}
{'═'*70}"""
    log(banner)

    # ── 1. Load Model ──
    log("[1/5] Loading 70B model...")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=torch.bfloat16,
        )
        log(f"  ✓ Loaded via Unsloth  |  VRAM: {vram_status()}")
    except ImportError:
        log("  ⚠ Unsloth not available, falling back to transformers + BnB")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  ✓ Model loaded  |  VRAM: {vram_status()}")

    # ── 2. Attach LoRA ──
    log("[2/5] Attaching LoRA adapters...")
    try:
        from unsloth import FastLanguageModel as FLM
        model = FLM.get_peft_model(
            model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets,
            bias="none", use_gradient_checkpointing="unsloth", random_state=42,
        )
    except (ImportError, Exception):
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets,
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"  ✓ LoRA: {trainable:,} / {total_p:,} ({100*trainable/total_p:.3f}%)")

    # ── 3. Initialize DeepSpeed ──
    log("[3/5] Initializing DeepSpeed engine...")
    if deepspeed is None:
        raise ImportError("DeepSpeed is required for 70B training. Install: pip install deepspeed")

    ds_config = build_ds_config(cfg)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, betas=(0.9, 0.95))

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.total_iterations, eta_min=cfg.lr * 0.1,
        ),
    )
    log(f"  ✓ DeepSpeed ZeRO-{cfg.ds_stage}  |  World: {get_world_size()} GPUs")
    log(f"  VRAM: {vram_status()}")

    # ── 4. WandB ──
    if not cfg.dry_run and is_main_process():
        log("[4/5] Initializing WandB...")
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=f"memex-70b-{get_world_size()}gpu-{int(time.time())}",
            config={
                **vars(cfg),
                "world_size": get_world_size(),
                "deepspeed_stage": cfg.ds_stage,
            },
        )
        log("  ✓ WandB ready")
    else:
        log("[4/5] WandB SKIPPED (dry-run or non-main rank)")

    # ── 5. Train ──
    log("[5/5] Starting training loop...\n")
    ppo = MemexPPO70B(engine, model, tokenizer, cfg)

    # Environment only on rank 0 (lightweight, no GPU needed)
    env = AMLEnvironment() if is_main_process() else None

    iters = 2 if cfg.dry_run else cfg.total_iterations
    eps_per = 1 if cfg.dry_run else cfg.episodes_per_iter
    best_score = -float("inf")

    # ── Auto-Revert State ──
    stable_path = os.path.join(cfg.output_dir, "_stable")
    stable_saved = False
    low_entropy_streak = 0
    low_score_streak = 0
    revert_count = 0
    max_reverts = 5

    for it in range(1, iters + 1):
        t0 = time.time()
        all_steps: List[StepData] = []
        ep_stats: List[EpisodeStats] = []

        # ── Collect trajectories ──
        for ep in range(eps_per):
            diff = random.choice(cfg.difficulties)
            typo = random.choice(cfg.typologies)

            # Sync random choices across ranks
            diff = broadcast_object(diff, src=0)
            typo = broadcast_object(typo, src=0)

            try:
                steps, stats = ppo.rollout(env, diff, typo)
                all_steps.extend(steps)
                ep_stats.append(stats)

                log(
                    f"  It {it:>3} Ep {ep+1} | {diff}/{typo} | "
                    f"steps={stats.steps:>2} score={stats.score:+.3f} | "
                    f"PF={stats.page_faults} AT={stats.async_timeouts} "
                    f"SP={stats.successful_pages} MI={stats.meta_injections}"
                )
                if ep == 0 and steps:
                    log(f"    [sample] {steps[0].response_text[:150]}...")

            except Exception as e:
                log(f"  ⚠ Episode failed: {e}")
                continue

        if not all_steps:
            log(f"  ⚠ No steps collected, skipping iteration {it}")
            continue

        # ── PPO Update (all ranks participate) ──
        ppo_stats = ppo.ppo_update(all_steps)

        # ── Stats ──
        mean_score = sum(s.score for s in ep_stats) / len(ep_stats)
        total_pf = sum(s.page_faults for s in ep_stats)
        total_at = sum(s.async_timeouts for s in ep_stats)
        total_sp = sum(s.successful_pages for s in ep_stats)
        total_mi = sum(s.meta_injections for s in ep_stats)
        elapsed = time.time() - t0
        entropy = ppo_stats["ppo/entropy"]
        kl = ppo_stats["ppo/kl"]

        log(
            f"\n  ═══ Iter {it}/{iters} ═══\n"
            f"    Mean score:  {mean_score:+.4f}\n"
            f"    PPO loss:    {ppo_stats['ppo/loss/policy']:.6f}\n"
            f"    KL:          {kl:.6f}\n"
            f"    Entropy:     {entropy:.4f}\n"
            f"    OS: PF={total_pf} AT={total_at} SP={total_sp} MI={total_mi}\n"
            f"    GPUs: {get_world_size()}  |  VRAM: {vram_status()}  |  {elapsed:.0f}s\n"
        )

        # ── Entropy Heartbeat Monitor (Auto-Revert) ──────────────────
        if entropy < 0.01:
            low_entropy_streak += 1
        else:
            low_entropy_streak = 0

        if mean_score <= 0.0:
            low_score_streak += 1
        else:
            low_score_streak = 0

        # Save stable checkpoint when healthy
        if entropy > 0.05 and mean_score > 0.3:
            ppo.save_stable_state(stable_path)
            stable_saved = True

        is_collapse = (
            low_entropy_streak >= 2
            or low_score_streak >= 2
            or abs(kl) > 10.0
        )

        if is_collapse and stable_saved and revert_count < max_reverts:
            revert_count += 1
            log(
                f"\n  ⚠️  COLLAPSE DETECTED (revert #{revert_count})  ⚠️\n"
                f"    Trigger: entropy={entropy:.4f} | KL={kl:.4f} | "
                f"score={mean_score:+.4f}\n"
                f"    Purging VRAM and reloading stable checkpoint..."
            )

            torch.cuda.empty_cache()
            gc.collect()

            success = ppo.load_stable_state(stable_path)
            if not success:
                log("    ❌ Failed to load stable state, continuing...")
            else:
                old_ec = cfg.entropy_coef
                old_temp = cfg.temperature
                old_lr = cfg.lr
                cfg.entropy_coef = min(cfg.entropy_coef * 1.5, 0.20)
                cfg.temperature = min(cfg.temperature + 0.1, 0.9)
                cfg.lr = cfg.lr * 0.7
                ppo.cfg = cfg

                # DeepSpeed manages the optimizer internally —
                # we must explicitly set the LR in its param groups.
                if ppo.engine and ppo.engine.optimizer:
                    for pg in ppo.engine.optimizer.param_groups:
                        pg['lr'] = cfg.lr

                log(
                    f"    ✅ Reverted successfully.\n"
                    f"    entropy_coef: {old_ec:.4f} → {cfg.entropy_coef:.4f}\n"
                    f"    temperature:  {old_temp:.2f} → {cfg.temperature:.2f}\n"
                    f"    lr:           {old_lr:.2e} → {cfg.lr:.2e}\n"
                )

            low_entropy_streak = 0
            low_score_streak = 0

            if not cfg.dry_run and is_main_process():
                import wandb
                wandb.log({
                    "revert/count": revert_count,
                    "revert/entropy_coef": cfg.entropy_coef,
                    "revert/temperature": cfg.temperature,
                    "revert/lr": cfg.lr,
                    "revert/trigger_entropy": entropy,
                    "revert/trigger_kl": kl,
                    "revert/trigger_score": mean_score,
                })

            del all_steps, ep_stats
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # ── WandB ──
        if not cfg.dry_run and is_main_process():
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
                "stability/entropy_coef": cfg.entropy_coef,
                "stability/temperature": cfg.temperature,
                "stability/revert_count": revert_count,
            })

        # ── Checkpoint ──
        if mean_score > best_score:
            best_score = mean_score
            ppo.save(os.path.join(cfg.output_dir, "best"), iteration=it)
        if it % cfg.save_every == 0:
            ppo.save(os.path.join(cfg.output_dir, f"iter-{it}"), iteration=it)

        # ── GC ──
        del all_steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final save ──
    ppo.save(os.path.join(cfg.output_dir, "final"), iteration=iters)
    log(f"\n{'═'*70}")
    log(f"  TRAINING COMPLETE  |  Best: {best_score:+.4f}  |  GPUs: {get_world_size()}")
    log(f"{'═'*70}\n")

    if not cfg.dry_run and is_main_process():
        import wandb
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation (single-GPU or multi-GPU)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model_path: str) -> None:
    """Evaluate a checkpoint across all 9 difficulty×typology combos."""
    log(f"\n{'═'*70}\n  EVALUATION: {model_path}\n{'═'*70}")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=4096, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = PPOConfig70B()
    ppo = MemexPPO70B(engine=None, model=model, tokenizer=tokenizer, config=cfg)
    env = AMLEnvironment()

    scores = []
    for diff in ["easy", "medium", "hard"]:
        for typo in ["structuring", "layering", "trade_based_ml"]:
            _, stats = ppo.rollout(env, diff, typo)
            scores.append(stats.score)
            log(
                f"  {diff:>6}/{typo:<15} | steps={stats.steps:>2} | "
                f"score={stats.score:+.4f} | "
                f"PF={stats.page_faults} SP={stats.successful_pages} MI={stats.meta_injections}"
            )

    log(f"\n  Mean: {sum(scores)/len(scores):+.4f}  "
        f"Min: {min(scores):+.4f}  Max: {max(scores):+.4f}")
    log(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex 70B Multi-GPU PPO Trainer")
    p.add_argument("--model", default=PPOConfig70B.model_name)
    p.add_argument("--lr", type=float, default=PPOConfig70B.lr)
    p.add_argument("--lora-r", type=int, default=PPOConfig70B.lora_r)
    p.add_argument("--lora-alpha", type=int, default=PPOConfig70B.lora_alpha)
    p.add_argument("--episodes", type=int, default=PPOConfig70B.episodes_per_iter)
    p.add_argument("--iterations", type=int, default=PPOConfig70B.total_iterations)
    p.add_argument("--temperature", type=float, default=PPOConfig70B.temperature)
    p.add_argument("--grad-accum", type=int, default=PPOConfig70B.grad_accum_steps)
    p.add_argument("--ds-stage", type=int, default=PPOConfig70B.ds_stage, choices=[2, 3])
    p.add_argument("--offload-optimizer", action="store_true")
    p.add_argument("--offload-params", action="store_true")
    p.add_argument("--wandb-project", default=PPOConfig70B.wandb_project)
    p.add_argument("--output-dir", default=PPOConfig70B.output_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval", type=str, default=None, metavar="PATH")
    p.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    args = p.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(PPOConfig70B(
            model_name=args.model, lr=args.lr,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            episodes_per_iter=args.episodes, total_iterations=args.iterations,
            temperature=args.temperature, grad_accum_steps=args.grad_accum,
            ds_stage=args.ds_stage,
            offload_optimizer=args.offload_optimizer,
            offload_params=args.offload_params,
            wandb_project=args.wandb_project, output_dir=args.output_dir,
            dry_run=args.dry_run,
        ))


if __name__ == "__main__":
    main()
