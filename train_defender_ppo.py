#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — Defender PPO Training (Mixed-Scenario).

Fork of train_ppo.py with these additions:
  - --scenario-source flag: 'procedural' (Phase 1) or 'mixed' (Phase 3)
  - --launderer-checkpoint: path to frozen Launderer for scenario generation
  - --mix-ratio: fraction of episodes using Launderer scenarios (0.0 to 1.0)
  - GAE (Generalized Advantage Estimation) replaces simple MC returns
  - Entity-F1 and typology-accuracy logged as separate WandB metrics

Usage:
  python train_defender_ppo.py --dry-run --scenario-source procedural
  python train_defender_ppo.py --scenario-source mixed --launderer-checkpoint checkpoints/launderer/best --mix-ratio 0.5
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
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import AMLAction, AMLObservation
from server.aml_environment import AMLEnvironment


# ═══════════════════════════════════════════════════════════════════════
# Configuration (extends base PPOConfig with self-play fields)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DefenderPPOConfig:
    # ── Model ──
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # ── LoRA ──
    lora_r: int = 16
    lora_alpha: int = -1
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── PPO ──
    lr: float = 5e-6
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    entropy_coef: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE λ parameter
    reward_clip: float = 2.0
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 4

    # ── Generation ──
    max_new_tokens: int = 192
    temperature: float = 0.5
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

    # ── Self-Play / Mixed Scenarios ──
    scenario_source: str = "procedural"  # "procedural" or "mixed"
    launderer_checkpoint: str = ""
    mix_ratio: float = 0.5  # fraction of Launderer scenarios in "mixed" mode

    # ── Logging / Checkpointing ──
    wandb_project: str = "memex-defender"
    save_every: int = 10
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "defender")
    dry_run: bool = False

    # ── PLR Curriculum ──
    use_plr: bool = False
    plr_buffer_size: int = 200

    def __post_init__(self):
        if self.lora_alpha == -1:
            self.lora_alpha = self.lora_r * 2


# ═══════════════════════════════════════════════════════════════════════
# System Prompt (same as train_ppo.py)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Prompt Formatting & JSON Parsing (same as train_ppo.py)
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


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Data
# ═══════════════════════════════════════════════════════════════════════

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
    scenario_source: str = "procedural"
    page_faults: int = 0
    async_timeouts: int = 0
    successful_pages: int = 0
    meta_injections: int = 0
    kernel_mode_uses: int = 0
    # Metrics for self-play tracking
    decision_correct: bool = False
    typology_correct: bool = False
    entity_f1: float = 0.0
    detection: str = ""  # TP, TN, FP, or FN


# ═══════════════════════════════════════════════════════════════════════
# Core PPO Trainer (extended with GAE)
# ═══════════════════════════════════════════════════════════════════════

class DefenderPPO:
    """Step-level PPO trainer for the Defender agent.

    Extensions over train_ppo.py::MemexPPO:
      - GAE (Generalized Advantage Estimation) with configurable λ
      - Mixed scenario source support (procedural + launderer)
      - Entity F1 and typology tracking per episode
    """

    def __init__(self, model, tokenizer, config: DefenderPPOConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_iterations, eta_min=config.lr * 0.1,
        )

    # ── Log Probability ──────────────────────────────────────────────

    def _compute_log_prob(
        self, query_ids: torch.Tensor, response_ids: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
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
        scenario_source: str = "procedural",
    ) -> Tuple[List[StepData], EpisodeStats]:
        """Collect one full episode trajectory."""
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
                break

        # Compute GAE advantages
        self._compute_gae(steps)

        # Extract episode metrics
        st = env._state
        gt = env._current_scenario.ground_truth if env._current_scenario else {}

        # Decision correctness
        decision_correct = False
        typology_correct = False
        entity_f1 = 0.0
        if st.decision_made and gt:
            # Check decision
            if st.findings:  # file_sar was called
                decision_correct = (gt.get("correct_decision") == "file_sar")
            else:  # close_alert
                decision_correct = (gt.get("correct_decision") == "close_alert")

            # Typology
            gt_typo = gt.get("typology", "")
            agent_typo = ""
            for f in st.findings:
                if f.lower() in ["structuring", "layering", "trade_based_ml"]:
                    agent_typo = f.lower()
                    break
            typology_correct = (agent_typo == gt_typo)

            # Entity F1
            gt_entities: Set[str] = set(gt.get("key_entities", []))
            # Approximate: entities from findings are hard to track perfectly
            # so use the state's watchlist_checked as proxy
            flagged: Set[str] = set(st.watchlist_checked) if st.watchlist_checked else set()
            if gt_entities and flagged:
                tp = len(flagged & gt_entities)
                prec = tp / max(len(flagged), 1)
                rec = tp / max(len(gt_entities), 1)
                entity_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        stats = EpisodeStats(
            score=st.accumulated_reward, steps=len(steps),
            difficulty=difficulty, typology=typology,
            scenario_source=scenario_source,
            page_faults=st.page_fault_count, async_timeouts=st.async_timeout_count,
            successful_pages=st.successful_pages, meta_injections=st.meta_injections,
            kernel_mode_uses=st.kernel_inject_reward_count,
            decision_correct=decision_correct,
            typology_correct=typology_correct,
            entity_f1=entity_f1,
        )
        return steps, stats

    def _compute_gae(self, steps: List[StepData]) -> None:
        """Compute GAE (Generalized Advantage Estimation) advantages.

        Uses γ (gamma) for discount and λ (gae_lambda) for bias-variance tradeoff.
        When λ=1.0, GAE reduces to Monte Carlo returns.
        When λ=0.0, GAE reduces to one-step TD.
        Default λ=0.95 balances both.
        """
        T = len(steps)
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda
        clip = self.cfg.reward_clip

        # Compute TD residuals: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        # Since we don't have a value function, approximate V(s) ≈ 0
        # This gives δ_t = r_t (clipped)
        rewards = [max(-clip, min(clip, s.reward)) for s in steps]

        # GAE: A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t]  # δ_t = r_t (no value baseline)
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        # Normalize per episode
        mean_adv = sum(advantages) / max(T, 1)
        std_adv = max((sum((a - mean_adv) ** 2 for a in advantages) / max(T, 1)) ** 0.5, 1e-8)
        for t in range(T):
            steps[t].advantage = (advantages[t] - mean_adv) / std_adv

    # ── PPO Update ───────────────────────────────────────────────────

    def ppo_update(self, all_steps: List[StepData]) -> Dict[str, float]:
        """Run PPO epochs. Same as MemexPPO but uses DefenderPPOConfig."""
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(self.model)

        total_policy_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.ppo_epochs):
            random.shuffle(all_steps)
            self.optimizer.zero_grad()
            epoch_kl = 0.0
            epoch_steps = 0

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
                    kl_loss = self.cfg.kl_coef * kl.abs()

                    loss = (policy_loss + kl_loss - self.cfg.entropy_coef * entropy) / self.cfg.grad_accum_steps
                    loss.backward()

                    total_policy_loss += policy_loss.item()
                    total_kl += kl.item()
                    total_entropy += entropy.item()
                    n_updates += 1
                    epoch_kl += kl.item()
                    epoch_steps += 1

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

            if epoch_steps > 0:
                mean_epoch_kl = abs(epoch_kl / epoch_steps)
                if mean_epoch_kl > 15.0:
                    break

            if len(all_steps) % self.cfg.grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

        n = max(n_updates, 1)
        return {
            "defender_ppo/loss/policy": total_policy_loss / n,
            "defender_ppo/kl": total_kl / n,
            "defender_ppo/entropy": total_entropy / n,
            "defender_ppo/steps_trained": n_updates,
            "defender_ppo/lr": self.scheduler.get_last_lr()[0],
        }

    # ── Checkpoint ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  💾 Saved Defender: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def vram_status() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{alloc:.1f}/{total:.1f} GB"


def train(cfg: DefenderPPOConfig) -> None:
    banner = f"""
{'═'*60}
  MEMEX DEFENDER PPO TRAINER  —  Self-Play Ready
  Model:     {cfg.model_name}
  LoRA:      r={cfg.lora_r}  α={cfg.lora_alpha}
  LR:        {cfg.lr}  |  Clip: {cfg.clip_eps}  |  KL: {cfg.kl_coef}
  GAE:       γ={cfg.gamma}  λ={cfg.gae_lambda}
  Scenarios: {cfg.scenario_source} (mix_ratio={cfg.mix_ratio})
  Launderer: {cfg.launderer_checkpoint or 'NONE'}
  Iters:     {cfg.total_iterations}  ×  {cfg.episodes_per_iter} ep/iter
  Dry:       {cfg.dry_run}
{'═'*60}"""
    print(banner)

    # ── 1. Load Model ──
    print("[1/4] Loading Defender model with Unsloth...")
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
        wandb.init(project=cfg.wandb_project, name=f"defender-{int(time.time())}",
                   config=vars(cfg))
    else:
        print("[3/4] WandB SKIPPED (dry-run)")

    # ── 4. Train ──
    print("[4/4] Starting training loop...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo = DefenderPPO(model, tokenizer, cfg, device)
    env = AMLEnvironment()

    # PLR Curriculum
    plr = None
    if cfg.use_plr:
        from curriculum.plr_engine import PLREngine
        plr = PLREngine(buffer_size=cfg.plr_buffer_size)

    iters = 2 if cfg.dry_run else cfg.total_iterations
    eps_per = 1 if cfg.dry_run else cfg.episodes_per_iter
    best_score = -float("inf")

    for it in range(1, iters + 1):
        t0 = time.time()
        all_steps: List[StepData] = []
        ep_stats: List[EpisodeStats] = []

        for ep in range(eps_per):
            # Decide scenario source
            use_launderer = (
                cfg.scenario_source == "mixed"
                and random.random() < cfg.mix_ratio
                and cfg.launderer_checkpoint
            )
            src = "launderer" if use_launderer else "procedural"

            # Select difficulty/typology
            if plr is not None:
                diff, typo = plr.sample_scenario(cfg.difficulties, cfg.typologies)
            else:
                diff = random.choice(cfg.difficulties)
                typo = random.choice(cfg.typologies)

            try:
                # NOTE: For launderer-generated scenarios, the self_play.py
                # orchestrator pre-loads scenarios into the env. In standalone
                # mode, this just uses procedural scenarios.
                steps, stats = ppo.rollout(env, diff, typo, scenario_source=src)
                all_steps.extend(steps)
                ep_stats.append(stats)

                if plr is not None:
                    plr.update(f"iter{it}_ep{ep}_{diff}_{typo}", diff, typo, stats.score)

                print(
                    f"  It {it:>3} Ep {ep+1} | {diff}/{typo} [{src}] | "
                    f"steps={stats.steps:>2} score={stats.score:+.3f} | "
                    f"dec={'✓' if stats.decision_correct else '✗'} "
                    f"typo={'✓' if stats.typology_correct else '✗'} "
                    f"F1={stats.entity_f1:.2f} | "
                    f"PF={stats.page_faults} MI={stats.meta_injections}"
                )

            except Exception as e:
                print(f"  ⚠ Episode failed: {e}")
                continue

        if not all_steps:
            continue

        # PPO Update
        ppo_stats = ppo.ppo_update(all_steps)

        # Aggregate stats
        mean_score = sum(s.score for s in ep_stats) / len(ep_stats)
        dec_acc = sum(1 for s in ep_stats if s.decision_correct) / len(ep_stats)
        typo_acc = sum(1 for s in ep_stats if s.typology_correct) / len(ep_stats)
        mean_f1 = sum(s.entity_f1 for s in ep_stats) / len(ep_stats)
        elapsed = time.time() - t0

        print(
            f"\n  ═══ Iter {it}/{iters} ═══\n"
            f"    Mean score:    {mean_score:+.4f}\n"
            f"    Decision acc:  {dec_acc:.2%}\n"
            f"    Typology acc:  {typo_acc:.2%}\n"
            f"    Entity F1:     {mean_f1:.4f}\n"
            f"    PPO loss:      {ppo_stats['defender_ppo/loss/policy']:.6f}\n"
            f"    KL:            {ppo_stats['defender_ppo/kl']:.6f}\n"
            f"    Entropy:       {ppo_stats['defender_ppo/entropy']:.4f}\n"
            f"    VRAM: {vram_status()}  |  {elapsed:.0f}s\n"
        )

        # WandB
        if not cfg.dry_run:
            import wandb
            # OS Metrics (mandatory)
            mean_pf = sum(s.page_faults for s in ep_stats) / len(ep_stats)
            mean_at = sum(s.async_timeouts for s in ep_stats) / len(ep_stats)
            mean_sp = sum(s.successful_pages for s in ep_stats) / len(ep_stats)
            mean_mi = sum(s.meta_injections for s in ep_stats) / len(ep_stats)
            mean_km = sum(s.kernel_mode_uses for s in ep_stats) / len(ep_stats)

            # Non-degeneracy metrics (TP/TN/FP/FN counts)
            n_ep = len(ep_stats)
            sar_rate = sum(1 for s in ep_stats if s.decision_correct and s.detection == "TP") / max(n_ep, 1)
            close_rate = sum(1 for s in ep_stats if s.decision_correct and s.detection == "TN") / max(n_ep, 1)

            log_dict = {
                "iteration": it,
                "defender/returns/mean": mean_score,
                "defender/decision_accuracy": dec_acc,
                "defender/typology_accuracy": typo_acc,
                "defender/entity_f1": mean_f1,
                # OS metrics (mandatory logging)
                "defender/os/page_faults": mean_pf,
                "defender/os/async_timeouts": mean_at,
                "defender/os/successful_pages": mean_sp,
                "defender/os/meta_injections": mean_mi,
                "defender/os/kernel_mode_uses": mean_km,
                # Non-degeneracy
                "defender/sar_rate": sar_rate,
                "defender/close_rate": close_rate,
                **ppo_stats,
                "perf/iter_seconds": elapsed,
            }
            if plr is not None:
                log_dict.update(plr.get_wandb_metrics())
            wandb.log(log_dict)

        # Checkpoint
        if mean_score > best_score:
            best_score = mean_score
            ppo.save(os.path.join(cfg.output_dir, "best"))
        if it % cfg.save_every == 0:
            ppo.save(os.path.join(cfg.output_dir, f"iter-{it}"))

        # GC
        del all_steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    ppo.save(os.path.join(cfg.output_dir, "final"))
    print(f"\n{'═'*60}")
    print(f"  DEFENDER TRAINING COMPLETE  |  Best: {best_score:+.4f}")
    print(f"{'═'*60}\n")

    if not cfg.dry_run:
        import wandb
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex Defender PPO Trainer (Self-Play)")
    p.add_argument("--model", default=DefenderPPOConfig.model_name)
    p.add_argument("--lr", type=float, default=DefenderPPOConfig.lr)
    p.add_argument("--lora-r", type=int, default=DefenderPPOConfig.lora_r)
    p.add_argument("--episodes", type=int, default=DefenderPPOConfig.episodes_per_iter)
    p.add_argument("--iterations", type=int, default=DefenderPPOConfig.total_iterations)
    p.add_argument("--temperature", type=float, default=DefenderPPOConfig.temperature)
    p.add_argument("--scenario-source", choices=["procedural", "mixed"], default="procedural")
    p.add_argument("--launderer-checkpoint", type=str, default="")
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--wandb-project", default=DefenderPPOConfig.wandb_project)
    p.add_argument("--output-dir", default=DefenderPPOConfig.output_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-plr", action="store_true", default=False)
    args = p.parse_args()

    train(DefenderPPOConfig(
        model_name=args.model, lr=args.lr, lora_r=args.lora_r,
        episodes_per_iter=args.episodes, total_iterations=args.iterations,
        temperature=args.temperature,
        scenario_source=args.scenario_source,
        launderer_checkpoint=args.launderer_checkpoint,
        mix_ratio=args.mix_ratio,
        wandb_project=args.wandb_project, output_dir=args.output_dir,
        dry_run=args.dry_run, use_plr=args.use_plr,
    ))


if __name__ == "__main__":
    main()
