#!/usr/bin/env python3
"""
Memex OS-Agent Benchmark — Self-Play Orchestrator.

Implements alternating best-response PPO training:

  Phase 1: Defender warm-start on procedural scenarios (N iterations)
  Phase 2: Freeze Defender → Train Launderer to fool it (M iterations)
  Phase 3: Freeze Launderer → Train Defender on mixed scenarios (N iterations)
  Phase 4+: Repeat Phases 2-3 (outer loop)

Critical constraint: Only one 8B model is loaded at a time (L4 has ~22 GB).
Model swapping is done via LoRA checkpoint save/load.

Usage:
    python self_play.py --dry-run
    python self_play.py --outer-rounds 3 --defender-iters 30 --launderer-iters 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Self-Play Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SelfPlayConfig:
    """Configuration for the self-play training loop."""

    # Model (shared base for both agents)
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048

    # Self-play schedule
    outer_rounds: int = 3       # number of alternating rounds
    defender_warmup_iters: int = 20   # Phase 1 only
    defender_iters: int = 15          # Phases 3+
    launderer_iters: int = 10         # Phases 2+
    defender_episodes_per_iter: int = 4
    launderer_episodes_per_iter: int = 4

    # Mixing ratio for Phase 3+ (fraction of Launderer scenarios)
    mix_ratio_start: float = 0.3   # Phase 3
    mix_ratio_schedule: str = "linear"  # or "fixed"
    mix_ratio_max: float = 0.7    # Final phase

    # Checkpoint management
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    keep_last_n: int = 3  # Keep last N checkpoints per agent

    # Infrastructure
    wandb_project: str = "memex-selfplay"
    dry_run: bool = False

    @property
    def defender_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, "defender")

    @property
    def launderer_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, "launderer")


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Population
# ═══════════════════════════════════════════════════════════════════════

class CheckpointPopulation:
    """Manages checkpoint populations for both agents.

    Keeps track of the N most recent checkpoints and the overall best
    for each agent, enabling population-based training in later rounds.
    """

    def __init__(self, base_dir: str, keep_n: int = 3):
        self.base_dir = base_dir
        self.keep_n = keep_n
        self._history: List[Dict[str, Any]] = []

    def register(self, agent: str, round_id: int, path: str, score: float) -> None:
        self._history.append({
            "agent": agent,
            "round": round_id,
            "path": path,
            "score": score,
            "timestamp": time.time(),
        })

    def best(self, agent: str) -> Optional[str]:
        """Get the path to the best checkpoint for an agent."""
        entries = [e for e in self._history if e["agent"] == agent]
        if not entries:
            return None
        return max(entries, key=lambda e: e["score"])["path"]

    def latest(self, agent: str) -> Optional[str]:
        entries = [e for e in self._history if e["agent"] == agent]
        if not entries:
            return None
        return max(entries, key=lambda e: e["timestamp"])["path"]

    def to_json(self) -> str:
        return json.dumps(self._history, indent=2)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


# ═══════════════════════════════════════════════════════════════════════
# Model Management
# ═══════════════════════════════════════════════════════════════════════

def unload_model() -> None:
    """Force-unload the current model from GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def vram_status() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{alloc:.1f}/{total:.1f} GB"


# ═══════════════════════════════════════════════════════════════════════
# Phase Runners
# ═══════════════════════════════════════════════════════════════════════

def run_defender_phase(
    cfg: SelfPlayConfig,
    iterations: int,
    scenario_source: str = "procedural",
    launderer_ckpt: str = "",
    mix_ratio: float = 0.0,
    round_id: int = 0,
    phase_label: str = "Phase1",
) -> float:
    """Train Defender for N iterations and return best score."""
    from train_defender_ppo import DefenderPPOConfig, train

    output_dir = os.path.join(cfg.defender_dir, f"round-{round_id}")

    defender_cfg = DefenderPPOConfig(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        total_iterations=iterations,
        episodes_per_iter=cfg.defender_episodes_per_iter,
        scenario_source=scenario_source,
        launderer_checkpoint=launderer_ckpt,
        mix_ratio=mix_ratio,
        output_dir=output_dir,
        wandb_project=cfg.wandb_project,
        dry_run=cfg.dry_run,
    )

    print(f"\n{'━'*60}")
    print(f"  {phase_label}: DEFENDER TRAINING (Round {round_id})")
    print(f"    Iterations: {iterations}  |  Source: {scenario_source}")
    print(f"    Mix ratio: {mix_ratio:.2f}  |  Launderer: {launderer_ckpt or 'N/A'}")
    print(f"{'━'*60}\n")

    train(defender_cfg)
    unload_model()

    # Score approximation: read best checkpoint metric if available
    # For now, return 0.0 (actual score tracked via WandB)
    return 0.0


def run_launderer_phase(
    cfg: SelfPlayConfig,
    iterations: int,
    defender_ckpt: str = "",
    round_id: int = 0,
    phase_label: str = "Phase2",
) -> float:
    """Train Launderer for M iterations and return best reward."""
    from train_launderer_ppo import LaundererPPOConfig, train

    output_dir = os.path.join(cfg.launderer_dir, f"round-{round_id}")

    launderer_cfg = LaundererPPOConfig(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length + 2048,  # Launderer needs more context
        total_iterations=iterations,
        episodes_per_iter=cfg.launderer_episodes_per_iter,
        defender_checkpoint=defender_ckpt,
        output_dir=output_dir,
        wandb_project=cfg.wandb_project,
        dry_run=cfg.dry_run,
    )

    print(f"\n{'━'*60}")
    print(f"  {phase_label}: LAUNDERER TRAINING (Round {round_id})")
    print(f"    Iterations: {iterations}  |  Defender: {defender_ckpt or 'N/A'}")
    print(f"{'━'*60}\n")

    train(launderer_cfg)
    unload_model()

    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Main Self-Play Loop
# ═══════════════════════════════════════════════════════════════════════

def self_play(cfg: SelfPlayConfig) -> None:
    """Run the full alternating best-response self-play loop."""

    banner = f"""
{'═'*60}
  MEMEX SELF-PLAY ORCHESTRATOR
  Base model:     {cfg.base_model}
  Outer rounds:   {cfg.outer_rounds}
  Schedule:       Warmup({cfg.defender_warmup_iters}) →
                  [L({cfg.launderer_iters}) → D({cfg.defender_iters})] × {cfg.outer_rounds}
  Mix ratio:      {cfg.mix_ratio_start} → {cfg.mix_ratio_max} ({cfg.mix_ratio_schedule})
  Dry:            {cfg.dry_run}
{'═'*60}"""
    print(banner)

    population = CheckpointPopulation(cfg.checkpoint_dir, keep_n=cfg.keep_last_n)

    # ── Phase 1: Defender Warm-Start ──
    print("\n" + "="*60)
    print("  PHASE 1: DEFENDER WARM-START (Procedural Only)")
    print("="*60)

    phase1_score = run_defender_phase(
        cfg, iterations=cfg.defender_warmup_iters,
        scenario_source="procedural",
        round_id=0, phase_label="Phase 1",
    )
    defender_ckpt = os.path.join(cfg.defender_dir, "round-0", "best")
    population.register("defender", 0, defender_ckpt, phase1_score)

    # ── Phases 2-3 (Alternating Loop) ──
    for rnd in range(1, cfg.outer_rounds + 1):
        # Compute mix ratio for this round
        if cfg.mix_ratio_schedule == "linear":
            t = rnd / max(cfg.outer_rounds, 1)
            mix_ratio = cfg.mix_ratio_start + t * (cfg.mix_ratio_max - cfg.mix_ratio_start)
        else:
            mix_ratio = cfg.mix_ratio_start

        print(f"\n{'▓'*60}")
        print(f"  OUTER ROUND {rnd}/{cfg.outer_rounds}  |  Mix ratio: {mix_ratio:.2f}")
        print(f"{'▓'*60}")

        # Phase 2: Train Launderer against frozen Defender
        phase2_score = run_launderer_phase(
            cfg, iterations=cfg.launderer_iters,
            defender_ckpt=population.best("defender") or defender_ckpt,
            round_id=rnd, phase_label=f"Phase 2 (Round {rnd})",
        )
        launderer_ckpt = os.path.join(cfg.launderer_dir, f"round-{rnd}", "best")
        population.register("launderer", rnd, launderer_ckpt, phase2_score)

        # Phase 3: Train Defender against mixed scenarios
        phase3_score = run_defender_phase(
            cfg, iterations=cfg.defender_iters,
            scenario_source="mixed",
            launderer_ckpt=population.best("launderer") or launderer_ckpt,
            mix_ratio=mix_ratio,
            round_id=rnd, phase_label=f"Phase 3 (Round {rnd})",
        )
        defender_ckpt = os.path.join(cfg.defender_dir, f"round-{rnd}", "best")
        population.register("defender", rnd, defender_ckpt, phase3_score)

    # ── Save population history ──
    pop_path = os.path.join(cfg.checkpoint_dir, "population_history.json")
    population.save(pop_path)

    print(f"\n{'═'*60}")
    print(f"  SELF-PLAY COMPLETE")
    print(f"  Best Defender:  {population.best('defender')}")
    print(f"  Best Launderer: {population.best('launderer')}")
    print(f"  Population:     {pop_path}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex Self-Play Orchestrator")
    p.add_argument("--base-model", default=SelfPlayConfig.base_model)
    p.add_argument("--outer-rounds", type=int, default=SelfPlayConfig.outer_rounds)
    p.add_argument("--defender-warmup", type=int, default=SelfPlayConfig.defender_warmup_iters)
    p.add_argument("--defender-iters", type=int, default=SelfPlayConfig.defender_iters)
    p.add_argument("--launderer-iters", type=int, default=SelfPlayConfig.launderer_iters)
    p.add_argument("--defender-episodes", type=int, default=SelfPlayConfig.defender_episodes_per_iter)
    p.add_argument("--launderer-episodes", type=int, default=SelfPlayConfig.launderer_episodes_per_iter)
    p.add_argument("--mix-start", type=float, default=SelfPlayConfig.mix_ratio_start)
    p.add_argument("--mix-max", type=float, default=SelfPlayConfig.mix_ratio_max)
    p.add_argument("--mix-schedule", choices=["linear", "fixed"], default="linear")
    p.add_argument("--checkpoint-dir", default=SelfPlayConfig.checkpoint_dir)
    p.add_argument("--wandb-project", default=SelfPlayConfig.wandb_project)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    self_play(SelfPlayConfig(
        base_model=args.base_model,
        outer_rounds=args.outer_rounds,
        defender_warmup_iters=args.defender_warmup,
        defender_iters=args.defender_iters,
        launderer_iters=args.launderer_iters,
        defender_episodes_per_iter=args.defender_episodes,
        launderer_episodes_per_iter=args.launderer_episodes,
        mix_ratio_start=args.mix_start,
        mix_ratio_max=args.mix_max,
        mix_ratio_schedule=args.mix_schedule,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
