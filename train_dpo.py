#!/usr/bin/env python3
"""
Memex OS-Agent — Offline DPO Trainer (Continuous Learning Engine).

Pulls human preference pairs from the frontend's SQLite database,
runs Direct Preference Optimization (DPO) on existing LoRA weights,
and saves updated adapters for hot-swapping into production inference.

Architecture:
  ┌─ SQLite (Prisma) ─┐     ┌─ DPO Loss ───────────────────────┐
  │ preference_pairs   │ ──→ │ β=0.1  ×  log σ(β(π_θ - π_ref)) │
  │ (chosen/rejected)  │     │ on existing LoRA weights          │
  └────────────────────┘     └──────────────┬────────────────────┘
                                            │
                                   ┌────────▼────────┐
                                   │ Updated LoRA     │
                                   │ adapters on disk │
                                   └─────────────────┘

Usage:
  # Pull pending pairs and run DPO (default: 3 epochs, β=0.1)
  python train_dpo.py --base-model checkpoints/best --db frontend/prisma/dev.db

  # Custom settings
  python train_dpo.py --base-model checkpoints/best --db frontend/prisma/dev.db \\
      --beta 0.1 --epochs 3 --lr 1e-6 --output checkpoints/dpo-latest
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DPOConfig:
    base_model: str = os.path.join(PROJECT_ROOT, "checkpoints", "best")
    db_path: str = os.path.join(PROJECT_ROOT, "frontend", "prisma", "dev.db")
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "dpo-latest")
    max_seq_length: int = 2048
    beta: float = 0.1        # DPO temperature — lower = more aggressive preference
    lr: float = 1e-6          # Conservative LR to avoid catastrophic forgetting
    epochs: int = 3
    max_pairs: int = 500      # Max pairs per training run
    min_pairs: int = 5        # Minimum pairs required to start training
    batch_size: int = 1       # Process one pair at a time (VRAM-safe)
    wandb_project: str = "memex-dpo"


# ═══════════════════════════════════════════════════════════════════════
# Database Interface
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PreferencePair:
    id: str
    original_prompt: str
    rejected_response: str
    chosen_response: str
    difficulty: str
    typology: str


def fetch_pending_pairs(db_path: str, limit: int = 500) -> List[PreferencePair]:
    """Pull unconsumed preference pairs from the Prisma SQLite database."""
    if not os.path.exists(db_path):
        print(f"  ⚠ Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, originalPrompt, rejectedResponse, chosenResponse,
               difficulty, typology
        FROM PreferencePair
        WHERE consumed = 0
        ORDER BY createdAt ASC
        LIMIT ?
        """,
        (limit,),
    )

    pairs = [
        PreferencePair(
            id=row["id"],
            original_prompt=row["originalPrompt"],
            rejected_response=row["rejectedResponse"],
            chosen_response=row["chosenResponse"],
            difficulty=row["difficulty"],
            typology=row["typology"],
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return pairs


def mark_pairs_consumed(db_path: str, pair_ids: List[str], run_id: str) -> int:
    """Mark preference pairs as consumed by this training run."""
    if not pair_ids:
        return 0
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in pair_ids)
    conn.execute(
        f"""
        UPDATE PreferencePair
        SET consumed = 1, consumedByRunId = ?, consumedAt = datetime('now')
        WHERE id IN ({placeholders})
        """,
        [run_id] + pair_ids,
    )
    conn.commit()
    count = conn.total_changes
    conn.close()
    return count


def record_training_run(
    db_path: str, run_id: str, pairs_used: int,
    final_loss: Optional[float], checkpoint_path: Optional[str],
) -> None:
    """Record this DPO training run in the database for audit."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO DPOTrainingRun (id, createdAt, pairsUsed, finalLoss, checkpointPath, hotSwapped)
        VALUES (?, datetime('now'), ?, ?, ?, 0)
        """,
        (run_id, pairs_used, final_loss, checkpoint_path),
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# DPO Loss
# ═══════════════════════════════════════════════════════════════════════

def compute_log_prob(model, tokenizer, prompt: str, response: str, device: str) -> torch.Tensor:
    """Compute mean per-token log probability for a prompt+response pair."""
    full_text = prompt + response
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(device)

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1536,
    )["input_ids"]
    resp_start = prompt_ids.shape[1]

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    # Only score the response tokens
    if resp_start >= logits.shape[0]:
        return torch.tensor(0.0, device=device)

    resp_logits = logits[resp_start - 1 : -1, :]
    resp_ids = inputs["input_ids"][0, resp_start:]

    if resp_ids.shape[0] == 0 or resp_logits.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    # Align shapes
    min_len = min(resp_logits.shape[0], resp_ids.shape[0])
    resp_logits = resp_logits[:min_len]
    resp_ids = resp_ids[:min_len]

    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_lp = log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
    return token_lp.mean()


def dpo_loss(
    model, ref_model_fn, tokenizer, prompt: str,
    chosen: str, rejected: str, beta: float, device: str,
) -> torch.Tensor:
    """Compute DPO loss for a single preference pair.

    L_DPO = -log σ(β × ((π_θ(chosen) - π_ref(chosen)) - (π_θ(rejected) - π_ref(rejected))))

    Args:
        model: The policy model (with LoRA adapters enabled).
        ref_model_fn: Callable that computes reference log probs (adapters disabled).
        tokenizer: The tokenizer.
        prompt: The original prompt text.
        chosen: The human-corrected (preferred) response.
        rejected: The agent's original (bad) response.
        beta: DPO temperature parameter.
        device: Target device.
    """
    # Policy log probs (WITH gradient)
    inputs_chosen = tokenizer(prompt + chosen, return_tensors="pt", truncation=True, max_length=2048).to(device)
    inputs_rejected = tokenizer(prompt + rejected, return_tensors="pt", truncation=True, max_length=2048).to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)["input_ids"]
    resp_start = prompt_ids.shape[1]

    # Chosen
    logits_c = model(**inputs_chosen).logits[0]
    resp_c_ids = inputs_chosen["input_ids"][0, resp_start:]
    if resp_c_ids.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    min_c = min(logits_c[resp_start - 1:-1].shape[0], resp_c_ids.shape[0])
    lp_c = F.log_softmax(logits_c[resp_start - 1:resp_start - 1 + min_c], dim=-1)
    pi_chosen = lp_c.gather(1, resp_c_ids[:min_c].unsqueeze(1)).squeeze(1).mean()

    # Rejected
    logits_r = model(**inputs_rejected).logits[0]
    resp_r_ids = inputs_rejected["input_ids"][0, resp_start:]
    if resp_r_ids.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    min_r = min(logits_r[resp_start - 1:-1].shape[0], resp_r_ids.shape[0])
    lp_r = F.log_softmax(logits_r[resp_start - 1:resp_start - 1 + min_r], dim=-1)
    pi_rejected = lp_r.gather(1, resp_r_ids[:min_r].unsqueeze(1)).squeeze(1).mean()

    # Reference log probs (NO gradient — adapters disabled)
    ref_chosen = ref_model_fn(prompt, chosen)
    ref_rejected = ref_model_fn(prompt, rejected)

    # DPO loss: -log σ(β × (Δ_chosen - Δ_rejected))
    delta = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    loss = -F.logsigmoid(delta)
    return loss


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: DPOConfig) -> None:
    """Run offline DPO training on pending preference pairs."""
    run_id = f"dpo-{int(time.time())}"

    print(f"\n{'═'*60}")
    print(f"  MEMEX DPO TRAINER — Continuous Learning")
    print(f"  Base:   {cfg.base_model}")
    print(f"  DB:     {cfg.db_path}")
    print(f"  β={cfg.beta}  |  LR={cfg.lr}  |  Epochs={cfg.epochs}")
    print(f"  Run:    {run_id}")
    print(f"{'═'*60}\n")

    # ── 1. Fetch pairs ──
    print("[1/4] Fetching preference pairs...")
    pairs = fetch_pending_pairs(cfg.db_path, limit=cfg.max_pairs)
    print(f"  Found {len(pairs)} unconsumed pairs")

    if len(pairs) < cfg.min_pairs:
        print(f"  ⚠ Need at least {cfg.min_pairs} pairs. Exiting.")
        return

    # ── 2. Load model ──
    print("[2/4] Loading model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ✓ Loaded on {device}")

    # Reference log prob function (adapters disabled)
    def ref_log_prob(prompt: str, response: str) -> torch.Tensor:
        model.eval()
        with model.disable_adapter():
            lp = compute_log_prob(model, tokenizer, prompt, response, device)
        model.train()
        return lp.detach()

    # ── 3. Train ──
    print("[3/4] Running DPO training...\n")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01,
    )

    model.train()
    final_loss = 0.0

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for i, pair in enumerate(pairs):
            loss = dpo_loss(
                model, ref_log_prob, tokenizer,
                prompt=pair.original_prompt,
                chosen=pair.chosen_response,
                rejected=pair.rejected_response,
                beta=cfg.beta, device=device,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{cfg.epochs} | Pair {i+1}/{len(pairs)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(len(pairs), 1)
        final_loss = avg_loss
        print(f"  ── Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}\n")

    # ── 4. Save & record ──
    print("[4/4] Saving updated adapters...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"  💾 Saved: {cfg.output_dir}")

    # Mark pairs as consumed
    pair_ids = [p.id for p in pairs]
    marked = mark_pairs_consumed(cfg.db_path, pair_ids, run_id)
    print(f"  ✓ Marked {marked} pairs as consumed")

    # Record run
    record_training_run(cfg.db_path, run_id, len(pairs), final_loss, cfg.output_dir)
    print(f"  ✓ Training run recorded: {run_id}")

    print(f"\n{'═'*60}")
    print(f"  DPO COMPLETE | {len(pairs)} pairs | Loss: {final_loss:.4f}")
    print(f"  Hot-swap: python hotswap.py --adapter {cfg.output_dir}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Memex DPO Trainer")
    p.add_argument("--base-model", default=DPOConfig.base_model)
    p.add_argument("--db", default=DPOConfig.db_path)
    p.add_argument("--output", default=DPOConfig.output_dir)
    p.add_argument("--beta", type=float, default=DPOConfig.beta)
    p.add_argument("--lr", type=float, default=DPOConfig.lr)
    p.add_argument("--epochs", type=int, default=DPOConfig.epochs)
    p.add_argument("--max-pairs", type=int, default=DPOConfig.max_pairs)
    p.add_argument("--wandb-project", default=DPOConfig.wandb_project)
    args = p.parse_args()

    train(DPOConfig(
        base_model=args.base_model, db_path=args.db,
        output_dir=args.output, beta=args.beta,
        lr=args.lr, epochs=args.epochs,
        max_pairs=args.max_pairs, wandb_project=args.wandb_project,
    ))


if __name__ == "__main__":
    main()
