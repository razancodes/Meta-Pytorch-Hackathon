# Memex PPO Training — Complete Setup & Testing Guide

## Prerequisites

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB (single A100/H100) | 24 GB+ (A100 40GB) |
| System RAM | 32 GB | 64 GB |
| Disk | 30 GB (model + checkpoints) | 50 GB |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.1+ |

> [!NOTE]
> Unsloth's 4-bit quantization + LoRA keeps peak VRAM around **12-14 GB** for an 8B model.
> A single RTX 4090 (24 GB) or A100 (40 GB) is sufficient.

---

## Step 1: Install Dependencies

```bash
cd /home/Muaz/Documents/Software/MetaHack

# Create a training-specific venv (isolate CUDA deps from the env server)
python -m venv .venv-train
source .venv-train/bin/activate

# Core ML stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Unsloth (4-bit quantized inference + LoRA training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# TRL (PPO Trainer)
pip install trl>=0.12.0

# Experiment tracking
pip install wandb

# Project dependencies
pip install pydantic>=2.0.0 fastapi uvicorn httpx
```

### Verify installation:
```bash
python -c "
from unsloth import FastLanguageModel
from trl import PPOTrainer, PPOConfig
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print('✓ All imports OK')
"
```

---

## Step 2: Dry Run (No GPU Required for Smoke Test)

The dry run validates the full pipeline (environment reset → trajectory collection → PPO update) with 2 iterations and no WandB. Use this to catch import errors, API mismatches, and environment bugs before committing GPU hours.

```bash
# Quick validation (CPU-compatible, ~2 minutes)
python train_ppo.py --dry-run

# Expected output:
#   MEMEX OS-AGENT BENCHMARK — PPO TRAINING
#   [1/5] Loading model with Unsloth (4-bit quantized)...  ✓
#   [2/5] Attaching LoRA adapters...  ✓
#   [3/5] Initializing TRL PPOTrainer...  ✓
#   [4/5] WandB SKIPPED (dry-run mode)
#   [5/5] Starting PPO training loop...
#   Iter 1 | Ep 1 | easy/structuring | steps=... | score=...
#   === Iter 1/2 Summary ===
#   ...
#   TRAINING COMPLETE
```

> [!WARNING]
> If you see `OutOfMemoryError` even in dry-run, your GPU has < 12 GB VRAM.
> Solutions:
> 1. Use `--model unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (smaller model)
> 2. Reduce `--batch-size 1`
> 3. Use CPU-only with `CUDA_VISIBLE_DEVICES="" python train_ppo.py --dry-run`

---

## Step 3: Full Training Run

```bash
# Login to WandB (first time only)
wandb login

# Standard training (50 iterations × 4 episodes each = 200 episodes)
python train_ppo.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --lr 5e-6 \
    --lora-r 16 \
    --batch-size 4 \
    --episodes 4 \
    --iterations 50 \
    --temperature 0.3 \
    --wandb-project memex-ppo

# Longer training with more exploration
python train_ppo.py \
    --iterations 200 \
    --episodes 8 \
    --lr 3e-6 \
    --temperature 0.5
```

### Custom model variants:
```bash
# Smaller model (faster, less VRAM)
python train_ppo.py --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit

# Mistral variant
python train_ppo.py --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit
```

---

## Step 4: Monitor Training (WandB Dashboard)

Open your WandB dashboard at `https://wandb.ai/<your-username>/memex-ppo`.

### Key metrics to watch:

| Metric | What it means | Healthy range |
|--------|---------------|---------------|
| `ppo/returns/mean` | Mean episode reward | Should trend from ~0.0 → +0.8 |
| `ppo/loss/total` | PPO policy loss | Should decrease and stabilize |
| `ppo/kl_divergence` | Policy drift from reference | Should stay < 6.0 |
| `os/page_faults` | Memory management failures | Should decrease over training |
| `os/async_timeouts` | Premature async retrieval | Should decrease to ~0 |
| `os/successful_pages` | Disk write mastery | Should increase to ~2-3 per episode |
| `os/meta_injections` | Kernel self-improvement | Should stabilize at ~1 per episode |

### Training curve expectations:
```
Iteration  1-10:   Score ~0.0 to +0.3 (agent learns tool format)
Iteration 10-30:   Score ~0.3 to +0.6 (learns investigation steps)
Iteration 30-50:   Score ~0.6 to +0.9 (masters OS mechanics)
Iteration 50+:     Score ~0.8 to +1.0 (generalizes across typologies)
```

---

## Step 5: Evaluate a Trained Checkpoint

```bash
# Evaluate the best checkpoint across all 9 difficulty×typology combinations
python train_ppo.py --eval checkpoints/best

# Evaluate the final checkpoint
python train_ppo.py --eval checkpoints/final

# Evaluate a specific iteration
python train_ppo.py --eval checkpoints/iter-30

# Expected output:
#   EVALUATION: checkpoints/best
#     easy/structuring     | steps=8  | score=+0.9500 | done=True
#     easy/layering        | steps=10 | score=+0.8200 | done=True
#     ...
#     hard/trade_based_ml  | steps=15 | score=+0.7100 | done=True
#   Mean: +0.8200  |  Min: +0.6800  |  Max: +0.9500
```

---

## Step 6: Run the Environment Smoke Tests (Verify Nothing Broke)

```bash
# Always re-run smoke tests after any changes
python tests/test_smoke.py

# Expected: 7/7 tests passed ✓
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: unsloth` | `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` |
| `CUDA out of memory` | Use `--model unsloth/Llama-3.2-3B-Instruct-bnb-4bit` or `--batch-size 1` |
| `PPO step failed` | Check batch_size ≥ mini_batch_size, ensure enough steps were collected |
| `wandb: Error` | Run `wandb login` or use `--dry-run` to skip WandB |
| `JSON parse errors in responses` | Lower temperature: `--temperature 0.2` |
| `Agent gets stuck in loops` | Increase `gen_repetition_penalty` in `TrainConfig` |

---

## File Reference

| File | Purpose |
|------|---------|
| `train_ppo.py` | Main training script (this doc) |
| `server/aml_environment.py` | The Memex environment (reset/step API) |
| `scenarios/procedural_generator.py` | Dynamic scenario generation (no memorization) |
| `graders/grader.py` | Dense reward matrix |
| `state_manager.py` | OS mechanics (RAM, Disk, Async, Kernel) |
| `tests/test_smoke.py` | Environment verification suite |
| `checkpoints/` | Saved LoRA adapters + tokenizer |
