# Memex PPO Training — Setup & Colab/Kaggle Guide

## Colab/Kaggle Installation Cell (Copy-Paste Ready)

```python
%%capture
# ═══════════════════════════════════════════════════════════
# CELL 1: Install Memex PPO Training Stack
# Runtime: GPU → T4 (free) or A100 (Pro)
# ═══════════════════════════════════════════════════════════

import torch
major, minor = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name(0)} | Compute: {major}.{minor}")

# Unsloth — 4-bit quantized model loading + LoRA
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Flash Attention (only for Ampere+, skip on T4)
if major >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"

# TRL (PPO reference), PEFT, Accelerate, BitsAndBytes
!pip install --no-deps trl peft accelerate bitsandbytes

# Experiment tracking + project deps
!pip install wandb pydantic>=2.0.0

# Verify imports
from unsloth import FastLanguageModel
import trl, peft, wandb
print(f"✓ Unsloth + TRL {trl.__version__} + PEFT {peft.__version__} ready")
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 2: Upload the Memex project or mount drive
# ═══════════════════════════════════════════════════════════

# Option A: Clone from GitHub
# !git clone https://github.com/YOUR_ORG/MetaHack.git
# %cd MetaHack

# Option B: Upload zip and extract
# from google.colab import files
# uploaded = files.upload()
# !unzip MetaHack.zip
# %cd MetaHack

# Option C: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/MetaHack
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 3: Dry-run validation (2 iterations, no WandB)
# ═══════════════════════════════════════════════════════════

!python train_ppo.py --dry-run
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 4: Full training run
# ═══════════════════════════════════════════════════════════

import wandb
wandb.login()

!python train_ppo.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --lr 5e-6 \
    --lora-r 16 \
    --episodes 4 \
    --iterations 50 \
    --temperature 0.3 \
    --wandb-project memex-ppo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5: Evaluate best checkpoint
# ═══════════════════════════════════════════════════════════

!python train_ppo.py --eval checkpoints/best
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 6: Demo (1MDB) — scripted or with trained model
# ═══════════════════════════════════════════════════════════

# Scripted demo (no GPU needed)
!python demo_eval.py --dry-run

# With trained model
# !python demo_eval.py --model checkpoints/best

# Download replay files for the Next.js frontend
# from google.colab import files
# !zip -r demo_output.zip demo_output/
# files.download('demo_output.zip')
```

---

## Hardware Requirements

| Resource | T4 (Free Tier) | A100 (Pro) |
|----------|---------------|------------|
| VRAM | 15 GB | 40+ GB |
| Est. per iteration | ~3 min | ~1 min |
| 50 iterations | ~2.5 hrs | ~50 min |
| Fits 8B 4-bit | ✓ (~10 GB peak) | ✓ |

## WandB Metrics

| Metric | Healthy Range | What to Watch |
|--------|---------------|---------------|
| `ppo/returns/mean` | 0.0 → +0.8 | Main training signal |
| `ppo/loss/policy` | Decreasing | Should stabilize |
| `ppo/kl` | < 0.5 | Spike = policy diverging |
| `os/page_faults` | Decreasing | Agent learning memory management |
| `os/async_timeouts` | → 0 | Agent learning patience |

## File Reference

| File | Purpose |
|------|---------|
| `train_ppo.py` | Custom PPO trainer (Unsloth + LoRA, T4-optimized) |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `server/aml_environment.py` | Memex environment (reset/step API) |
| `scenarios/procedural_generator.py` | Dynamic scenario generation |
| `graders/grader.py` | Dense reward matrix (-1.0 to +1.0) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async, Kernel) |
| `tests/test_smoke.py` | Environment verification (7/7 tests) |
| `demo_output/` | AGUI JSON payloads for frontend replay |
| `checkpoints/` | Saved LoRA adapters + tokenizer |
