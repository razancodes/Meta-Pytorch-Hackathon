# Memex PPO Training Guide

> Complete training pipeline for the Memex OS-Agent Benchmark.
> Supports two tiers: **T4 (8B)** for prototyping and **A100 cluster (70B)** for production.

---

## Prerequisites

| Dependency | Purpose |
|-----------|---------|
| `unsloth` | 4-bit NF4 quantization + fast LoRA |
| `peft` | LoRA adapter management |
| `bitsandbytes` | Quantization backend |
| `deepspeed` | Multi-GPU sharding (70B only) |
| `wandb` | Experiment tracking |
| `pydantic>=2.0` | Environment type contracts |

---

## Tier 1: T4 Training (8B Model)

**Target:** Google Colab Free / Kaggle T4 (15 GB VRAM)
**Script:** `train_ppo.py`
**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Peak VRAM:** ~10 GB / 15 GB

### Colab Setup (Copy-Paste Cells)

```python
%%capture
# ═══════════════════════════════════════════════════════════
# CELL 1: Install Training Stack
# Runtime → GPU → T4 (free) or A100 (Pro)
# ═══════════════════════════════════════════════════════════

import torch
major, minor = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name(0)} | Compute: {major}.{minor}")

# Unsloth — 4-bit quantized model loading + LoRA
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Flash Attention (Ampere+ only, skip on T4)
if major >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"

# RL + adapter stack
!pip install --no-deps trl peft accelerate bitsandbytes

# Tracking + project deps
!pip install wandb pydantic>=2.0.0

# Verify
from unsloth import FastLanguageModel
import trl, peft, wandb
print(f"✓ Unsloth + TRL {trl.__version__} + PEFT {peft.__version__} ready")
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 2: Clone the project
# ═══════════════════════════════════════════════════════════

!git clone https://github.com/razancodes/Meta-Pytorch-Hackathon.git
%cd Meta-Pytorch-Hackathon
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 3: Verify environment (no GPU needed)
# ═══════════════════════════════════════════════════════════

!python tests/test_smoke.py
# Expected: 7/7 tests passed ✓
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 4: Dry-run (2 iterations, 1 episode, no WandB)
# ═══════════════════════════════════════════════════════════

!python train_ppo.py --dry-run
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5: Full training (~2.5 hours on T4)
# ═══════════════════════════════════════════════════════════

import wandb
wandb.login()

!python train_ppo.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --lr 5e-6 \
    --lora-r 16 \
    --episodes 4 \
    --iterations 50 \
    --temperature 0.5 \
    --wandb-project memex-ppo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 6: Evaluate best checkpoint (9 combos)
# ═══════════════════════════════════════════════════════════

!python train_ppo.py --eval checkpoints/best
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 7: Run 1MDB demo + download AGUI replay
# ═══════════════════════════════════════════════════════════

# Scripted (deterministic, no GPU)
!python demo_eval.py --dry-run

# With trained model
# !python demo_eval.py --model checkpoints/best

# Download replay files for Next.js frontend
# from google.colab import files
# !zip -r demo_output.zip demo_output/
# files.download('demo_output.zip')
```

---

## Tier 2: A100 Cluster Training (70B Model)

**Target:** Multi-node A100-80GB (on-site compute / cloud)
**Script:** `train_ppo_70b.py`
**Model:** `meta-llama/Meta-Llama-3.1-70B-Instruct`
**Backend:** DeepSpeed ZeRO-3
**Peak VRAM:** ~50 GB / 80 GB per GPU

### VRAM Budget (4× A100-80GB)

| Component | Total | Per-GPU (sharded) |
|-----------|-------|-------------------|
| Weights (4-bit NF4) | ~35 GB | ~9 GB |
| LoRA adapters (r=32) | ~150 MB | replicated |
| Optimizer (fp32 AdamW) | ~280 GB | ~70 GB |
| Gradients (bf16) | ~140 GB | ~35 GB |
| **Peak** | — | **~50 GB / 80 GB** |

### Launch Commands

```bash
# Install (on each node)
pip install deepspeed unsloth trl peft accelerate bitsandbytes wandb

# 4-GPU single-node
deepspeed --num_gpus 4 train_ppo_70b.py \
    --iterations 50 --episodes 2

# 8-GPU multi-node (2 nodes × 4 GPUs)
deepspeed --num_gpus 4 --num_nodes 2 --hostfile hosts.txt \
    train_ppo_70b.py --iterations 50 --episodes 2

# With CPU offloading (fewer GPUs, slower)
deepspeed --num_gpus 2 train_ppo_70b.py \
    --offload-optimizer --offload-params

# Dry-run
deepspeed --num_gpus 4 train_ppo_70b.py --dry-run

# Evaluate
python train_ppo_70b.py --eval checkpoints_70b/best
```

### Key Architectural Details

1. **Rank-0 environment:** Only GPU 0 runs the AML environment. Trajectories are broadcast to all ranks via `torch.distributed.broadcast_object_list()`.
2. **`disable_adapter()` KL trick:** Works under ZeRO-3 — LoRA operates at module level, independent of parameter sharding. No second model copy needed.
3. **DeepSpeed engine:** Handles `backward()`, gradient accumulation, all-reduce, and clipping internally.
4. **Checkpoints:** `engine.save_checkpoint()` saves sharded optimizer states per-rank. Tokenizer saved by rank 0 only.

---

## PPO Stability Engineering

Both trainers include **10 production-grade safety features** to prevent policy collapse during long-horizon RL training:

### Mathematical Fixes

| Feature | What | Why |
|---------|------|-----|
| **Mean log-prob** | `token_lp.mean()` instead of `.sum()` | KL divergence is now scale-invariant — same magnitude regardless of whether responses are 40 or 200 tokens |
| **Ratio clamping** | `clamp(log_ratio, -10, 10)` before `exp()` | Prevents inf/NaN from policy drift between PPO epochs |
| **Entropy bonus** | `- entropy_coef × H(π)` in loss | Keeps the policy exploring; prevents collapsing to a single degenerate action |
| **Return clipping** | `clip(returns, -2.0, +2.0)` | Bounds gradient signals from outlier terminal rewards |
| **Empty response guard** | Dummy EOS if model generates 0 tokens | Prevents NaN from `.mean()` on an empty tensor |

### Hyperparameters

| Parameter | T4 (8B) | A100 (70B) | Purpose |
|-----------|---------|------------|----------|
| `lr` | `5e-6` | `2e-6` | Lower LR for 70B parametric stability |
| `kl_coef` | `0.05` | `0.03` | KL penalty weight against frozen base |
| `entropy_coef` | `0.01` | `0.01` | Exploration bonus |
| `clip_eps` | `0.2` | `0.2` | Standard PPO clipping |
| `reward_clip` | `2.0` | `2.0` | Return clipping bound |
| `grad_accum_steps` | `4` | `8` | Effective batch size |
| `max_grad_norm` | `1.0` | `0.5` | Tighter gradient clipping for 70B |

---

## WandB Monitoring

| Metric | Healthy Range | What to Watch |
|--------|---------------|---------------|
| `ppo/returns/mean` | 0.0 → +0.8 over training | Main signal — should increase |
| `ppo/loss/policy` | Decreasing, then stable | Convergence indicator |
| `ppo/kl` | -1.0 to +1.0 | Values outside [-2, +2] = policy drifting from base |
| `ppo/entropy` | Slowly decreasing, never 0 | Drop to 0 = mode collapse — increase `entropy_coef` |
| `os/page_faults` | Decreasing → 0 | Agent learning memory management |
| `os/async_timeouts` | Decreasing → 0 | Agent learning to wait for async I/O |
| `os/successful_pages` | Increasing | Agent using disk writes proactively |
| `os/meta_injections` | ≥ 1 per episode | Agent injecting compliance rules |

---

## Hardware Reference

| Config | GPU | VRAM | Script | Model | Est. 50 iters |
|--------|-----|------|--------|-------|---------------|
| Colab Free | T4 | 15 GB | `train_ppo.py` | 8B 4-bit | ~2.5 hrs |
| Colab Pro | A100-40GB | 40 GB | `train_ppo.py` | 8B 4-bit | ~50 min |
| On-site | 4× A100-80GB | 320 GB | `train_ppo_70b.py` | 70B 4-bit | ~4 hrs |
| On-site | 8× A100-80GB | 640 GB | `train_ppo_70b.py` | 70B 4-bit | ~2 hrs |

---

## File Reference

| File | Purpose |
|------|---------|
| `train_ppo.py` | Step-level PPO (Unsloth 4-bit + LoRA, T4-optimized) |
| `train_ppo_70b.py` | Multi-GPU PPO (DeepSpeed ZeRO-3, A100 cluster) |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `server/aml_environment.py` | Core environment (15 tools + OS mechanics) |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario builder |
| `graders/grader.py` | Dense reward engine (per-step + terminal) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async Queue, Kernel) |
| `models.py` | Pydantic type definitions |
| `tests/test_smoke.py` | Environment verification (7/7 tests) |
| `checkpoints/` | T4 training output (LoRA adapters + tokenizer) |
| `checkpoints_70b/` | 70B training output (DeepSpeed sharded checkpoints) |
| `demo_output/` | AGUI JSON payloads for frontend replay |
