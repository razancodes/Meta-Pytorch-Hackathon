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
    --iterations 15 \
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
| **Degenerate response detection** | If >80% repeated tokens, assign -0.15 penalty | Detects and penalizes gibberish output |
| **Fault-tolerant env.step()** | try/except around environment step, -0.10 penalty | Malformed actions no longer crash the training loop |
| **Type-safe parse_action()** | Force `params` to dict, catch TypeError/ValueError | Prevents `'str' object has no attribute 'get'` crashes |
| **KL early stopping** | Break PPO epochs if \|KL\| > 15 | Prevents catastrophic gradient updates when policy drifts too far |
| **🔄 Auto-Revert** | Entropy heartbeat monitor + checkpoint reload | Detects mode collapse and automatically reverts to last stable weights |

### Auto-Revert ("Time Machine")

The training loop includes an **Entropy Heartbeat Monitor** that detects irreversible collapse and auto-reverts:

| Trigger | Condition |
|---------|----------|
| Entropy death | `entropy < 0.01` for 2 consecutive iterations |
| Score death | `mean_score ≤ 0.0` for 2 consecutive iterations |
| KL explosion | `\|KL\| > 10.0` in any iteration |

**On revert:**
1. Free old optimizer state + purge VRAM (`del optimizer` → `gc.collect()` → `torch.cuda.empty_cache()`)
2. Reload LoRA weights from last stable checkpoint
3. Bump `entropy_coef × 1.5`, `temperature + 0.1`, `lr × 0.7`
4. Rebuild optimizer with correct remaining cosine schedule
5. (70B) Explicitly update DeepSpeed `engine.optimizer.param_groups` LR

Stable checkpoints are saved only when `entropy > 0.05 AND mean_score > 0.3`. Maximum 5 reverts per run.

> **ZeRO-3 Safety:** On multi-GPU clusters, stable checkpoints use `deepspeed.zero.GatheredParameters` to materialize full tensors from shards before saving. Without this, each rank would save its own 1/N shard — corrupting every checkpoint silently.

### Hyperparameters

| Parameter | T4 (8B) | A100 (70B) | Purpose |
|-----------|---------|------------|----------|
| `lr` | `5e-6` | `2e-6` | Lower LR for 70B parametric stability |
| `kl_coef` | `0.05` | `0.03` | KL penalty weight against frozen base |
| `entropy_coef` | `0.05` | `0.05` | Exploration bonus |
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
| `train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| `train_adversary.py` | GAN-style adversarial battle loop for generating DPO scenarios |
| `hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `server/aml_environment.py` | Core environment (15 tools + OS mechanics) |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario builder |
| `scenarios/adversary_agent.py` | LLM-backed evasive scenario generator |
| `adversarial_successes.db` | SQLite database storing failed scenarios for DPO |
| `graders/grader.py` | Dense reward engine (per-step + terminal) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async Queue, Kernel) |
| `models.py` | Pydantic type definitions |
| `tests/test_smoke.py` | Environment verification (7/7 tests) |
| `checkpoints/` | T4 training output (LoRA adapters + tokenizer) |
| `checkpoints_70b/` | 70B training output (DeepSpeed sharded checkpoints) |
| `demo_output/` | AGUI JSON payloads for frontend replay |
| `frontend/prisma/schema.prisma` | Prisma schema for DPO preference pairs |
| `frontend/app/api/preferences/` | Next.js API for capturing user corrections |

---

## Adversarial "GAN-Style" Training Loop

To generate challenging negative examples for the DPO pipeline, Memex includes an adversarial scenario generator that plays the role of a "money launderer" attempting to evade the Defender agent.

### Setup

```bash
# Run the battle orchestrator (generates scenarios and tests them against the Defender)
python train_adversary.py --adversary-model gpt-4o-mini --episodes 20 --difficulty hard
```

### Workflow

1. **Adversary Generation:** `adversary_agent.py` uses an LLM (or procedural fallback) to generate complex, evasive transaction graphs (e.g., mule rings, shell company pass-throughs, phantom invoices).
2. **Defender Evaluation:** The Defender runs the investigation against the generated scenario.
3. **Persistence:** If the Defender fails (score < threshold), the scenario is recorded in `adversarial_successes.db` (SQLite).
4. **Continuous Learning:** These evasive scenarios are then used as high-value negative examples in the DPO pipeline to harden the agent against new ML typologies.

---

## Continuous Learning (DPO Pipeline)

### Setup (Frontend)

```bash
cd frontend
npm install @prisma/client prisma
npx prisma init --datasource-provider sqlite
# Copy schema from prisma/schema.prisma (already provided)
npx prisma migrate dev --name init
```

### Workflow

1. **User corrects agent output** → POST to `/api/preferences` with `originalPrompt`, `rejectedResponse`, `chosenResponse`
2. **Batch DPO training** (run offline when enough pairs accumulate):
   ```bash
   python train_dpo.py --base-model checkpoints/best --db frontend/prisma/dev.db
   ```
3. **Hot-swap adapters** into running inference server:
   ```bash
   python hotswap.py --base unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --adapter checkpoints/dpo-latest
   ```

### DPO Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| `beta` | `0.1` | DPO temperature (lower = more aggressive preference) |
| `lr` | `1e-6` | Conservative to avoid catastrophic forgetting |
| `epochs` | `3` | Per-batch passes |
| `min_pairs` | `5` | Minimum pairs to trigger training |
