# Memex PPO Training Guide

> Complete training pipeline for the Memex OS-Agent Benchmark.
> **PPO is the production training path.** Self-play (Launderer vs Defender) is the primary adversarial loop.
> GRPO is experimental/ablation only. DPO handles offline refinement from human corrections.

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

## Tier 1: L4 Training (8B Model)

**Target:** Colab Pro L4 (24 GB VRAM)
**Script:** `train_ppo.py` (standalone) / `train_defender_ppo.py` + `train_launderer_ppo.py` (self-play)
**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Peak VRAM:** ~10 GB (PPO) / ~12 GB (self-play with model swap)

### Colab Setup (Copy-Paste Cells)

```python
%%capture
# ═══════════════════════════════════════════════════════════
# CELL 1: Install Training Stack
# Runtime → GPU → L4 (Colab Pro) or T4 (free tier)
# ═══════════════════════════════════════════════════════════
#
# ⚠️ DO NOT install flash-attn separately!
# Unsloth uses its own custom Triton attention kernels that are
# faster than FlashAttention-2, and falls back to PyTorch SDPA
# automatically. Installing flash-attn compiles from source
# (~20-45 min on L4) and wastes Compute Units for zero benefit.
# None of our training scripts import or reference flash-attn.

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Unsloth — 4-bit quantized model loading + LoRA + Triton attention
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

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
# Expected: 8/8 tests passed ✓
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 4: Dry-run (2 iterations, 1 episode, no WandB)
# ═══════════════════════════════════════════════════════════

!python train_ppo.py --dry-run
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5: Full PPO training with PLR (~2 hours on L4)
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
    --use-plr \
    --wandb-project memex-ppo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5c: Self-Play training (Launderer vs Defender)
# Alternating best-response: Warmup → Launderer → Defender
# Only one model loaded at a time (L4-safe)
# ═══════════════════════════════════════════════════════════

!python self_play.py \
    --outer-rounds 3 \
    --defender-warmup 20 \
    --launderer-iters 10 \
    --defender-iters 15 \
    --wandb-project memex-selfplay

# Or individual agent training:
# Defender only (procedural scenarios)
# !python train_defender_ppo.py --dry-run --scenario-source procedural

# Launderer only
# !python train_launderer_ppo.py --dry-run
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5b: GRPO training (EXPERIMENTAL — ablation only)
# ⚠️ GRPO lacks importance-ratio clipping, uses |KL| instead
# of forward KL, and has gradient-magnitude coupling to group
# size. Use train_ppo.py for production. Requires --experimental.
# ═══════════════════════════════════════════════════════════

!python train_grpo.py --experimental \
    --lr 2e-4 \
    --lora-r 16 \
    --group-size 4 \
    --episodes 4 \
    --iterations 150 \
    --use-plr \
    --wandb-project memex-grpo
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

| Parameter | L4 (PPO) | L4 (GRPO) | A100 (70B) | Purpose |
|-----------|----------|-----------|------------|---------|
| `lr` | `5e-6` | `2e-4` | `2e-6` | Higher LR for GRPO (critic-free); lower for 70B |
| `kl_coef` | `0.05` | `0.05` | `0.03` | KL penalty weight against frozen base |
| `entropy_coef` | `0.05` | `0.05` | `0.05` | Exploration bonus |
| `clip_eps` | `0.2` | N/A (raw REINFORCE) | `0.2` | Standard PPO clipping (GRPO lacks this) |
| `reward_clip` | `2.0` | `2.0` | `2.0` | Return clipping bound |
| `grad_accum_steps` | `4` | `4` | `8` | Effective batch size |
| `max_grad_norm` | `1.0` | `1.0` | `0.5` | Tighter gradient clipping for 70B |

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
| Colab Pro | L4 | 24 GB | `train_ppo.py` / `self_play.py` | 8B 4-bit | ~2 hrs |
| Colab Pro | A100-40GB | 40 GB | `train_ppo.py` | 8B 4-bit | ~50 min |
| On-site | 4× A100-80GB | 320 GB | `train_ppo_70b.py` | 70B 4-bit | ~4 hrs |
| On-site | 8× A100-80GB | 640 GB | `train_ppo_70b.py` | 70B 4-bit | ~2 hrs |

---

## File Reference

| File | Purpose |
|------|---------|
| **Self-Play Pipeline** | |
| `self_play.py` | **Alternating best-response orchestrator** (Warmup → Launderer → Defender × N rounds) |
| `train_defender_ppo.py` | Defender PPO with GAE (λ=0.95), mixed scenarios, entity-F1/typology tracking |
| `train_launderer_ppo.py` | Launderer single-step PPO (generates evasive scenarios to fool Defender) |
| `server/launderer_env.py` | One-step MDP for Launderer (validates JSON, runs frozen Defender, computes reward) |
| **Standalone Training** | |
| `train_ppo.py` | Step-level PPO (Unsloth 4-bit + LoRA, L4-optimized, `--use-plr`) |
| `train_grpo.py` | Group Relative Policy Optimization (**EXPERIMENTAL** — no clipping, `\|KL\|`, gradient coupling) |
| `train_ppo_70b.py` | Multi-GPU PPO (DeepSpeed ZeRO-3, A100 cluster, proof of scalability) |
| `train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| `train_adversary.py` | Adversarial scenario generator & heuristic filter (**DEPRECATED** — use `self_play.py`) |
| **Infrastructure** | |
| `hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `curriculum/plr_engine.py` | PLR buffer: regret-weighted scenario sampling |
| `curriculum/oracle.py` | Proxy regret oracle (`1.0 - protagonist_score`) |
| `server/aml_environment.py` | Core environment (18 tools + OS mechanics) |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario builder (emits `is_suspicious` ground truth) |
| `scenarios/adversary_agent.py` | Local Llama-3.1-8B evasive scenario generator |
| `graders/grader.py` | Dense reward engine (per-step + terminal) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async Queue, Kernel with finite mode set) |
| `models.py` | Pydantic type definitions (incl. `TypologyEnum`, `CurriculumState`) |
| `tests/test_smoke.py` | Environment verification (8/8 tests) |
| `tests/test_plr.py` | PLR engine unit tests |
| `checkpoints/` | Training output (LoRA adapters + tokenizer + PLR buffer) |
| `checkpoints_70b/` | 70B training output (DeepSpeed sharded checkpoints) |
| `demo_output/` | AGUI JSON payloads for frontend replay |
| `frontend/components/case/CurriculumPanel.tsx` | 5th AGUI panel (PLR curriculum visualization) |
| `frontend/prisma/schema.prisma` | Prisma schema for DPO preference pairs |
| `frontend/app/api/preferences/` | Next.js API for capturing user corrections |

---

## Adversarial Scenario Generation

### Self-Play (Production Path)

The **primary adversarial approach** is the two-agent self-play loop in `self_play.py`, where a Launderer-8B generates evasive scenarios and a Defender-8B learns to investigate them. See Cell 5c above.

### Legacy Heuristic Filter (Deprecated)

The legacy `train_adversary.py` generates scenarios and filters them through a static heuristic function. This is an **offline data curation tool**, not true self-play.

> **⚠️ Deprecated:** Use `self_play.py` for true adversarial training. The heuristic filter remains available for offline DPO data generation.

### Setup

```bash
# Run the battle orchestrator (uses local Llama-3.1-8B — no API key needed)
python train_adversary.py --episodes 20 --difficulty hard

# Or with procedural fallback (no GPU needed)
MEMEX_BACKEND=procedural python train_adversary.py --episodes 20 --difficulty hard
```

### Workflow

1. **Adversary Generation:** `adversary_agent.py` uses a local Llama-3.1-8B model (or procedural fallback) to generate complex, evasive transaction graphs (e.g., mule rings, shell company pass-throughs, phantom invoices).
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
