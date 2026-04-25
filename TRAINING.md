# Memex Training Guide

> Complete training pipeline for the Memex OS-Agent Benchmark.
> **GRPO** (TRL + Unsloth) is the **primary training path** — the Defender agent
> learns to investigate AML alerts using Group Relative Policy Optimization.
> Self-play (Launderer vs Defender) provides adversarial curriculum generation.

---

## Prerequisites

| Dependency | Purpose |
|-----------|---------|
| `unsloth` | 4-bit NF4 quantization + fast LoRA (2× faster, 70% less VRAM) |
| `trl` | GRPOTrainer — Group Relative Policy Optimization |
| `peft` | LoRA adapter management |
| `bitsandbytes` | Quantization backend |
| `wandb` | Experiment tracking |
| `datasets` | HuggingFace prompt dataset |
| `pydantic>=2.0` | Environment type contracts |
| `openenv-core` | OpenEnv environment SDK |

---

## ★ GRPO Training (Primary — TRL + Unsloth)

The Defender agent is trained using [GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300), which eliminates the need for a critic network by comparing G completions per prompt. This follows the exact OpenEnv + TRL pattern recommended by the hackathon organizers.

### Why GRPO?

| Feature | PPO (Legacy) | GRPO (Current) |
|---------|-------------|----------------|
| Value function | Needs critic V(s) — approximated with EMA baseline | **No critic** — advantage from group comparison |
| Advantage | GAE with EMA baseline | `A_i = (r_i - mean(r_group)) / std(r_group)` |
| Generation | 1 completion per prompt | **G completions per prompt** (e.g., G=4) |
| Framework | Custom from-scratch (2000+ lines) | **TRL `GRPOTrainer`** (~400 lines) |
| Memory | Manual model swapping | Unsloth handles 4-bit + gradient checkpointing |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO TRAINING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Prompt Dataset (from procedural scenario engine)            │
│     └── AMLEnvironment.reset() → initial alert observations    │
│                                                                 │
│  2. Model: Meta-Llama-3.1-8B-Instruct (Unsloth 4-bit + LoRA)  │
│     └── Generates G=4 completions per prompt                   │
│                                                                 │
│  3. Reward Function (OpenEnv pattern)                           │
│     └── Each completion → parsed tool call → env.step() → r_i  │
│                                                                 │
│  4. GRPO Update                                                 │
│     └── Advantage: A_i = (r_i - mean(r_group)) / std(r_group) │
│     └── Policy gradient with KL penalty (β=0.04)              │
│                                                                 │
│  5. Output: LoRA adapter checkpoint + training curves           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training on Colab / HF

**Target:** Colab Pro L4 (24 GB VRAM) or HF Jobs (L4/T4)
**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Peak VRAM:** ~12-14 GB (4-bit + LoRA + G=4 generation)

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

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Unsloth — 4-bit quantized model loading + LoRA + Triton attention
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# RL + adapter stack
!pip install --no-deps trl peft accelerate bitsandbytes

# Dataset + tracking + project deps
!pip install datasets wandb pydantic>=2.0.0 matplotlib

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

# Optional: symlink checkpoints to Google Drive for persistence
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/memex_checkpoints
# !ln -s /content/drive/MyDrive/memex_checkpoints checkpoints_drive
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
# CELL 4: Dry-run (4 prompts, 1 epoch, no WandB)
# ═══════════════════════════════════════════════════════════

!python train_grpo.py --dry-run
# Verifies: model loading, prompt generation, reward function, GRPO update
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5: ★ GRPO Training (~3-5 hours on L4)
# THIS IS THE PRIMARY TRAINING CELL
# ═══════════════════════════════════════════════════════════

import wandb
wandb.login()

!python train_grpo.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --num-prompts 100 \
    --num-generations 4 \
    --lr 5e-6 \
    --beta 0.04 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 4 \
    --wandb-project memex-grpo \
    --output-dir checkpoints/defender-grpo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5b (ALTERNATIVE): Run via HF Jobs CLI
# Uses pay-as-you-go HF compute ($0.80/hr for L4)
# ═══════════════════════════════════════════════════════════

# !pip install huggingface_hub[cli]
# !hf jobs uv run --flavor l4x1 python train_grpo.py \
#     --num-prompts 100 --num-generations 4 --wandb-project memex-grpo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 6: Evaluate best checkpoint (9 combos)
# ═══════════════════════════════════════════════════════════

!python eval_harness.py --checkpoint checkpoints/defender-grpo
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 7: Run 1MDB demo + download AGUI replay
# ═══════════════════════════════════════════════════════════

# Scripted (deterministic, no GPU)
!python demo_eval.py --dry-run

# With trained model
# !python demo_eval.py --model checkpoints/defender-grpo

# Download replay files for Next.js frontend
# from google.colab import files
# !zip -r demo_output.zip demo_output/
# files.download('demo_output.zip')
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 8: Save checkpoints to Google Drive
# ═══════════════════════════════════════════════════════════

import shutil, os

src = "/content/Meta-Pytorch-Hackathon/checkpoints"
dst = "/content/drive/MyDrive/memex_checkpoints"

shutil.copytree(src, dst, dirs_exist_ok=True)
print("✅ Done! Find it in your Drive → memex_checkpoints/")
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 9: Push trained model to HuggingFace Hub
# ═══════════════════════════════════════════════════════════

from huggingface_hub import HfApi
api = HfApi()

# Push the LoRA adapter
api.upload_folder(
    folder_path="checkpoints/defender-grpo",
    repo_id="MuazTPM/memex-defender-grpo",
    repo_type="model",
    commit_message="Defender GRPO checkpoint (Unsloth + TRL)"
)
print("✅ Model pushed to HuggingFace Hub!")
```

**GRPO CLI Reference:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Base model |
| `--num-prompts` | `100` | Unique scenario prompts |
| `--num-generations` | `4` | G — group size for GRPO |
| `--lr` | `5e-6` | Learning rate |
| `--beta` | `0.04` | KL penalty coefficient |
| `--loss-type` | `grpo` | Loss variant: `grpo`, `dapo`, `dr_grpo` |
| `--lora-r` | `16` | LoRA rank |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `1` | Per-device train batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--max-completion-length` | `2048` | Max tokens per completion |
| `--wandb-project` | `memex-grpo` | WandB project name |
| `--output-dir` | `checkpoints/defender-grpo` | Output directory |
| `--dry-run` | off | Quick test: 4 prompts, no WandB |

---

## Decomposed Reward Functions (Anti-Gaming Design)

We pass **4 independent reward functions** to `GRPOTrainer`. TRL sums them for the final reward per completion. This makes reward hacking much harder — gaming one signal doesn't help if the others penalize the degenerate behavior.

```
R_total = R_format + R_investigation + R_execution + R_os_mechanics
```

### R1: Format Compliance — Prevents gibberish

| Output Type | Reward |
|------------|--------|
| Valid JSON with known tool name | `+0.2` |
| Valid JSON with unknown tool | `+0.1` |
| No valid JSON found | `-0.5` |
| Empty or degenerate (>80% repeated tokens) | `-1.0` |

### R2: Investigation Quality — Prevents lazy tool choice

| Tool Category | Reward |
|--------------|--------|
| Investigation tool (evidence gathering) | `+0.3` |
| OS-mechanic tool (memory/async/kernel) | `+0.2` |
| Terminal tool (file_sar/close_alert) | `+0.1` |
| Empty/dummy parameters | `-0.3` |

### R3: Environment Execution — Ground-truth from AMLEnvironment

Each completion is executed against a fresh `AMLEnvironment` instance via the OpenEnv pattern. This is the HARDEST reward to game because it requires correct tool interaction.

| Signal | Value | Source |
|--------|-------|--------|
| Action cost | `-0.02` | Every tool call |
| Redundant tool | `-0.03` | Duplicate call hash |
| Page fault | `-0.05` | Accessed evicted RAM data |
| Async timeout | `-0.10` | Premature async retrieval |
| Successful disk write | `+0.10` | Good write_to_case_file |
| Kernel update | `+0.15` | Useful compliance rule injection |
| Investigation bonus | `+0.02–0.05` | First use of each tool type |
| **Terminal: TP** | `+1.00` | Correct SAR on suspicious |
| **Terminal: TN** | `+0.50` | Correct close on clean |
| **Terminal: FP** | `-0.75` | False SAR on clean |
| **Terminal: FN** | `-2.00` | Missed money laundering |

### R4: OS Mechanics — Rewards innovative OS-agent features

| OS Tool | Reward | OS Concept |
|---------|--------|------------|
| `write_to_case_file` (with content) | `+0.3` | Virtual Memory (RAM→Disk paging) |
| `search_compliance_manual` | `+0.3` | Knowledge retrieval |
| `update_system_prompt` (with rule) | `+0.2` | Kernel-level meta-prompting |
| `request_wire_trace` | `+0.2` | Async job scheduling (Interrupts) |
| `retrieve_async_result` | `+0.1` | Interrupt handling |
| Empty write/injection | `-0.1` | Anti-gaming (prevents hollow calls) |
| Non-OS tool | `0.0` | Neutral |

---

## VRAM Budget (L4 = 24 GB)

| Component | VRAM |
|-----------|------|
| Base Llama 3.1 8B 4-bit (NF4) | ~5.5 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| KV cache (G=4 × 2048 seq) | ~4.0 GB |
| Optimizer (AdamW fp32) | ~1.2 GB |
| Activations (gradient checkpoint) | ~3-4 GB |
| **Total** | **~14 GB** |
| **Headroom** | **~10 GB ✓** |

---

## GRPO Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lr` | `5e-6` | Learning rate |
| `beta` | `0.04` | KL penalty weight against frozen base |
| `num_generations` | `4` | G — group size (more = better advantage estimation) |
| `loss_type` | `grpo` | Standard GRPO; `dapo` for token-level normalization |
| `max_completion_length` | `2048` | Max tokens per completion |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `warmup_ratio` | `0.05` | LR warmup |
| `lr_scheduler_type` | `cosine` | Cosine decay schedule |
| `scale_rewards` | `True` | Normalize rewards across batch |
| `bf16` | `True` | Mixed-precision training |

---

## WandB Monitoring

| Metric | Healthy Range | What to Watch |
|--------|---------------|---------------|
| `reward/mean` | 0.0 → +0.8 over training | Main signal — should increase |
| `loss` | Decreasing, then stable | Convergence indicator |
| `kl` | 0.0 to 0.5 | Values > 1.0 = policy drifting too far from base |
| `completion_length/mean` | 50–500 tokens | Short = lazy, Long = verbose |
| `reward/std` | Decreasing | Group advantage becoming more discriminating |

---

## Legacy: Self-Play PPO (Archived)

> The PPO-based self-play pipeline is preserved for ablation comparison. GRPO is the recommended training path.

```python
# Self-play dry-run (legacy PPO — for reference only)
# !python self_play.py --dry-run
# !python train_defender_ppo.py --dry-run --scenario-source procedural
# !python train_launderer_ppo.py --dry-run
```

**Self-Play CLI Reference (PPO):**

| Flag | Default | Description |
|------|---------|-------------|
| `--outer-rounds` | `3` | Number of L→D alternating rounds |
| `--defender-warmup` | `20` | Phase 1 iterations (procedural only) |
| `--launderer-iters` | `10` | Launderer PPO iterations per round |
| `--defender-iters` | `15` | Defender mixed-mode iterations per round |
| `--wandb-project` | `memex-selfplay` | WandB project name |
| `--dry-run` | off | 2 iters × 1 ep per phase |

---

## Hardware Reference

| Config | GPU | VRAM | Script | Est. Time |
|--------|-----|------|--------|-----------|
| HF Jobs | T4 | 16 GB | `train_grpo.py` | ~5-7 hrs |
| Colab Pro | L4 | 24 GB | `train_grpo.py` | ~3-5 hrs |
| HF Jobs | L4 | 24 GB | `train_grpo.py` | ~3-5 hrs |
| Colab Pro | A100 | 40 GB | `train_grpo.py` | ~1.5-2 hrs |

---

## File Reference

| File | Purpose |
|------|---------|
| **GRPO Training (Primary)** | |
| `train_grpo.py` | **★ Defender GRPO training** (TRL GRPOTrainer + Unsloth) |
| **Self-Play Pipeline (Legacy)** | |
| `self_play.py` | Alternating best-response orchestrator (PPO-based) |
| `train_defender_ppo.py` | Defender PPO with GAE, EMA baseline, batch normalization |
| `train_launderer_ppo.py` | Launderer single-step PPO (generates evasive scenarios) |
| `server/launderer_env.py` | One-step MDP for Launderer (validates JSON, runs frozen Defender) |
| **Standalone Training (Legacy)** | |
| `train_ppo.py` | Step-level PPO (Unsloth 4-bit + LoRA, L4-optimized) |
| `train_ppo_70b.py` | Multi-GPU PPO (DeepSpeed ZeRO-3, A100 cluster) |
| `train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| **Infrastructure** | |
| `hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `eval_harness.py` | Checkpoint benchmarking across typology/difficulty grid |
| `curriculum/plr_engine.py` | PLR buffer: regret-weighted scenario sampling |
| `curriculum/oracle.py` | Proxy regret oracle (`1.0 - protagonist_score`) |
| `server/aml_environment.py` | Core environment (18 tools + OS mechanics) |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario builder |
| `graders/grader.py` | Dense reward engine (per-step + terminal + investigation bonuses) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async Queue, Kernel) |
| `models.py` | Pydantic type definitions |
| `tests/test_smoke.py` | Environment verification (8/8 tests) |

---

## Continuous Learning (DPO Pipeline)

### Setup (Frontend)

```bash
cd frontend
npm install @prisma/client prisma
npx prisma init --datasource-provider sqlite
npx prisma migrate dev --name init
```

### Workflow

1. **User corrects agent output** → POST to `/api/preferences` with `originalPrompt`, `rejectedResponse`, `chosenResponse`
2. **Batch DPO training** (run offline when enough pairs accumulate):
   ```bash
   python train_dpo.py --base-model checkpoints/defender-grpo --db frontend/prisma/dev.db
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

---

## Research Context

Our training pipeline aligns with several 2025-2026 RL research directions:

| Technique | Paper/Method | How Memex Uses It |
|-----------|-------------|-------------------|
| **GRPO** | DeepSeekMath (Shao et al., 2024) | Group-relative advantage estimation — no critic, G=4 completions per prompt |
| **Turn-level dense rewards** | TIPS (Xie et al., ICLR 2026) | `grade_step()` provides per-tool-call shaping for OS mechanics |
| **Adaptive environment generation** | EnvGen (2025) | PLR engine + Launderer self-play dynamically adjust scenario difficulty |
| **Anti-gaming reward design** | Incentive audit best practices | Hard caps, closed action sets, formal lazy-policy analysis |
| **Potential-based shaping** | Ng et al. (1999) | Per-step OS rewards are potential-based: they reward state improvement without altering the optimal terminal policy |
| **Unsloth quantization** | 4-bit NF4 + LoRA | 2× faster training, 70% less VRAM |

---

## Future Work

- **Multi-step GRPO via `environment_factory`**: Use TRL's built-in multi-turn agent support to run full investigation episodes (15-25 tool calls) as GRPO completions
- **Launderer GRPO**: Train the scenario generator with GRPO — single-step JSON generation scored against a frozen Defender
- **GRPO self-play orchestrator**: Alternate Defender/Launderer GRPO rounds for adversarial curriculum
- **Leave-one-out baseline (LOOP)**: K=2 leave-one-out for lower-variance advantage estimation
- **TIPS-style potential computation**: Use a frozen policy's log-likelihood as per-step potential, replacing flat bonuses with information-theoretic shaping signals
