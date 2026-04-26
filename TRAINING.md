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
│     └── Deterministic seed per prompt for R3 replay            │
│                                                                 │
│  2. Model: Meta-Llama-3.1-8B-Instruct (Unsloth 4-bit + LoRA)  │
│     └── Generates G=4 completions per prompt                   │
│     └── Compute dtype: float16 (required by Unsloth 4-bit)     │
│                                                                 │
│  3. Multi-Step Reward (4 decomposed functions)                  │
│     └── parse_all_tool_calls() → extracts ALL tool calls       │
│     └── Each completion → multi-step env.step() → R_total      │
│                                                                 │
│  4. GRPO Update                                                 │
│     └── Advantage: A_i = (r_i - mean(r_group)) / std(r_group) │
│     └── Policy gradient with KL penalty (β=0.01)              │
│                                                                 │
│  5. Output: LoRA adapter checkpoint + training curves           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### Multi-Step Tool Call Parsing

The model generates completions containing **multiple tool calls** in a single response (e.g., 5-10 ```` ```json ```` blocks). The `parse_all_tool_calls()` function uses `re.finditer` to extract every valid tool call from the completion. This allows GRPO to score full investigation trajectories rather than single steps.

#### Deterministic Scenario Seeding

Each prompt is generated with a deterministic seed (`scenario_seed = i * 7919 + 42`). During R3 (environment execution), the same seed replays the exact same scenario, ensuring all G completions per prompt are evaluated against identical environment states. This eliminates variance noise in advantage estimation.

#### Float16 Compute Dtype

Unsloth's 4-bit quantization internally uses float16 as the BNB compute dtype. Using bfloat16 causes `RuntimeError: Half vs BFloat16` inside Unsloth's LoRA kernels. A100 handles fp16 natively with no performance penalty.

### Training on Colab / HF

**Target:** A100 (40/80 GB VRAM) — recommended for production runs
**Also works on:** L4 (24 GB), T4 (16 GB) with reduced batch sizes
**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
**Peak VRAM:** ~14 GB (4-bit + LoRA + G=4 generation)

### Colab Setup (Copy-Paste Cells)

```python
%%capture
# ═══════════════════════════════════════════════════════════
# CELL 1: Install Training Stack
# Runtime → GPU → A100 (Colab Pro) or L4
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
# CELL 5: ★ GRPO Training (~3-5 hours on A100)
# THIS IS THE PRIMARY TRAINING CELL
# ═══════════════════════════════════════════════════════════

import wandb
wandb.login()

!python train_grpo.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 250 \
    --num-generations 4 \
    --lr 5e-6 \
    --beta 0.04 \
    --epochs 2 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-completion-length 2048 \
    --wandb-project memex-grpo \
    --output-dir checkpoints/defender-grpo-v2
```

```python
# ═══════════════════════════════════════════════════════════
# CELL 5b (ALTERNATIVE): Run via HF Jobs CLI
# Uses pay-as-you-go HF compute ($0.80/hr for L4)
# ═══════════════════════════════════════════════════════════

# !pip install huggingface_hub[cli]
# !hf jobs uv run --flavor l4x1 python train_grpo.py \
#     --num-prompts 500 --num-generations 4 --wandb-project memex-grpo
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
| `--model` | `unsloth/Meta-Llama-3.1-8B-Instruct` | Base model |
| `--num-prompts` | `250` | Unique scenario prompts |
| `--num-generations` | `4` | G — group size for GRPO |
| `--lr` | `5e-6` | Learning rate |
| `--beta` | `0.04` | KL penalty coefficient |
| `--loss-type` | `grpo` | Loss variant: `grpo`, `dapo`, `dr_grpo` |
| `--lora-r` | `16` | LoRA rank |
| `--epochs` | `2` | Training epochs |
| `--batch-size` | `1` | Per-device train batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--max-completion-length` | `2048` | Max tokens per completion |
| `--wandb-project` | `memex-grpo` | WandB project name |
| `--output-dir` | `checkpoints/defender-grpo-v2` | Output directory |
| `--dry-run` | off | Quick test: 4 prompts, no WandB |

---

## Decomposed Reward Functions (Anti-Gaming Design)

We pass **4 independent reward functions** to `GRPOTrainer`. TRL sums them for the final reward per completion. This makes reward hacking much harder — gaming one signal doesn't help if the others penalize the degenerate behavior.

All reward functions except R1 use **multi-step scoring** via `parse_all_tool_calls()`, which extracts every ```` ```json ```` tool-call block from the model's completion. This means the agent is rewarded for its entire investigation trajectory, not just the first tool call.

```
R_total = R_format + R_investigation + R_execution + R_os_mechanics
```

### R1: Format Compliance — Prevents gibberish

| Output Type | Reward |
|------------|--------|
| Valid JSON with known tool name | `+1.0` |
| Valid JSON with unknown tool | `+0.3` |
| No valid JSON found | `-1.0` |
| Empty or degenerate (< 5 chars or >80% repeated tokens) | `-2.0` |

### R2: Investigation Quality — Prevents lazy tool choice

Multi-step: scores **all** tool calls in the completion for category diversity.

| Condition | Reward |
|----------|--------|
| Uses investigation tools (evidence gathering) | `+0.3` |
| Uses OS-mechanic tools (memory/async/kernel) | `+0.2` |
| Uses terminal tools (file_sar/close_alert) | `+0.1` |
| All tool calls have empty/dummy parameters | `-0.3` |
| No valid tool calls found | `0.0` |

*These stack: a completion using investigation + OS + terminal tools earns +0.6.*

### R3: Environment Execution — Ground-truth from AMLEnvironment

Each completion's **full tool-call sequence** is executed against a deterministically-seeded `AMLEnvironment` instance. The seed matches the one used during prompt generation, ensuring environment state consistency across all G completions in a group.

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
| No tool calls parsed | `-0.50` | Penalty for unparseable output |

### R4: OS Mechanics — Rewards innovative OS-agent features

Multi-step: scores **all unique** OS tool calls (deduplicated to prevent reward hacking).

| OS Tool | Reward | OS Concept |
|---------|--------|------------|
| `write_to_case_file` (with content) | `+0.3` | Virtual Memory (RAM→Disk paging) |
| `search_compliance_manual` (with query) | `+0.3` | Knowledge retrieval |
| `update_system_prompt` (with rule) | `+0.2` | Kernel-level meta-prompting |
| `request_wire_trace` | `+0.2` | Async job scheduling (Interrupts) |
| `retrieve_async_result` | `+0.1` | Interrupt handling |
| Empty write/injection | `-0.1` | Anti-gaming (prevents hollow calls) |
| Non-OS tool | `0.0` | Neutral |

---

## VRAM Budget (A100 = 40 GB)

| Component | VRAM |
|-----------|------|
| Base Llama 3.1 8B 4-bit (NF4) | ~5.5 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| KV cache (G=4 × 1024 seq) | ~2.0 GB |
| Optimizer (AdamW fp32) | ~1.2 GB |
| Activations (gradient checkpoint) | ~3-4 GB |
| **Total** | **~12 GB** |
| **Headroom (A100 40GB)** | **~28 GB ✓** |

---

## GRPO Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------| 
| `lr` | `5e-6` | Learning rate |
| `beta` | `0.04` | KL penalty weight against frozen base |
| `num_generations` | `4` | G — group size (more = better advantage estimation) |
| `loss_type` | `grpo` | Standard GRPO; `dapo` for token-level normalization |
| `max_completion_length` | `2048` | Max tokens per completion |
| `max_seq_length` | `4096` | Max total sequence length |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `warmup_steps` | `20` | LR warmup (fixed steps, not ratio) |
| `lr_scheduler_type` | `cosine` | Cosine decay schedule |
| `scale_rewards` | `True` | Normalize rewards across batch |
| `fp16` | `True` | Float16 training (required by Unsloth 4-bit) |
| `gradient_accumulation_steps` | `8` | 8 micro-batches per optimizer step |
| `num_train_epochs` | `2` | Training epochs over the prompt dataset |

---

## WandB Monitoring

**Dashboard:** [wandb.ai/n0s0ktesting-testing-labs/memex-grpo](https://wandb.ai/n0s0ktesting-testing-labs/memex-grpo)

| Metric | Healthy Range | What to Watch |
|--------|---------------|---------------|
| `reward/mean` | -1.5 → +0.8 over training | Main signal — should trend upward |
| `reward/std` | > 0.0 | **Critical**: if 0, GRPO has no gradient signal |
| `loss` | > 1e-3, then decreasing | Must be non-zero — convergence indicator |
| `grad_norm` | > 0.01 | Must be non-zero — parameters are updating |
| `kl` | 0.0 → 0.5, rising over time | Model diverging from base (expected) |
| `completions/mean_length` | 400–900 tokens | Short = lazy, Long = verbose |
| `rewards/reward_format_compliance/mean` | → 1.0 | Model learning ```` ```json ```` format |
| `rewards/reward_format_compliance/std` | → 0.0 | All completions achieving correct format |
| `rewards/reward_investigation_quality/mean` | → 0.5+ | Diverse tool usage |
| `rewards/reward_environment_execution/mean` | Increasing | Better environment interaction |
| `rewards/reward_os_mechanics/mean` | > 0.0 | Agent using OS tools |
| `rewards/reward_os_mechanics/std` | > 0.0 | Variance in OS tool adoption |
| `completions/clipped_ratio` | 0.0 (normal for LoRA) | LoRA updates are small — ratio stays within trust region |

### Understanding clip_ratio = 0

With LoRA (rank 16), each optimizer step changes token probabilities by < 0.01%. The GRPO clipping threshold is ε = 0.2 (20%). Since LoRA updates are orders of magnitude smaller than the clipping bound, `clip_ratio` will be 0.0 for most or all of training. **This is normal and expected** — it means updates are within the trust region without needing to be clipped.

The metrics that confirm learning are: `loss > 0`, `grad_norm > 0`, `kl` increasing, and `reward/mean` trending upward.

---

## Training Observations

### Reward Signal Validation

After the multi-step scoring overhaul, reward variance is confirmed healthy:

| Metric | Before Fix | After Fix |
|--------|:----------:|:---------:|
| `reward_std` | 0.00 (dead) | > 0.10 ✅ |
| `reward range` | flat 1.34 | -1.49 to +0.63 |
| `R4 (OS mechanics)` | always 0.00 | 0.00 – 0.50 ✅ |
| `format compliance` | always 1.00 | -1.00 to +1.00 ✅ |
| `grad_norm` | 0.001 (dead) | > 0.01 ✅ |

### What the Model Learns

| Behavior | Untrained | Trained |
|----------|:---------:|:-------:|
| Output format | Mix of inline JSON and ```` ```json ```` | Consistently uses ```` ```json ```` blocks |
| Investigation depth | 1-2 tool calls | 5-10 tool calls across categories |
| OS tool usage | Never used | `write_to_case_file`, `request_wire_trace` |
| Terminal decision | Always files SAR | Differentiates TP vs FP |

---

## Evaluation

After training, benchmark your checkpoint across all 6 typologies × 3 difficulties:

```bash
# Full evaluation (6 scenarios)
python eval_harness.py --checkpoint checkpoints/defender-grpo-v2

# Single scenario
python eval_harness.py --checkpoint checkpoints/defender-grpo-v2 --scenarios 1mdb_layering
```

---

## Alternative Approaches (Archived)

> The following pipelines are preserved in `archive/` for ablation comparison. **GRPO is the recommended training path.**

### Self-Play PPO

```python
# Self-play dry-run (legacy PPO)
# python self_play.py --dry-run
# python archive/train_defender_ppo.py --dry-run --scenario-source procedural
# python archive/train_launderer_ppo.py --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--outer-rounds` | `3` | Number of L→D alternating rounds |
| `--defender-warmup` | `20` | Phase 1 iterations (procedural only) |
| `--launderer-iters` | `10` | Launderer PPO iterations per round |
| `--defender-iters` | `15` | Defender mixed-mode iterations per round |
| `--wandb-project` | `memex-selfplay` | WandB project name |
| `--dry-run` | off | 2 iters × 1 ep per phase |

### DPO Continuous Learning

```bash
# Batch DPO training (offline, from user corrections)
python archive/train_dpo.py --base-model checkpoints/defender-grpo-v2 --db frontend/prisma/dev.db

# Hot-swap adapters into running server
python archive/hotswap.py --base unsloth/Meta-Llama-3.1-8B-Instruct --adapter checkpoints/dpo-latest
```

### Archived File Reference

| File | Purpose |
|------|---------|
| `archive/train_defender_ppo.py` | Defender PPO with GAE, EMA baseline, batch normalization |
| `archive/train_launderer_ppo.py` | Launderer single-step PPO (generates evasive scenarios) |
| `archive/train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| `archive/hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |
