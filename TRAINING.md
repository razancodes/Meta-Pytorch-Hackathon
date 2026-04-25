# Memex Training Guide

> Complete training pipeline for the Memex OS-Agent Benchmark.
> **Self-play** (Launderer vs Defender) is the **production training path**.
> Standalone PPO (`train_ppo.py`) is available for ablation.
> DPO handles offline refinement from human corrections.

---

## Prerequisites

| Dependency | Purpose |
|-----------|---------|
| `unsloth` | 4-bit NF4 quantization + fast LoRA |
| `peft` | LoRA adapter management |
| `bitsandbytes` | Quantization backend |
| `wandb` | Experiment tracking |
| `pydantic>=2.0` | Environment type contracts |

---

## Training on Colab (L4 / T4)

**Target:** Colab Pro L4 (24 GB VRAM) or free-tier T4 (15 GB)
**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Peak VRAM:** ~10 GB (standalone PPO) / ~12 GB (self-play with model swap)

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
# CELL 4: Dry-run (2 iters, 1 episode, no WandB)
# ═══════════════════════════════════════════════════════════

# Defender PPO dry-run
!python train_defender_ppo.py --dry-run --scenario-source procedural

# Launderer PPO dry-run
!python train_launderer_ppo.py --dry-run

# Self-play dry-run (all three phases, tiny iters)
!python self_play.py --dry-run
```

> **⚠️ CLI Note:** `self_play.py --dry-run` automatically uses 2 iterations / 1 episode per phase. There is no `--iterations-per-phase` flag — phase iteration counts are controlled by `--defender-warmup`, `--launderer-iters`, and `--defender-iters` (overridden to small values in dry-run mode).

```python
# ═══════════════════════════════════════════════════════════
# CELL 5: ★ Self-Play training (~6-8 hours on L4)
# THIS IS THE PRIMARY TRAINING CELL
# Alternating best-response: Warmup → Launderer → Defender
# Only one model loaded at a time (VRAM-safe)
# ═══════════════════════════════════════════════════════════

import wandb
wandb.login()

!python self_play.py \
    --outer-rounds 3 \
    --defender-warmup 20 \
    --launderer-iters 10 \
    --defender-iters 15 \
    --wandb-project memex-selfplay

# Or individual agent training:
# !python train_defender_ppo.py --scenario-source procedural --iterations 50
# !python train_launderer_ppo.py --iterations 50 --defender-checkpoint checkpoints/defender/best
```

**Self-Play CLI Reference:**

| Flag | Default | Description |
|------|---------|-------------|
| `--outer-rounds` | `3` | Number of L→D alternating rounds |
| `--defender-warmup` | `20` | Phase 1 iterations (procedural only) |
| `--launderer-iters` | `10` | Launderer PPO iterations per round |
| `--defender-iters` | `15` | Defender mixed-mode iterations per round |
| `--defender-episodes` | `4` | Episodes per Defender iteration |
| `--launderer-episodes` | `4` | Episodes per Launderer iteration |
| `--mix-start` | `0.3` | Initial Launderer scenario fraction |
| `--mix-max` | `0.7` | Final Launderer scenario fraction |
| `--mix-schedule` | `linear` | Mix ratio schedule (`linear` or `fixed`) |
| `--wandb-project` | `memex-selfplay` | WandB project name |
| `--dry-run` | off | 2 iters × 1 ep per phase |

```python
# ═══════════════════════════════════════════════════════════
# CELL 5b: Standalone PPO (ALTERNATIVE — no adversarial)
# Single-agent PPO with PLR curriculum. Use self_play.py
# for the full two-agent adversarial pipeline instead.
# ═══════════════════════════════════════════════════════════

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

---

## VRAM Budget (L4 = 24 GB)

| Component | VRAM |
|-----------|------|
| Base 8B 4-bit (NF4) | ~5.5 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| KV cache (2048 seq) | ~2.0 GB |
| Optimizer (AdamW fp32) | ~1.2 GB |
| Activations (gradient checkpoint) | ~3-6 GB |
| **Total** | **~12-15 GB** |
| **Headroom** | **~9 GB ✓** |

> Only one model is loaded at a time during self-play. The Launderer and Defender never coexist in VRAM.

---

## PPO Stability Engineering

Both trainers include **14 production-grade safety features** to prevent policy collapse during long-horizon RL training:

### Mathematical Fixes

| Feature | What | Why |
|---------|------|-----|
| **Mean log-prob** | `token_lp.mean()` instead of `.sum()` | KL divergence is now scale-invariant — same magnitude regardless of whether responses are 40 or 200 tokens |
| **Ratio clamping** | `clamp(log_ratio, -10, 10)` before `exp()` | Prevents inf/NaN from policy drift between PPO epochs |
| **Entropy bonus** | `- entropy_coef × H(π)` in loss | Keeps the policy exploring; prevents collapsing to a single degenerate action |
| **Return clipping** | `clip(returns, -2.0, +2.0)` | Bounds gradient signals from outlier terminal rewards |
| **Directional KL** | `kl.clamp(min=0)` not `kl.abs()` | Only penalizes divergence FROM reference; does not penalize convergence TOWARD reference |
| **Empty response guard** | Dummy EOS if model generates 0 tokens | Prevents NaN from `.mean()` on an empty tensor |
| **Degenerate response detection** | If >80% repeated tokens, assign -0.15 penalty | Detects and penalizes gibberish output |
| **Fault-tolerant env.step()** | try/except around environment step, -0.10 penalty | Malformed actions no longer crash the training loop |
| **Type-safe parse_action()** | Force `params` to dict, catch TypeError/ValueError | Prevents `'str' object has no attribute 'get'` crashes |
| **KL early stopping** | Break PPO epochs if \|KL\| > 15 | Prevents catastrophic gradient updates when policy drifts too far |
| **Terminal reward de-duplication** | Subtract prior step rewards from terminal composite | The grader's `grade()` includes accumulated micro-rewards in the terminal total. Without subtraction, GAE would double-count every per-step reward — once in the trajectory and once in the terminal signal. |
| **🔄 Auto-Revert** | Entropy heartbeat monitor + checkpoint reload | Detects mode collapse and automatically reverts to last stable weights |
| **🆕 Cross-episode batch normalization** | Normalize advantages across all episodes in a batch, not per episode | Per-episode normalization destroys inter-episode ranking: a great episode's worst step gets negative advantage. Batch normalization preserves global credit assignment. |
| **🆕 EMA reward baseline** | Exponential moving average of episode returns as constant V(s) approximation | Reduces advantage variance without a critic network. α=0.1 tracks the expected return over time, centering GAE deltas around it. |

### Investigation Progress Bonuses

The Defender receives small, first-use-only positive bonuses for using core investigation tools. These create a discoverable reward gradient toward the terminal action, solving the cold-start problem where the model can't find positive rewards through random exploration:

| Tool | Bonus | Purpose |
|------|-------|---------|
| `review_alert` | +0.03 | Correct first step |
| `get_customer_profile` | +0.02 | KYC gathering |
| `query_transactions` | +0.02 | Core investigation |
| `check_watchlist` | +0.02 | Sanctions screening |
| `trace_network` | +0.02 | Network analysis |
| `check_source_of_funds` | +0.02 | Due diligence |
| `write_to_case_file` | +0.03 | Memory management (OS mechanic) |
| `file_sar` / `close_alert` | +0.05 | Terminal decision |

**Total possible bonus: ~+0.26** (won't dominate terminal reward of ±1.0). Each tool type awards its bonus only once per episode.

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

Stable checkpoints are saved only when `entropy > 0.05 AND mean_score > 0.3`. Maximum 5 reverts per run.

---

## Research Context

Our training pipeline aligns with several 2025-2026 RL research directions:

| Technique | Paper/Method | How Memex Uses It |
|-----------|-------------|-------------------|
| **Turn-level dense rewards** | TIPS (Xie et al., ICLR 2026) | `grade_step()` provides per-tool-call shaping for OS mechanics |
| **Value-free advantage estimation** | LOOP (2025), GRPO | V(s)≈EMA approximation with batch normalization; future work: K=2 leave-one-out baseline |
| **Adaptive environment generation** | EnvGen (2025) | PLR engine + Launderer self-play dynamically adjust scenario difficulty |
| **Anti-gaming reward design** | Incentive audit best practices | Hard caps, closed action sets, formal lazy-policy analysis |
| **Potential-based shaping** | Ng et al. (1999) | Per-step OS rewards are potential-based: they reward state improvement without altering the optimal terminal policy |

### Hyperparameters

| Parameter | PPO (L4) | Purpose |
|-----------|----------|---------|
| `lr` | `5e-6` | Learning rate |
| `kl_coef` | `0.05` | KL penalty weight against frozen base |
| `entropy_coef` | `0.05` | Exploration bonus |
| `clip_eps` | `0.2` | Standard PPO clipping |
| `reward_clip` | `2.0` | Return clipping bound |
| `grad_accum_steps` | `4` | Effective batch size |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `gae_lambda` | `0.95` | GAE bias-variance tradeoff (Defender only) |
| `gamma` | `0.99` | Discount factor |

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
| Colab Pro | L4 | 24 GB | `self_play.py` | 8B 4-bit | ~6-8 hrs |
| Colab Pro | A100-40GB | 40 GB | `self_play.py` | 8B 4-bit | ~3 hrs |

---

## File Reference

| File | Purpose |
|------|---------|
| **Self-Play Pipeline** | |
| `self_play.py` | **Alternating best-response orchestrator** (Warmup → Launderer → Defender × N rounds) |
| `train_defender_ppo.py` | Defender PPO with GAE (λ=0.95), EMA baseline, batch normalization, mixed scenarios, entity-F1/typology tracking |
| `train_launderer_ppo.py` | Launderer single-step PPO (generates evasive scenarios to fool Defender) |
| `server/launderer_env.py` | One-step MDP for Launderer (validates JSON, runs frozen Defender, computes reward) |
| **Standalone Training** | |
| `train_ppo.py` | Step-level PPO (Unsloth 4-bit + LoRA, L4-optimized, `--use-plr`) |
| `train_ppo_70b.py` | Multi-GPU PPO (DeepSpeed ZeRO-3, A100 cluster, proof of scalability) |
| `train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| **Infrastructure** | |
| `hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |
| `demo_eval.py` | 1MDB demo with AGUI replay capture |
| `eval_harness.py` | Checkpoint benchmarking across typology/difficulty grid |
| `curriculum/plr_engine.py` | PLR buffer: regret-weighted scenario sampling |
| `curriculum/oracle.py` | Proxy regret oracle (`1.0 - protagonist_score`) |
| `server/aml_environment.py` | Core environment (18 tools + OS mechanics) |
| `scenarios/procedural_generator.py` | Procedural POMDP scenario builder (emits `is_suspicious` ground truth) |
| `scenarios/adversary_agent.py` | Local Llama-3.1-8B evasive scenario generator |
| `graders/grader.py` | Dense reward engine (per-step + terminal + investigation bonuses) |
| `state_manager.py` | OS mechanics (RAM, Disk, Async Queue, Kernel with finite mode set) |
| `models.py` | Pydantic type definitions (incl. `TypologyEnum`, `CurriculumState`) |
| `tests/test_smoke.py` | Environment verification (8/8 tests) |
| `tests/test_plr.py` | PLR engine unit tests |
| `checkpoints/` | Training output (LoRA adapters + tokenizer + PLR buffer) |
| `demo_output/` | AGUI JSON payloads for frontend replay |
| `frontend/components/case/CurriculumPanel.tsx` | 5th AGUI panel (PLR curriculum visualization) |
| `frontend/prisma/schema.prisma` | Prisma schema for DPO preference pairs |
| `frontend/app/api/preferences/` | Next.js API for capturing user corrections |

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

---

## Future Work

- **Adaptive step budget**: Scale `max_steps` with difficulty (easy=12, medium=18, hard=25) to create tighter resource constraints and force more efficient policies
- **Leave-one-out baseline (LOOP)**: Run K=2 episodes per scenario, use cross-episode returns as V(s) baseline for lower-variance GAE
- **Typology-ambiguous scenarios**: Generate scenarios where the initial alert doesn't reveal the typology, forcing deeper investigation before the agent can commit to a tool strategy
- **TIPS-style potential computation**: Use a frozen policy's log-likelihood of the correct terminal decision as a per-step potential, replacing flat bonuses with information-theoretic shaping signals
