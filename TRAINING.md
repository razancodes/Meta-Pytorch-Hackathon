# Memex Training Guide — World Feedback for Financial Crime Detection

> Complete training pipeline for the Memex OS-Agent Benchmark.
> **Research title:** *World Feedback for Financial Crime Detection: Outcome-Grounded Reward Signals for Long-Horizon AML Investigation Agent*
> **GRPO** (TRL + Unsloth) is the **primary training path** — the Defender agent
> learns to investigate AML alerts using Group Relative Policy Optimization.
> Self-play (Launderer vs Defender) provides adversarial curriculum generation.
> The entire ecosystem uses the **Qwen 2.5** model family for unified
> tokenization and ChatML prompt formatting.

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
│  2. Model: Qwen2.5-7B-Instruct (Unsloth 4-bit + LoRA)        │
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

> **Why Qwen 2.5?** We use the Qwen 2.5 family across the entire stack (7B for training, 1.5B for compaction, 72B for production inference) to ensure **unified ChatML prompt formatting** and **consistent tokenization**. This eliminates the tokenizer mismatch trap that occurs when mixing model families.

### Training on Colab / HF

**Target:** A100 (40/80 GB VRAM) — recommended for production runs
**Also works on:** L4 (24 GB), T4 (16 GB) with reduced batch sizes
**Model:** `unsloth/Qwen2.5-7B-Instruct`
**Peak VRAM:** ~14 GB (4-bit + LoRA + G=4 generation)

### Colab Setup (Copy-Paste Cells)

```python
%%capture
# CELL 1: Install Training Stack
# Runtime → GPU → A100 (Colab Pro) or L4
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
# CELL 2: Clone the project

!git clone https://github.com/razancodes/Meta-Pytorch-Hackathon.git
%cd Meta-Pytorch-Hackathon

# Optional: symlink checkpoints to Google Drive for persistence
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/memex_checkpoints
# !ln -s /content/drive/MyDrive/memex_checkpoints checkpoints_drive
```

```python
# CELL 3: Verify environment (no GPU needed)

!python tests/test_smoke.py
# Expected: 8/8 tests passed ✓
```

```python
# CELL 4: Dry-run (4 prompts, 1 epoch, no WandB)

!python train_grpo.py --dry-run
# Verifies: model loading, prompt generation, reward function, GRPO update
```

```python
# CELL 5: ★ GRPO Training (~3-5 hours on A100)
# This is the primary training cell.

import wandb
wandb.login()

!python train_grpo.py \
    --model unsloth/Qwen2.5-7B-Instruct \
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
# CELL 5b (ALTERNATIVE): Run via HF Jobs CLI
# Uses pay-as-you-go HF compute ($0.80/hr for L4)

# !pip install huggingface_hub[cli]
# !hf jobs uv run --flavor l4x1 python train_grpo.py \
#     --num-prompts 500 --num-generations 4 --wandb-project memex-grpo
```

```python
# CELL 6: Evaluate best checkpoint (9 combos)

!python eval_harness.py --checkpoint checkpoints/defender-grpo
```

```python
# CELL 7: Run 1MDB demo + download AGUI replay

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
# CELL 8: Save checkpoints to Google Drive

import shutil, os

src = "/content/Meta-Pytorch-Hackathon/checkpoints"
dst = "/content/drive/MyDrive/memex_checkpoints"

shutil.copytree(src, dst, dirs_exist_ok=True)
print("✅ Done! Find it in your Drive → memex_checkpoints/")
```

```python
# CELL 9: Push trained model to HuggingFace Hub

from huggingface_hub import HfApi
api = HfApi()

# Push the LoRA adapter
api.upload_folder(
    folder_path="checkpoints/defender-grpo",
    repo_id="MuazTPM/defender-model",
    repo_type="model",
    commit_message="Defender GRPO checkpoint (Unsloth + TRL)"
)
print("✅ Model pushed to HuggingFace Hub!")
```

**GRPO CLI Reference:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/Qwen2.5-7B-Instruct` | Base model |
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

## ★ Workshop Experiments (GPU Server Required)

As detailed in the `paper_context.md` submission blueprint, the following two experiments must be executed on a GPU-enabled server (A100 recommended) to generate the empirical results required for the ICML submission. They validate the "World Feedback" architecture against standard RL tuning methods.

### Experiment 3: GRPO Reward Ablation

This experiment measures the independent contribution of each decomposed reward function. You must train 4 distinct models using identical hyperparameter seeds to isolate the effects of the shaping signals.

Run the following commands sequentially on your GPU server. Each run will take approximately 3-5 hours on an A100.

```bash
# 1. Full Condition (Baseline — All rewards enabled)
python train_grpo.py --output-dir checkpoints/exp3-full --num-prompts 250 --epochs 2

# 2. No-OS Condition (Ablates R4: OS Mechanics)
# Tests if the agent can discover OS tools via random exploration alone
python train_grpo.py --output-dir checkpoints/exp3-no-os --num-prompts 250 --epochs 2 --disable-reward R4

# 3. Terminal-Only Condition (Ablates R1, R2, R4)
# Simulates standard sparse-reward RL (rewarded only at the end of the episode)
python train_grpo.py --output-dir checkpoints/exp3-terminal --num-prompts 250 --epochs 2 --disable-reward R1 R2 R4

# 4. Format-Only Condition (Ablates R3, R4)
# Negative control. Agent is only rewarded for valid JSON, oblivious to the environment
python train_grpo.py --output-dir checkpoints/exp3-format --num-prompts 250 --epochs 2 --disable-reward R3 R4
```

*Note: Once these four checkpoints are trained, evaluate them locally using `eval_harness.py` to construct Table 2 in the paper.*

### Experiment 4: Adversarial Self-Play Curriculum

This experiment tests the transition from static procedural generation to a competitive two-agent dynamic. The `self_play.py` script alternatingly trains the Defender to catch laundering and the Launderer to generate evasive transaction graphs that successfully bypass the Defender's logic.

*Warning: This requires loading both models into VRAM simultaneously. Monitor your memory usage closely.*

```bash
# Run 3 alternating rounds (Launderer generation → Defender update)
python self_play.py \
    --defender-model unsloth/Qwen2.5-7B-Instruct \
    --launderer-model unsloth/Qwen2.5-7B-Instruct \
    --outer-rounds 3 \
    --defender-warmup 20 \
    --wandb-project memex-selfplay
```

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
| Base Qwen 2.5 7B 4-bit (NF4) | ~5.0 GB |
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
python archive/hotswap.py --base unsloth/Qwen2.5-7B-Instruct --adapter checkpoints/dpo-latest
```

### Archived File Reference

| File | Purpose |
|------|---------|
| `archive/train_defender_ppo.py` | Defender PPO with GAE, EMA baseline, batch normalization |
| `archive/train_launderer_ppo.py` | Launderer single-step PPO (generates evasive scenarios) |
| `archive/train_dpo.py` | Offline DPO trainer (continuous learning from user corrections) |
| `archive/hotswap.py` | Zero-downtime LoRA adapter hot-swap utility |

---

## ★ Production Deployment — AgentOS-Kernel (`agent_os_core/`)

> **Target:** A100 80GB VRAM (bare metal)  
> **Zero API calls.** Everything runs locally.

The `agent_os_core/` directory contains the **production inference runtime** — a high-performance middleware that replaces the Unsloth/TRL training loop with real-time, GPU-native agentic reasoning. It uses a fundamentally different architecture from the training pipeline:

| Aspect | GRPO Training (root) | AgentOS-Kernel (`agent_os_core/`) |
|--------|---------------------|-----------------------------------|
| Model | Qwen2.5-7B (4-bit LoRA) | Qwen2.5-72B-Instruct-AWQ (vLLM) |
| Purpose | Learn investigative policies | Deploy trained policies at scale |
| Memory | TRL handles context | 3-tier Cognitive Cache (L1/L2/L3) |
| Tool execution | In-process Python | Rust/Tokio via PyO3 (GIL-bypass) |
| Format enforcement | Reward shaping (R1) | JSON-schema constrained decoding |

### Prerequisites

| Dependency | Purpose |
|-----------|---------|
| `vllm` | High-throughput LLM serving (AWQ quantization) |
| `transformers` | Qwen2.5-1.5B compaction model |
| `sentence-transformers` | BGE embeddings + cross-encoder reranking |
| `lancedb` | L3 persistent vector storage |
| `tiktoken` | Token counting for L1 sliding window |
| `maturin` | Build Rust PyO3 bindings |

### VRAM Budget (A100 80GB)

| Component | Model | VRAM |
|-----------|-------|------|
| Reasoning Engine | Qwen2.5-72B-Instruct-AWQ via vLLM | ~38 GB |
| Compaction Engine | Qwen2.5-1.5B-Instruct via transformers | ~3 GB |
| Embedder | BAAI/bge-base-en-v1.5 | ~0.4 GB |
| Reranker | BAAI/bge-reranker-v2-m3 | ~1.1 GB |
| Tool Runtime | Rust/Tokio via PyO3 | 0 GB |
| **Total** | | **~42.5 GB** |
| **Headroom (A100 80GB)** | | **~37.5 GB ✓** |

### Setup

```bash
# 1. Build the Rust async tool runtime
cd agent_os_core
pip install maturin tiktoken lancedb numpy pyarrow
maturin develop --release

# 2. Install inference models (first run downloads ~40GB)
pip install vllm transformers sentence-transformers

# 3. Run tests in mock mode (no GPU required)
python test_runtime.py       # Rust runtime: 5 tests
python test_memory.py        # L1/L2 cache: 5 tests
python test_l3.py            # L3 index: 5 tests
python test_integration.py   # Full orchestrator: 6 tests
```

### 3-Tier Cognitive Cache

The Cognitive Cache solves **context starvation** — the [Lost in the Middle](https://arxiv.org/abs/2307.03172) problem where evidence gets buried in the attention dead zone.

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1 — Raw Conversation Window (6K tokens)                   │
│  │  Sliding window of user/assistant/tool turns              │
│  │  Token-counted via tiktoken                               │
│  │  Overflow → evict oldest N turns → compact to L2          │
│                                                              │
│  L2 — Structured Scratchpad (2K tokens)                     │
│  │  Entity→facts map: {"CUST-A": ["PEP", "offshore"]}      │
│  │  Compacted by Qwen2.5-1.5B (structured JSON extraction)  │
│  │  Injected at PROMPT START (high attention position)       │
│  │  Deduplicates facts across compaction cycles              │
│  │  Overflow → archive oldest entities to L3                 │
│                                                              │
│  L3 — LanceDB Persistent Archive (unbounded)               │
│     BGE-base-en-v1.5 embeddings for vector search            │
│     Cross-encoder gating (BGE-reranker-v2-m3)               │
│     Only injected when relevance score > 0.50               │
│     Injected at PROMPT END (high attention position)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### How It Integrates with the Root Environment

The AgentOS orchestrator (`agent_os.py`) can be wired to execute tools against the OpenEnv server (`openenv_server.py`):

```python
# Connect AgentOS to the Memex environment
from agent_os import AgentOS

os = AgentOS(use_mock_llm=False)  # Uses vLLM + Qwen2.5-72B

# For production: set Rust runtime to HTTP dispatch mode
os._tool_runtime.set_mode("http", "http://localhost:8000")

# Now tool calls route through the FastAPI server
result = await os.step("Investigate alert ALERT-2024-1MDB-7701")
```

### Mock Mode (Testing Without GPU)

All components support `use_mock=True` for CPU-only testing:

```python
from agent_os import AgentOS

os = AgentOS(
    use_mock_llm=True,           # Keyword-based mock instead of vLLM
    use_mock_compactor=True,     # Heuristic compaction instead of Qwen 1.5B
    use_mock_embeddings=True,    # Deterministic hash vectors instead of BGE
)
result = await os.step("Investigate this alert")
```

---

## Complete File Reference

| File | Purpose |
|------|---------|
| **Root — Environment** | |
| `models.py` | Pydantic data contracts: `AMLAction`, `AMLObservation`, `AMLState`, `AGUIState` |
| `state_manager.py` | OS mechanics engine: Virtual Memory (RAM/Disk), Interrupts (async jobs), Kernel directives |
| `server/aml_environment.py` | Core OpenEnv environment: 18-tool dispatch, reward calculation, scenario lifecycle |
| `openenv_server.py` | FastAPI server: `/reset`, `/step`, `/state`, `/health` — OpenEnv SDK compatible |
| `client.py` | HTTP client wrapper with typed methods for all 18 tools |
| `inference.py` | Standalone ReAct inference agent (OpenAI-compatible API) |
| `openenv.yaml` | OpenEnv environment manifest |
| **Root — Training & Evaluation** | |
| `train_grpo.py` | ★ Primary GRPO training script (TRL + Unsloth, 4 reward functions) |
| `self_play.py` | Two-agent self-play orchestrator (Defender vs Launderer) |
| `eval_harness.py` | Multi-typology benchmark suite (6 scripted scenarios) |
| `demo_eval.py` | 1MDB demo evaluation with AGUI replay recording |
| **Root — Scenarios & Grading** | |
| `scenarios/procedural_generator.py` | Procedural AML scenario generation engine |
| `scenarios/compliance_manual.py` | Searchable compliance knowledge base |
| `graders/grader.py` | Dense reward engine (TP/TN/FP/FN scoring) |
| `curriculum/plr_engine.py` | Prioritized Level Replay curriculum |
| **AgentOS-Kernel** | |
| `agent_os_core/agent_os.py` | Production orchestrator: vLLM + Cognitive Cache + Rust runtime |
| `agent_os_core/memory_manager.py` | L1/L2 cognitive cache with LLM compaction |
| `agent_os_core/l3_index.py` | L3 LanceDB index with cross-encoder gating |
| `agent_os_core/src/lib.rs` | Rust Tokio async runtime (PyO3 bindings) |
| `agent_os_core/Cargo.toml` | Rust crate config (pyo3, tokio, reqwest) |
| `agent_os_core/pyproject.toml` | Maturin build config |
| **Documentation** | |
| `README.md` | Project overview, architecture, quick-start |
| `TRAINING.md` | This file — full training + deployment guide |
| `BLOG.md` | Post-mortem: debugging, reward design, 1MDB walkthrough |
| **Infrastructure** | |
| `Dockerfile` | HF Spaces deployment container |
| `requirements.txt` | Python dependencies |
| `.hfignore` | HF Space upload exclusions (excludes `agent_os_core/`) |
| `.gitignore` | Git exclusions (build artifacts, checkpoints, caches) |
