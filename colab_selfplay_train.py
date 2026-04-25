#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MEMEX AML SELF-PLAY TRAINING  —  Colab L4 Script                       ║
║                                                                          ║
║  Two-Agent PPO Self-Play for AML Investigation Environment               ║
║  Optimized for: NVIDIA L4 (24 GB VRAM) / Google Colab Pro                ║
║                                                                          ║
║  Schedule:                                                               ║
║    Cell 1  — GPU check & environment setup                               ║
║    Cell 2  — Clone repo & install dependencies                           ║
║    Cell 3  — Smoke test (compileall + quick import check)                ║
║    Cell 4  — Phase 1: Defender warm-start (procedural only)              ║
║    Cell 5  — Phase 2: Launderer PPO (against frozen Defender)            ║
║    Cell 6  — Phase 3: Defender PPO on mixed scenarios                    ║
║    Cell 7  — Full self-play orchestrator (Phases 1-3 × N rounds)         ║
║    Cell 8  — Evaluation & demo                                           ║
║    Cell 9  — Push checkpoints to HuggingFace Hub                         ║
║                                                                          ║
║  VRAM Budget (L4 = 24 GB):                                               ║
║    Base 8B 4-bit:   ~5.5 GB                                              ║
║    LoRA adapters:   ~0.3 GB                                              ║
║    KV cache:        ~2.0 GB                                              ║
║    Optimizer:       ~1.2 GB                                              ║
║    Activations:     ~3-6 GB (gradient checkpointing)                     ║
║    Headroom:        ~9 GB  ← safe margin                                 ║
║                                                                          ║
║  Usage:                                                                  ║
║    1. Open in Colab, select L4 runtime                                   ║
║    2. Run cells top-to-bottom                                            ║
║    3. For quick test: run Cell 4 with --dry-run                          ║
║    4. For full run: run Cell 7 (orchestrator handles all phases)          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: GPU CHECK & ENVIRONMENT SETUP                                  ║
# ║                                                                          ║
# ║  Verifies L4 GPU is available, prints VRAM, sets environment vars.       ║
# ║  ABORT if no GPU or < 20 GB VRAM — these trainers require GPU.           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# # 🧠 Memex AML Self-Play Training
# **Two-Agent PPO**: Defender (AML investigator) vs Launderer (scenario generator)
#
# **Runtime**: L4 GPU (24 GB VRAM) recommended
#
# **Schedule**:
# 1. Defender warm-start on procedural scenarios
# 2. Launderer PPO against frozen Defender
# 3. Defender PPO on mixed (procedural + Launderer) scenarios
# 4. Repeat 2-3 for N outer rounds

# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 1: GPU Check & Environment Setup
# ═══════════════════════════════════════════════════════════════════════

import os
import subprocess
import sys

# Force CUDA memory management for large model training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings
os.environ["WANDB_SILENT"] = "true"  # Cleaner output

print("=" * 70)
print("  MEMEX SELF-PLAY — ENVIRONMENT CHECK")
print("=" * 70)

# Check GPU
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No CUDA GPU detected. Switch to L4 runtime.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  ✓ GPU:  {gpu_name}")
    print(f"  ✓ VRAM: {vram_gb:.1f} GB")

    if vram_gb < 20:
        print(f"  ⚠ WARNING: {vram_gb:.1f} GB may be insufficient. L4 (24 GB) recommended.")
    else:
        print(f"  ✓ VRAM sufficient for 8B 4-bit + LoRA training")

    # CUDA version
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ PyTorch: {torch.__version__}")

except ImportError:
    print("  ⚠ PyTorch not yet installed (will install in Cell 2)")

print(f"  ✓ Python: {sys.version.split()[0]}")
print(f"  ✓ Disk:   {subprocess.getoutput('df -h / | tail -1 | awk \"{print $4}\"')} free")
print("=" * 70)


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Clone Repository & Install Dependencies
# ═══════════════════════════════════════════════════════════════════════
#
# Installs:
#   - unsloth (fast LoRA fine-tuning, 2x speedup + 60% VRAM reduction)
#   - wandb (experiment tracking)
#   - Project dependencies (fastapi, pydantic, etc.)
#
# ⏱ Expected time: ~3-5 minutes
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  CELL 2: INSTALLING DEPENDENCIES")
print("=" * 70)

# ── 2a. Clone the repository ──
REPO_URL = "https://github.com/razancodes/Meta-Pytorch-Hackathon.git"
PROJECT_DIR = "/content/MetaHack"

if not os.path.exists(PROJECT_DIR):
    print(f"\n[1/4] Cloning {REPO_URL}...")
    os.system(f"git clone {REPO_URL} {PROJECT_DIR}")
else:
    print(f"\n[1/4] Repo already cloned. Pulling latest...")
    os.system(f"git -C {PROJECT_DIR} pull origin main")

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
print(f"  ✓ Working directory: {os.getcwd()}")

# ── 2b. Install Unsloth (must come BEFORE torch upgrade to avoid conflicts) ──
print("\n[2/4] Installing Unsloth (fast LoRA engine)...")
os.system(
    "pip install -q 'unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git' "
    "&& pip install -q --no-deps 'trl<0.9.0' peft accelerate bitsandbytes"
)

# ── 2c. Install WandB ──
print("\n[3/4] Installing WandB...")
os.system("pip install -q wandb")

# ── 2d. Install project dependencies ──
print("\n[4/4] Installing project dependencies...")
os.system("pip install -q -r requirements.txt 2>/dev/null")

# Verify critical imports
print("\n  Verifying imports...")
try:
    import torch
    from unsloth import FastLanguageModel
    print(f"  ✓ torch {torch.__version__}")
    print(f"  ✓ unsloth loaded")
    print(f"  ✓ VRAM: {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    print("  Try: Runtime → Restart Runtime, then re-run this cell")

print("\n" + "=" * 70)
print("  ✓ DEPENDENCIES INSTALLED")
print("=" * 70)


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Smoke Test — Verify Codebase Integrity
# ═══════════════════════════════════════════════════════════════════════
#
# Runs:
#   - python -m compileall (syntax check on all .py files)
#   - Quick import of all training modules
#   - Procedural scenario generation test
#   - Grader math spot-check
#
# ⏱ Expected time: ~10 seconds
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  CELL 3: SMOKE TEST")
print("=" * 70)

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# 3a. Compile check
print("\n[1/4] Compile check...")
result = os.system("python -m compileall -q . 2>&1")
print(f"  {'✓' if result == 0 else '❌'} compileall {'passed' if result == 0 else 'FAILED'}")

# 3b. Import check
print("\n[2/4] Import check...")
try:
    from models import AMLState, AMLAction, AMLObservation
    from server.aml_environment import AMLEnvironment
    from graders.grader import AMLGrader
    from state_manager import StateManager
    from scenarios.procedural_generator import generate_scenario
    print("  ✓ All core modules imported")
except Exception as e:
    print(f"  ❌ Import failed: {e}")

# 3c. Scenario generation test
print("\n[3/4] Scenario generation test...")
try:
    sc = generate_scenario("easy", "structuring")
    gt = sc.ground_truth
    print(f"  ✓ Generated: {gt.get('typology', '?')} / is_suspicious={gt.get('is_suspicious', '?')}")
    print(f"  ✓ Key entities: {gt.get('key_entities', [])[:3]}")
except Exception as e:
    print(f"  ❌ Scenario generation failed: {e}")

# 3d. Grader math spot-check
print("\n[4/4] Grader math spot-check...")
try:
    grader = AMLGrader()

    # TP case: suspicious + file_sar
    state = AMLState()
    state.step_count = 10
    result = grader.grade(
        ground_truth={"is_suspicious": True, "typology": "structuring",
                      "key_entities": ["CUST001"], "key_findings": ["sub_threshold"]},
        decision="file_sar", findings=["sub_threshold"],
        entities_flagged=["CUST001"], typology="structuring", state=state,
    )
    print(f"  ✓ TP score: {result['total']:+.4f} (detection={result['detection']})")
    assert result["detection"] == "TP", f"Expected TP, got {result['detection']}"

    # TN case: clean + close_alert
    state2 = AMLState()
    state2.step_count = 8
    result2 = grader.grade(
        ground_truth={"is_suspicious": False, "typology": "clean",
                      "key_entities": [], "key_findings": []},
        decision="close_alert", findings=[], entities_flagged=[],
        typology="false_positive", state=state2,
    )
    print(f"  ✓ TN score: {result2['total']:+.4f} (detection={result2['detection']})")
    assert result2["detection"] == "TN", f"Expected TN, got {result2['detection']}"

    print("  ✓ All grader checks passed")
except Exception as e:
    print(f"  ❌ Grader check failed: {e}")

print("\n" + "=" * 70)
print("  ✓ SMOKE TEST COMPLETE")
print("=" * 70)


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Phase 1 — Defender Warm-Start (Procedural Only)
# ═══════════════════════════════════════════════════════════════════════
#
# Trains the Defender agent on procedural AML scenarios.
# This bootstraps the Defender before adversarial self-play begins.
#
# Config:
#   Model:       Llama-3.1-8B-Instruct (4-bit quantized)
#   LoRA:        r=16, α=32, targets=all linear layers
#   PPO:         lr=5e-6, clip=0.2, KL=0.05, GAE(γ=0.99, λ=0.95)
#   Schedule:    20 iterations × 4 episodes/iter = 80 episodes
#   Max steps:   25 per episode
#
# ⏱ Expected time:
#   - Dry run:   ~2 minutes (2 iters × 1 ep)
#   - Full run:  ~90 minutes (20 iters × 4 ep)
#
# Output: checkpoints/defender/round-0/best/
# ═══════════════════════════════════════════════════════════════════════

import gc
import torch

# ── Configuration ──
DRY_RUN = False           # Set True for quick 2-iteration test
DEFENDER_WARMUP_ITERS = 20  # Phase 1 iterations (20 recommended)
EPISODES_PER_ITER = 4      # Episodes per iteration
WANDB_PROJECT = "memex-selfplay"

# Set to True and run `wandb login` in a cell above if you want tracking
USE_WANDB = False

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 4: PHASE 1 — DEFENDER WARM-START")
print("=" * 70)

# Force clean VRAM state
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"  VRAM before: {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

from train_defender_ppo import DefenderPPOConfig, train as train_defender

defender_cfg = DefenderPPOConfig(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=2048,
    total_iterations=DEFENDER_WARMUP_ITERS,
    episodes_per_iter=EPISODES_PER_ITER,
    scenario_source="procedural",
    launderer_checkpoint="",  # No launderer yet
    mix_ratio=0.0,
    output_dir=os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-0"),
    wandb_project=WANDB_PROJECT,
    dry_run=DRY_RUN,
)

phase1_score = train_defender(defender_cfg)

# Clean up
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

DEFENDER_CKPT = os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-0", "best")
print(f"\n  ✓ Phase 1 complete. Best score: {phase1_score:+.4f}")
print(f"  ✓ Defender checkpoint: {DEFENDER_CKPT}")
print(f"  ✓ VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Phase 2 — Launderer PPO (vs Frozen Defender)
# ═══════════════════════════════════════════════════════════════════════
#
# Trains the Launderer to generate scenarios that fool the Defender.
# The Defender is frozen (loaded for inference only during scoring).
#
# Architecture:
#   1. Launderer generates scenario JSON
#   2. Frozen Defender plays the scenario
#   3. Launderer reward = -Defender_score (adversarial)
#
# VRAM-safe: Full unload/reload cycle per scoring round.
# The Launderer and Defender NEVER coexist in VRAM.
#
# Config:
#   PPO:         lr=1e-5, clip=0.2, KL=0.05
#   Schedule:    10 iterations × 4 episodes/iter = 40 episodes
#   Max tokens:  2048 (full scenario JSON)
#
# ⏱ Expected time:
#   - Dry run:   ~3 minutes
#   - Full run:  ~120 minutes (includes Defender scoring)
#
# Output: checkpoints/launderer/round-1/best/
# ═══════════════════════════════════════════════════════════════════════

import gc
import torch

# ── Configuration ──
DRY_RUN = False
LAUNDERER_ITERS = 10  # Phase 2 iterations (10 recommended)
EPISODES_PER_ITER = 4

# Path to Defender checkpoint from Phase 1
DEFENDER_CKPT = os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-0", "best")

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 5: PHASE 2 — LAUNDERER PPO (vs Frozen Defender)")
print("=" * 70)

# Verify Defender checkpoint exists
if not os.path.exists(DEFENDER_CKPT):
    print(f"  ❌ Defender checkpoint not found: {DEFENDER_CKPT}")
    print(f"  → Run Cell 4 first, or set DEFENDER_CKPT to your checkpoint path")
else:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Defender checkpoint: {DEFENDER_CKPT}")

    from train_launderer_ppo import LaundererPPOConfig, train as train_launderer

    launderer_cfg = LaundererPPOConfig(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=4096,    # Launderer needs more context for JSON
        max_new_tokens=2048,
        total_iterations=LAUNDERER_ITERS,
        episodes_per_iter=EPISODES_PER_ITER,
        defender_checkpoint=DEFENDER_CKPT,
        defender_max_steps=15,
        output_dir=os.path.join(PROJECT_DIR, "checkpoints", "launderer", "round-1"),
        wandb_project=WANDB_PROJECT,
        dry_run=DRY_RUN,
    )

    phase2_score = train_launderer(launderer_cfg)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    LAUNDERER_CKPT = os.path.join(PROJECT_DIR, "checkpoints", "launderer", "round-1", "best")
    print(f"\n  ✓ Phase 2 complete. Best reward: {phase2_score:+.4f}")
    print(f"  ✓ Launderer checkpoint: {LAUNDERER_CKPT}")
    print(f"  ✓ VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Phase 3 — Defender PPO on Mixed Scenarios
# ═══════════════════════════════════════════════════════════════════════
#
# Trains the Defender on a MIX of procedural + Launderer scenarios.
# This forces the Defender to handle adversarial cases.
#
# Architecture:
#   1. Pre-generate Launderer scenarios (batch, validated)
#   2. Each episode: flip coin → procedural or Launderer scenario
#   3. Defender plays episode, gets composite reward
#
# VRAM-safe: Launderer scenarios are pre-generated and cached
# before Defender training begins. No concurrent model loading.
#
# Config:
#   Mix ratio:   0.5 (50% Launderer scenarios)
#   PPO:         Same as Phase 1
#   Schedule:    15 iterations × 4 episodes/iter = 60 episodes
#
# ⏱ Expected time:
#   - Dry run:   ~5 minutes (includes pre-generation)
#   - Full run:  ~120 minutes
#
# Output: checkpoints/defender/round-1/best/
# ═══════════════════════════════════════════════════════════════════════

import gc
import torch

# ── Configuration ──
DRY_RUN = False
DEFENDER_MIXED_ITERS = 15  # Phase 3 iterations
EPISODES_PER_ITER = 4
MIX_RATIO = 0.5           # 50% launderer, 50% procedural

# Checkpoints from previous phases
LAUNDERER_CKPT = os.path.join(PROJECT_DIR, "checkpoints", "launderer", "round-1", "best")

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 6: PHASE 3 — DEFENDER PPO (MIXED SCENARIOS)")
print("=" * 70)

if not os.path.exists(LAUNDERER_CKPT):
    print(f"  ❌ Launderer checkpoint not found: {LAUNDERER_CKPT}")
    print(f"  → Run Cell 5 first, or set LAUNDERER_CKPT to your checkpoint path")
else:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Launderer checkpoint: {LAUNDERER_CKPT}")
    print(f"  Mix ratio: {MIX_RATIO}")

    from train_defender_ppo import DefenderPPOConfig, train as train_defender

    defender_mixed_cfg = DefenderPPOConfig(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=2048,
        total_iterations=DEFENDER_MIXED_ITERS,
        episodes_per_iter=EPISODES_PER_ITER,
        scenario_source="mixed",
        launderer_checkpoint=LAUNDERER_CKPT,
        mix_ratio=MIX_RATIO,
        output_dir=os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-1"),
        wandb_project=WANDB_PROJECT,
        dry_run=DRY_RUN,
    )

    phase3_score = train_defender(defender_mixed_cfg)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"\n  ✓ Phase 3 complete. Best score: {phase3_score:+.4f}")
    print(f"  ✓ VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Full Self-Play Orchestrator (All Phases × N Rounds)
# ═══════════════════════════════════════════════════════════════════════
#
# ⚡ THIS IS THE MAIN TRAINING CELL ⚡
#
# Runs the complete alternating best-response self-play loop:
#
#   Phase 1: Defender warm-start (procedural, 20 iters)
#   ┌── Outer Round 1..N:
#   │   Phase 2: Launderer PPO vs frozen Defender (10 iters)
#   │   Phase 3: Defender PPO on mixed scenarios (15 iters)
#   └── End Round
#
# The orchestrator handles:
#   - Checkpoint management (population tracking, best selection)
#   - VRAM-safe model swapping (full unload/reload)
#   - Linear mix ratio schedule (0.3 → 0.7)
#   - Real score propagation to population.best()
#
# If you already ran Cells 4-6, you can SKIP this cell.
# If you want a fresh start, DELETE checkpoints/ and run this.
#
# Config:
#   Outer rounds:   3  (total ~6 hours on L4)
#   Defender warm:  20 iters
#   Launderer:      10 iters per round
#   Defender mixed: 15 iters per round
#   Mix schedule:   linear 0.3 → 0.7
#
# ⏱ Expected time:
#   - Dry run:   ~10 minutes
#   - 3 rounds:  ~6-8 hours on L4
#
# Output: checkpoints/population_history.json
# ═══════════════════════════════════════════════════════════════════════

import gc
import torch

# ── Configuration ──
DRY_RUN = False            # Quick 2-iter test per phase
OUTER_ROUNDS = 3           # Number of L→D alternating rounds
DEFENDER_WARMUP = 20       # Phase 1 iterations
DEFENDER_ITERS = 15        # Phase 3+ iterations per round
LAUNDERER_ITERS = 10       # Phase 2+ iterations per round
DEFENDER_EPISODES = 4      # Episodes per Defender iteration
LAUNDERER_EPISODES = 4     # Episodes per Launderer iteration
MIX_START = 0.3            # Initial Launderer scenario fraction
MIX_MAX = 0.7              # Final Launderer scenario fraction
MIX_SCHEDULE = "linear"    # "linear" or "fixed"
WANDB_PROJECT = "memex-selfplay"

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 7: FULL SELF-PLAY ORCHESTRATOR")
print("=" * 70)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"  VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB")

from self_play import SelfPlayConfig, self_play

sp_cfg = SelfPlayConfig(
    base_model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=2048,
    outer_rounds=OUTER_ROUNDS,
    defender_warmup_iters=DEFENDER_WARMUP,
    defender_iters=DEFENDER_ITERS,
    launderer_iters=LAUNDERER_ITERS,
    defender_episodes_per_iter=DEFENDER_EPISODES,
    launderer_episodes_per_iter=LAUNDERER_EPISODES,
    mix_ratio_start=MIX_START,
    mix_ratio_max=MIX_MAX,
    mix_ratio_schedule=MIX_SCHEDULE,
    checkpoint_dir=os.path.join(PROJECT_DIR, "checkpoints"),
    wandb_project=WANDB_PROJECT,
    dry_run=DRY_RUN,
)

self_play(sp_cfg)

gc.collect()
torch.cuda.empty_cache()
print(f"\n  ✓ Self-play complete")
print(f"  ✓ VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 8: Evaluation & Demo
# ═══════════════════════════════════════════════════════════════════════
#
# Runs the trained Defender through the 1MDB demo scenario.
# Captures AGUI visualization payloads for frontend replay.
#
# Can run in two modes:
#   - LLM mode: loads best Defender checkpoint + runs inference
#   - Scripted mode: no GPU needed (for testing frontend)
#
# ⏱ Expected time: ~2-5 minutes
# ═══════════════════════════════════════════════════════════════════════

import gc
import torch

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 8: EVALUATION & DEMO")
print("=" * 70)

# ── Choose mode ──
USE_LLM = True   # False = scripted (no GPU needed)

# Find best Defender checkpoint
BEST_DEFENDER = None
for ckpt_path in [
    os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-0", "best"),
    os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-1", "best"),
    os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-2", "best"),
    os.path.join(PROJECT_DIR, "checkpoints", "defender", "round-3", "best"),
    os.path.join(PROJECT_DIR, "checkpoints", "best"),
]:
    if os.path.exists(ckpt_path):
        BEST_DEFENDER = ckpt_path

if USE_LLM and BEST_DEFENDER:
    print(f"  Running LLM demo with: {BEST_DEFENDER}")
    gc.collect()
    torch.cuda.empty_cache()

    from demo_eval import run_llm_demo
    run_llm_demo(
        model_path=BEST_DEFENDER,
        output_dir=os.path.join(PROJECT_DIR, "demo_output"),
    )
elif USE_LLM and not BEST_DEFENDER:
    print("  ⚠ No checkpoint found. Running scripted demo instead.")
    from demo_eval import run_scripted_demo
    run_scripted_demo(output_dir=os.path.join(PROJECT_DIR, "demo_output"))
else:
    print("  Running scripted demo (no GPU)")
    from demo_eval import run_scripted_demo
    run_scripted_demo(output_dir=os.path.join(PROJECT_DIR, "demo_output"))

print(f"\n  ✓ Demo output saved to: {PROJECT_DIR}/demo_output/")
print(f"  ✓ Files: agui_step_*.json + episode_meta.json")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Push Checkpoints to HuggingFace Hub (Optional)
# ═══════════════════════════════════════════════════════════════════════
#
# Uploads the best Defender and Launderer LoRA adapters to HF Hub.
# Requires: `huggingface-cli login` with a write token.
#
# ⏱ Expected time: ~1-2 minutes
# ═══════════════════════════════════════════════════════════════════════

PUSH_TO_HF = False   # Set True to push
HF_REPO_ID = "your-username/memex-aml-defender"  # Change this

if PUSH_TO_HF:
    print("\n" + "=" * 70)
    print("  CELL 9: PUSHING TO HUGGINGFACE HUB")
    print("=" * 70)

    os.system("pip install -q huggingface_hub")

    from huggingface_hub import HfApi

    api = HfApi()
    if BEST_DEFENDER and os.path.exists(BEST_DEFENDER):
        print(f"  Uploading Defender: {BEST_DEFENDER} → {HF_REPO_ID}")
        api.upload_folder(
            folder_path=BEST_DEFENDER,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ Defender pushed to https://huggingface.co/{HF_REPO_ID}")
    else:
        print("  ⚠ No Defender checkpoint found to push")
else:
    print("\n  [CELL 9] Skipped — set PUSH_TO_HF = True to upload checkpoints")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Cleanup & Download Checkpoints
# ═══════════════════════════════════════════════════════════════════════
#
# Zips all checkpoints + demo output for local download.
# Also prints a summary of all training artifacts.
# ═══════════════════════════════════════════════════════════════════════

import os
import glob

os.chdir(PROJECT_DIR)

print("\n" + "=" * 70)
print("  CELL 10: ARTIFACTS SUMMARY")
print("=" * 70)

# List all checkpoints
print("\n  Checkpoints:")
for root, dirs, files in os.walk("checkpoints"):
    for d in sorted(dirs):
        path = os.path.join(root, d)
        n_files = len(os.listdir(path))
        print(f"    📁 {path}/ ({n_files} files)")
    break  # Only top-level

# List demo output
print("\n  Demo output:")
for f in sorted(glob.glob("demo_output/*")):
    size = os.path.getsize(f) / 1024
    print(f"    📄 {f} ({size:.1f} KB)")

# Population history
pop_path = "checkpoints/population_history.json"
if os.path.exists(pop_path):
    import json
    with open(pop_path) as f:
        pop = json.load(f)
    print(f"\n  Population history ({len(pop)} entries):")
    for entry in pop:
        print(f"    {entry['agent']:>10} round={entry['round']}  score={entry['score']:+.4f}  path={entry['path']}")

# Zip for download
print("\n  Creating archive for download...")
os.system("tar -czf /content/memex_checkpoints.tar.gz checkpoints/ demo_output/ 2>/dev/null")
if os.path.exists("/content/memex_checkpoints.tar.gz"):
    size_mb = os.path.getsize("/content/memex_checkpoints.tar.gz") / 1e6
    print(f"  ✓ Archive: /content/memex_checkpoints.tar.gz ({size_mb:.1f} MB)")
    print("  → Use Files panel (left sidebar) to download")

print("\n" + "=" * 70)
print("  ✓ TRAINING COMPLETE — ALL ARTIFACTS SAVED")
print("=" * 70)
