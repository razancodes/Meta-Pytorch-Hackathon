# Memex: Project Context

> Living document tracking the current state of the Memex OS-Agent Benchmark.
> Last updated: 2026-04-23

## What Is Memex?

Memex is a **B2B enterprise benchmarking environment** built for the Meta / Hugging Face OpenEnv Hackathon. It tests whether an LLM can function as a Turing-complete Operating System over long-horizon tasks, using complex Anti-Money Laundering (AML) investigations as the obstacle course.

The "OS" metaphor is an **architectural framework**, not a literal operating system. It solves the LLM context window bottleneck by giving the agent explicit tools to manage its own memory, handle async operations, and self-improve its decision rules.

---

## Current Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Models** | `models.py` | Single source of truth for all Pydantic types: `AMLAction`, `AMLObservation`, `AMLState`, `AsyncJobInfo`, `AGUIState` |
| **State Manager** | `state_manager.py` | OS mechanics engine: RAM eviction (2-slot context), Disk persistence, Async Job Queue, Kernel Directives |
| **Procedural Generator** | `scenarios/procedural_generator.py` | Dynamic POMDP scenario builder: 3 typologies × 3 difficulties, unique IDs per episode |
| **Adversary Agent** | `scenarios/adversary_agent.py` | ★ NEW (2026-04-23) LLM-backed evasive scenario generator (mule rings, pass-throughs, phantom invoices) |
| **Battle Orchestrator** | `train_adversary.py` | ★ NEW (2026-04-23) GAN-style adversarial loop that tests the Defender and saves evasive graphs to SQLite |
| **Adversarial DB** | `adversarial_successes.db` | ★ NEW (2026-04-23) SQLite database storing failed scenarios (negative examples) |
| **Compliance Manual** | `scenarios/compliance_manual.py` | Searchable AML rule corpus for kernel updates |
| **Environment** | `server/aml_environment.py` | Core environment: 15 tool handlers + State Manager integration |
| **Grader** | `graders/grader.py` | Dense reward: per-step micro-rewards + terminal composite score [-1.0, +1.0] |
| **OpenEnv Server** | `openenv_server.py` | Production FastAPI entrypoint — OpenEnv SDK `create_app()` + standalone fallback |
| **Legacy Server** | `server/app.py` | FastAPI HTTP server (dual-mode: OpenEnv / standalone, used by trainers) |
| **Client** | `client.py` | HTTP client with all 15 tool wrappers |
| **Inference** | `inference.py` | ReAct agent loop with OS-mechanic awareness |
| **PPO Trainer (L4)** | `train_ppo.py` | Custom step-level PPO (Unsloth 4-bit + LoRA, L4-optimized, `--use-plr`) |
| **GRPO Trainer (L4)** | `train_grpo.py` | Group Relative Policy Optimization (no critic, group-relative advantage) |
| **PPO Trainer (70B)** | `train_ppo_70b.py` | Multi-GPU DeepSpeed ZeRO-3 PPO for 70B on A100 cluster (proof of scalability) |
| **DPO Trainer** | `train_dpo.py` | Offline DPO continuous learning from user preference pairs |
| **LoRA Hot-Swap** | `hotswap.py` | Zero-downtime LoRA adapter reload into running models |
| **PLR Engine** | `curriculum/plr_engine.py` | Prioritized Level Replay: regret-weighted scenario sampling buffer |
| **Proxy Oracle** | `curriculum/oracle.py` | Regret computation: `1.0 - protagonist_score` |
| **Demo** | `demo_eval.py` | 1MDB-inspired demo with AGUI replay capture |
| **Tests** | `tests/test_smoke.py` | 8 end-to-end smoke tests |
| **PLR Tests** | `tests/test_plr.py` | PLR engine unit tests |

### OS Mechanics

1. **Virtual Memory (RAM Eviction)**
   - Agent context window = last 2 observations only
   - `write_to_case_file(content)` → pages data to persistent disk (+0.10 reward, hard-capped at 3 per episode)
   - Page Fault: referencing evicted data NOT on disk → -0.05 penalty

2. **Interrupts (Async Queue)**
   - `request_wire_trace(entity_id)` → background job with 2–4 step ETA
   - `retrieve_async_result(job_id)` → fetch when ETA=0
   - Async Timeout: premature retrieval → -0.10 penalty

3. **Kernel Updates (Self-Improvement)**
   - `search_compliance_manual(query)` → find AML rules
   - `update_system_prompt(rule)` → inject into active kernel directives (+0.15 reward, hard-capped at 2 per episode)

### Tool Roster (15 total)

**Domain (10):** review_alert, get_customer_profile, query_transactions, check_watchlist, trace_network, check_source_of_funds, check_market_price, assess_risk, file_sar, close_alert

**OS-Mechanic (5):** write_to_case_file, request_wire_trace, retrieve_async_result, search_compliance_manual, update_system_prompt

### Reward System

**Per-step:** action cost (-0.02), redundancy (-0.03), unique tool (+0.03), page fault (-0.05), async timeout (-0.10), disk write (+0.10, max 3/episode), kernel inject (+0.15, max 2/episode)

**Terminal:** decision accuracy (0.30) + typology (0.15) + findings coverage (0.20) + entity F1 (0.15) + UBO identification (0.05) + Phase 3 pillar tools (0.05) + efficiency (0.10) → mapped to [-1.0, +1.01] (including OS micro-rewards)

### Scenario Generation

3 ML typologies: **Structuring** (smurfing), **Layering** (shell companies), **Trade-Based ML** (invoice manipulation)
3 difficulties: **Easy** (1 culprit, minimal noise), **Medium** (3+ entities, moderate noise), **Hard** (5+ entities, deep networks, decoys)

All entity IDs are procedurally generated per episode — no memorization possible.

---

## Test Status

```
✓ 9/9 typology/difficulty combos generate valid scenarios
✓ Unique IDs confirmed across episodes (anti-memorization)
✓ Compliance manual search works
✓ Noise scales with difficulty
✓ Full episode: structuring (easy) → score +1.01
✓ Full episode: layering (medium) → score +1.01
✓ Full episode: trade-based ML (hard) → score +1.01
✓ PLR curriculum engine: buffer, sampling, metrics, save/load
```

**All 8/8 smoke tests pass.**

---

## Frontend Architecture: AGUI (Agentic Graphical User Interface)

The backend is invisible by design — an LLM managing RAM eviction and async queues produces no visual signal. The **AGUI** solves this by emitting a strict JSON `agui_state` payload after every environment step, which the Next.js frontend consumes to render a real-time **5-Panel Tactical Dashboard**:

```
┌─────────────────────────────────────┬─────────────────────────────────┐
│        ENTITY GRAPH (Panel 1)       │   SYSTEM RESOURCES (Panel 3)    │
│        (60% Width)                  │    - RAM & Disk Storage         │
│                                     │    - Active Async Processes     │
│  [3D Globe / Flat Map Toggle]       │    - Kernel Directives          │
│  react-globe.gl vectors             ├─────────────────────────────────┤
│  Cytoscape.js Cola Physics          │   AGENT TERMINAL (Panel 4)      │
│                                     │    - ReAct scratchpad reasoning │
│  (Smooth Incremental Updates)       │    - Tool execution logs        │
│                                     ├─────────────────────────────────┤
│   *GLOBAL THREAT MAP (Panel 2)      │ CURRICULUM ENGINE (Panel 5, opt)│
│    is toggled with Entity Graph     │    - PLR Regret Metrics         │
│                                     │    - Scenario Difficulty Gauge  │
└─────────────────────────────────────┴─────────────────────────────────┘
```

**Data contract:** `state_manager.py` builds the `AGUIState` object (defined in `models.py`) containing `ram_usage` (capacity string + active context list), `disk_storage` (list of persisted findings), `async_jobs` (job IDs with ETAs and statuses), `kernel_directives` (base + agent-injected rules), and `curriculum` (PLR buffer state: enabled flag, buffer_size, mean_regret, max_regret, difficulty gauge, diversity score). This payload is nested inside `observation.metadata.agui_state` on every `/step` response.

The frontend renders a **5th panel** (`CurriculumPanel.tsx`) below the terminal when `curriculum.enabled === true`, showing regret bars, difficulty gauges, and buffer coverage.

**Frontend replay:** `demo_eval.py` captures per-step AGUI snapshots as `step_NNN.json` files in `demo_output/`. The Next.js frontend reads these sequentially to replay the full investigation as an animated OS simulation — judges see the agent's RAM filling, data paging to disk, async jobs completing, and kernel directives accumulating in real time.

---

## L4 Training Pipeline & VRAM Optimization

Training an 8B-parameter language model on consumer GPU hardware (NVIDIA L4, ~24GB VRAM) demands aggressive memory engineering. Our `train_ppo.py` implements a custom step-level PPO loop with three architectural decisions that make this possible:

### 1. Unsloth 4-bit Quantization + LoRA

The base model (Meta-Llama-3.1-8B-Instruct) is loaded via Unsloth's `FastLanguageModel` in 4-bit NF4 quantization (~5GB VRAM). Only the LoRA adapter layers (rank 16, alpha 32) are trainable — roughly 0.5% of total parameters. This drops the trainable parameter footprint to ~40MB.

### 2. The `disable_adapter()` Trick

Standard PPO requires a frozen reference model to compute the KL divergence penalty. Naively, this means loading two copies of the model (~10GB each) — impossible on an L4.

Our solution: **one model, two modes.** During the forward pass, we call `model.disable_adapter()` to temporarily bypass the LoRA layers, producing reference logits from the frozen base weights. Then `model.enable_adapter()` restores the trainable policy. This yields an exact KL penalty with zero additional VRAM overhead.

### 3. Step-Level Processing

Rather than batching entire episodes (which would require storing all token sequences simultaneously), we process **one environment step at a time**. Each step: tokenize → forward → compute log-prob → accumulate into a trajectory buffer. Gradients are accumulated over 4 steps before a single optimizer update. Peak VRAM: **~10GB / 15GB** — comfortable headroom for the L4's memory.

**Colab/Kaggle deployment:** `TRAINING.md` provides 6 copy-paste cells: install stack → upload project → dry-run → train → evaluate → demo. Training 50 iterations on an L4 takes ~2.5 hours.

### 4. PPO Stability Engineering

The training loop includes production-grade safety features to prevent policy collapse:

- **Mean per-token KL:** Log-probabilities use `.mean()` (not `.sum()`), making the KL divergence penalty scale-invariant regardless of response length. This is critical — `.sum()` causes KL to explode proportionally to token count, dwarfing environmental rewards and causing learned helplessness.
- **Entropy bonus:** A `- entropy_coef × H(π)` term in the PPO loss prevents the policy from collapsing to a single deterministic action. Set to 0.05 (5x stronger than initial value) after observing entropy death within 4 iterations.
- **Ratio clamping:** `log_ratio` is clamped to `[-10, 10]` before `exp()` to prevent inf/NaN from policy drift between PPO epochs.
- **Return clipping:** Discounted returns are clipped to `[-2.0, +2.0]` to bound gradient signals from outlier episodes.
- **Empty response guard:** If the model generates 0 tokens, a dummy EOS token is substituted to prevent NaN from `.mean()` on an empty tensor.
- **Fault-tolerant `env.step()`:** Each environment step is wrapped in try/except. Malformed actions that crash the handler receive a -0.10 penalty instead of terminating the entire episode.
- **Degenerate response detection:** If a model response has >80% repeated tokens (e.g., "Search search search..."), the episode is terminated early with a -0.15 penalty.
- **KL early stopping:** If mean KL exceeds ±15 during a PPO epoch, remaining epochs are skipped to prevent catastrophic gradient updates.
- **Type-safe `parse_action()`:** All JSON parsing forces `params` to dict type, catches `TypeError`/`ValueError`, and never crashes regardless of model output.
- **Auto-Revert ("Time Machine"):** Entropy heartbeat monitor detects collapse (entropy <0.01 ×2 iters, score ≤0 ×2 iters, |KL|>10) and auto-reverts to last stable checkpoint with bumped hyperparameters (entropy_coef ×1.5, temperature +0.1, lr ×0.7). Old optimizer state is explicitly freed before rebuild to prevent VRAM doubling. Max 5 reverts per run. On ZeRO-3, stable checkpoints use `GatheredParameters` to materialize full tensors from shards.

### 5. DPO Continuous Learning Pipeline

A human-in-the-loop feedback system for post-deployment improvement:

- **Frontend:** Prisma-backed SQLite database (WAL mode for concurrent access) stores `PreferencePair` records (original prompt + rejected/chosen responses). Next.js API route (`/api/preferences`) exposes POST/GET/PATCH endpoints.
- **DPO Trainer (`train_dpo.py`):** Pulls unconsumed pairs, runs DPO loss (`-log σ(β × Δ)`) against frozen base using the same `disable_adapter()` trick, with `clip_grad_norm_(1.0)` to prevent explosions from adversarial pairs. Saves updated LoRA adapters.
- **Hot-Swap (`hotswap.py`):** Reloads updated adapters into a live model with zero downtime. Dual-method: PEFT `load_adapter()` → manual state dict injection fallback.

---

## Pitch Strategy: Gym vs. Stage

Memex uses a **dual-data architecture** to satisfy two orthogonal goals: RL generalization and judge persuasion.

### The Gym — `procedural_generator.py`

The training environment. Every call to `env.reset()` constructs a mathematically fresh POMDP graph:
- Entity IDs are randomly generated (e.g., `CUST8X3F`, `ENT_A72`) — no static labels to memorize
- Typology (structuring / layering / trade-based ML) is either specified or randomly selected
- Difficulty scales noise: easy = 1 decoy, hard = 5+ decoys with deep fan-out networks
- 9 unique combinations (3 typologies × 3 difficulties) ensure broad coverage

This forces the PPO agent to learn **transferable OS mechanics** — when to page data to disk, when to fire async traces, when to inject compliance rules — rather than memorizing "if entity X, then file SAR."

### The Stage — `demo_eval.py`

The presentation environment. A hardcoded, high-impact scenario inspired by the **1MDB sovereign wealth fund scandal**:
- 5 named entities: Taek Jho Lowe (PEP), PetraStar Energy Fund, Golden Star Holdings (BVI shell), Arabian Blue Consulting (Seychelles), and a legitimate decoy (Chen Wei Trading)
- 8 transactions totaling $681M in layered wire transfers + a $6M cover-up reversal
- Ground truth: `file_sar`, typology `layering`, 6 key findings (PEP connection, offshore source, shared registered address, rapid fan-out, no source documentation, reversed transaction)

The demo runs in two modes:
1. **Scripted** (`--dry-run`): 15 hardcoded investigation steps, no GPU needed. Produces a perfect +1.01 score for deterministic stage presentations.
2. **LLM-driven** (`--model checkpoints/best`): The trained agent investigates the 1MDB case autonomously, proving that procedural training transfers to real-world scenarios.

Both modes capture AGUI state for frontend replay, giving judges a cinematic walkthrough of the agent managing its own operating system while solving a $681M money laundering case. This includes dynamic 3D threat mapping via `react-globe.gl` and smooth incremental entity graph updates to ensure a highly persuasive visual presentation.

---

## 70B Distributed Training Pipeline (A100 Cluster)

The scaling pivot from 8B/L4 to 70B/multi-A100 is implemented in `train_ppo_70b.py`.

### Why DeepSpeed ZeRO-3?

| Component | 70B Memory | ZeRO-3 (4× A100-80GB) |
|-----------|-----------|------------------------|
| Model weights (4-bit NF4) | ~35 GB | Sharded: ~9 GB/GPU |
| LoRA adapters (r=32, α=64) | ~150 MB | Replicated on each GPU |
| Optimizer states (fp32 AdamW) | ~280 GB | Sharded: ~70 GB/GPU |
| Gradients (bf16) | ~140 GB | Sharded: ~35 GB/GPU |
| **Peak per-GPU** | — | **~50 GB / 80 GB** |

ZeRO-3 shards **all three** (parameters + optimizer + gradients) across GPUs, making 70B feasible on 4× A100-80GB with comfortable headroom.

### Architectural Decisions

1. **Rank-0 Environment Interaction:** Only rank 0 runs the AML environment. Trajectories (tokenized prompts, responses, rewards) are broadcast to all ranks via `torch.distributed.broadcast_object_list()`. This avoids N redundant environment instances.

2. **`disable_adapter()` Under ZeRO-3:** The KL trick works because LoRA operates at the module level — when ZeRO-3 gathers parameters for a forward pass, disabling adapters simply means the gathered weights skip LoRA additions. No architectural conflict.

3. **DeepSpeed Engine Backward:** Replaces manual `loss.backward()` + `optimizer.step()` with `engine.backward(loss)` + `engine.step()`. DeepSpeed handles gradient accumulation, all-reduce, and gradient clipping internally.

4. **Distributed Checkpointing:** Uses `engine.save_checkpoint()` which saves sharded optimizer states per-rank. Only rank 0 saves the tokenizer.

### Launch Commands

```bash
# 4-GPU single-node
deepspeed --num_gpus 4 train_ppo_70b.py --iterations 50 --episodes 2

# 8-GPU multi-node (2 nodes × 4 GPUs)
deepspeed --num_gpus 4 --num_nodes 2 --hostfile hosts.txt \
  train_ppo_70b.py --iterations 50 --episodes 2

# With CPU offloading (lower VRAM, slower)
deepspeed --num_gpus 2 train_ppo_70b.py --offload-optimizer --offload-params

# Dry-run (2 iterations, 1 episode)
deepspeed --num_gpus 4 train_ppo_70b.py --dry-run
```

---

## Dependencies

- **Runtime:** Python 3.10+, FastAPI, Pydantic v2, httpx, openai
- **Training (optional):** unsloth, trl, peft, torch, wandb
- **Validation:** openenv-core, Docker

---

## Deployment

- **HF Spaces:** `openenv push --ignore-file .hfignore` → deploys to `MuazTPM/aml_investigation_env` (Docker, port 7860)
- **OpenEnv CLI:** `openenv serve` → reads `openenv.yaml` → `openenv_server:app`
- **Local:** `uvicorn openenv_server:app --host 0.0.0.0 --port 8000`
- **Docker:** `docker build -t memex . && docker run -p 7860:7860 memex`
- **Smoke Tests:** `python tests/test_smoke.py` → 8/8 tests

---

## Recent Changes

### 2026-04-23

1. **OpenEnv Server (`openenv_server.py`):** Production-grade FastAPI wrapper using OpenEnv SDK `create_app()` with standalone fallback. Boot-time validation, structured logging, HF Spaces ready (port 7860).
2. **AMLEnvironment fix:** Added `super().__init__()` to `AMLEnvironment.__init__()` for OpenEnv ABC compatibility.
3. **Dockerfile rewrite:** HF Spaces compliant — non-root user (UID 1000), port 7860, `HEALTHCHECK`, `openenv_server:app` entrypoint.
4. **`.hfignore`:** Dedicated exclude file for `openenv push` — excludes `frontend/`, `venv/`, binary files. Reduced upload from 558MB to ~5MB.
5. **HF Space deployed:** `MuazTPM/aml_investigation_env` live at https://huggingface.co/spaces/MuazTPM/aml_investigation_env
6. **Frontend Overhaul:** Transitioned AGUI to a 2-column layout (Entity Graph on left, State/Terminal stacked on right). Rebranded Nexus to Memex with a new smoky gray (`#131316`) color palette.
7. **3D Physics & Visuals:** Integrated `react-globe.gl` for dynamic 3D threat mapping. Optimized Cytoscape `cola` physics with incremental add/remove logic for stable, non-destructive graph animations during replay.
8. **Adversarial Training Loop:** Added an LLM-backed Adversary Agent (`adversary_agent.py`) and Battle Orchestrator (`train_adversary.py`) to generate evasive AML scenarios. Failed defender attempts are saved to an SQLite database (`adversarial_successes.db`) as negative examples for the DPO pipeline.
9. **Reward Integrity:** Implemented hard caps to prevent PPO reward farming: maximum 3 rewarded disk writes (+0.10) and 2 rewarded kernel injections (+0.15) per episode. Subsequent calls still incur the standard action penalty.

### 2026-04-22

1. **PPO stability overhaul (10 fixes):** Fixed catastrophic policy collapse caused by `.sum()` KL divergence. Added entropy bonus, ratio clamping, return clipping, and empty response guards to both `train_ppo.py` and `train_ppo_70b.py`.
2. **Production hardening:** Both trainers now handle edge cases (zero-token responses, ratio overflow, gradient NaN) that would corrupt weights in long-horizon training runs.
3. **WandB monitoring:** Added `ppo/entropy` metric for real-time mode collapse detection.

### 2026-04-20

1. **Codebase sanitization:** Removed duplicate type definitions from `app.py`, updated `client.py` with all 15 tools, rewrote README to reflect Memex OS-Agent architecture
2. **Git consolidation:** Resolved stuck rebase, recovered all OS-mechanic features to `main`
3. **PPO trainer:** Custom step-level PPO with Unsloth 4-bit + LoRA, L4-optimized (peak ~10GB VRAM)
4. **70B scaling pivot:** `train_ppo_70b.py` with DeepSpeed ZeRO-3 for multi-node A100 cluster
5. **Demo system:** 1MDB-inspired scenario with AGUI replay capture for frontend visualization
