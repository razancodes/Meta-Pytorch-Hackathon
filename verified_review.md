# Verified Hackathon Audit: Memex AML OS-Agent Benchmark

> **Reviewer**: Senior Engineer, Meta/OpenEnv Evaluation Panel  
> **Date**: 2026-04-25  
> **Scope**: Full repository audit + critique of prior Gemini 3.1 Pro review  
> **Verdict**: This is a top-tier submission with surgical gaps that must be closed before judging.

---

## Preamble: Analysis of the Prior Review

The Gemini 3.1 Pro review identified four weaknesses. I'll grade each claim against what actually exists in the codebase.

### Claim 1: "Missing the Most Crucial Proof — Reward Curves" — ✅ VALID, but understated

The prior review correctly identifies the absence of embedded reward plots in the README. However, it underestimates the severity. The rubric says:

> *"Save plots as .png or .jpg and commit them to the repo (don't leave them only in a Colab cell or a deleted Wandb run)"*

This is **the single highest-risk gap** in the submission. The environment is phenomenal. The training infrastructure is battle-hardened. But if a judge opens the README and sees zero plots, the 20% "Showing Improvement" category scores close to zero. This alone can drop you from 1st to 5th.

**However**, the prior review missed something important: your `TRAINING.md` already has a WandB monitoring table (lines 347-357) that defines *exactly* the metrics to plot:

| Metric | Healthy Range | What to Watch |
|--------|---------------|---------------|
| `ppo/returns/mean` | 0.0 → +0.8 | Main signal |
| `os/page_faults` | Decreasing → 0 | Memory management |
| `os/async_timeouts` | Decreasing → 0 | Interrupt handling |
| `os/successful_pages` | Increasing | Disk writes |
| `os/meta_injections` | ≥ 1 per episode | Kernel updates |

This table is *exactly* what the judges want to see as chart axes. The instrumentation exists — you just haven't run it and committed the outputs.

> [!CAUTION]
> **Priority 0 — Must fix before submission**: Run training, export WandB plots, commit as `.png` files, embed in README. Without this, the 20% category is effectively zero.

---

### Claim 2: "Over-Indexing on Training Infrastructure vs. Environment Innovation" — ⚠️ PARTIALLY VALID, but the conclusion is wrong

The prior review claims the repo "feels like a masterclass in Unsloth, LoRA, ZeRO-3 rather than an environment showcase." I disagree with the framing.

**What the prior review missed:** The self-play pipeline (`self_play.py`, `train_defender_ppo.py`, `train_launderer_ppo.py`, `server/launderer_env.py`) is not just training infrastructure — it IS environment innovation. The Launderer is a second agent that operates *inside* a one-step MDP (`LaundererEnv`) to generate evasive scenarios. The Defender operates inside the main 18-tool POMDP. Together they form a **multi-agent environment** (Theme #1) with the Launderer dynamically expanding the environment's state space. Very few hackathon submissions will have an adversarial scenario generator that is itself trained via PPO.

**Where the criticism lands:** The README and `PROJECT_CONTEXT.md` DO bury the OS mechanics under training details. The `disable_adapter()` trick, the ZeRO-3 VRAM budget, and the auto-revert heartbeat — these are impressive engineering, but they're *training concerns*, not *environment concerns*. The judges grade the environment (40%), not the optimizer.

**My revised assessment:** The environment itself (the OS-mechanic POMDP) is genuinely novel and publishable. The training pipeline (self-play PPO with VRAM-safe model swapping) is a strong differentiator for the 10% "Reward & Training Pipeline" category. But the README needs to lead with the environment, not the trainer.

---

### Claim 3: "Domain Complexity & Cognitive Overload" — ❌ MOSTLY WRONG

The prior review warns that AML jargon will confuse judges. I strongly disagree.

The rubric says: *"Pick an ambitious, original problem... Is the domain underexplored in RL/LLM training?"*

AML investigation is **exactly** the kind of underexplored professional domain the judges want. FinCEN SAR filing, UBO tracing, mule-ring detection — these are real enterprise workflows that no other hackathon team will attempt. The judges are from Meta and HuggingFace; they are ML researchers, not financial regulators. But they are *looking* for submissions that go beyond chess and grid-worlds. The domain complexity is a **strength**, not a weakness.

**Where there IS a real risk:** The 18 tools, 3 typologies × 3 difficulties, Phase 3 FinCEN tools, PLR curriculum, DPO pipeline, adversarial self-play, AGUI frontend, 70B scaling — the sheer *breadth* of the system can make a reviewer feel like they can't evaluate it in the time they have. The prior review conflated "domain complexity" (good) with "system sprawl" (risky). The fix is narrative focus, not domain simplification.

---

### Claim 4: "The Colab Script is Not Minimal" — ✅ VALID, and this is now worse than the prior review realized

The prior review referenced `colab_selfplay_train.py` (740 lines). But since then, the project has evolved to have **multiple training entry points**:

| Script | Lines | Purpose |
|--------|-------|---------|
| `train_ppo.py` | ~1,100 | Standalone PPO |
| `train_defender_ppo.py` | 994 | Defender PPO with GAE |
| `train_launderer_ppo.py` | 763 | Launderer PPO |
| `train_grpo.py` | ~800 | GRPO (experimental) |
| `train_ppo_70b.py` | ~1,300 | 70B DeepSpeed |
| `train_dpo.py` | ~400 | DPO offline |
| `self_play.py` | 341 | Orchestrator |
| `colab_selfplay_train.py` | 742 | Colab notebook-style |

A judge opening this repo sees **8 training scripts**. The rubric asks for "a working training script using Unsloth or HF TRL, ideally as a Colab notebook." The answer should be ONE obvious file.

`TRAINING.md` does a good job of defining Cell 1–8 copy-paste blocks. But the prior review's recommendation of creating a `judge_quick_train.ipynb` is correct — you need a single notebook that a judge can open in Colab and click "Run All."

---

## Part 2: What the Prior Review Completely Missed

The Gemini review was surface-level. Here are critical issues it failed to identify.

### 🔴 Critical: The `aml_environment.py` Docstring is Wrong

Line 7-14 of [aml_environment.py](file:///c:/Users/MRaza/Documents/AML-Meta-Pytorch/server/aml_environment.py#L7-L14):
```python
"""
Tool Roster (15 tools):
  Legacy domain tools (9): ...
  OS-Mechanic tools (6): ...
"""
```
The actual `AVAILABLE_TOOLS` list on lines 48-70 has **18 tools**. The docstring says 15. This is a code-level inconsistency that a thorough judge will notice. The README says 18, the `PROJECT_CONTEXT.md` says 18, but the source file's own docstring says 15. Fix this.

### 🔴 Critical: The `README.md` Reward Table Doesn't Match `grader.py`

The README (lines 216-226) describes the terminal reward as:

| Component | Weight |
|-----------|--------|
| Decision | 0.30 |
| Typology | 0.15 |
| Findings | 0.20 |
| Entities | 0.15 |
| UBO ID | 0.05 |
| Pillar | 0.05 |
| Efficiency | 0.10 |

But `grader.py` (lines 52-68) uses a completely different weight scheme:

```python
class RewardWeights:
    detect: float = 1.0       # NOT 0.30
    entity_f1: float = 0.5    # NOT 0.15
    typology: float = 0.3     # NOT 0.15
    efficiency: float = 0.2   # NOT 0.10
    os: float = 0.2           # NOT mentioned in README table
```

The README presents a normalized [0,1] weight table that sums to 1.0. The actual code uses unnormalized weights that sum to 2.2. These are fundamentally different reward structures. A judge reading the README then looking at `grader.py` will conclude the documentation is inaccurate.

> [!WARNING]
> **Priority 1**: Either update the README to match the actual code weights, or add a note explaining the normalization. The current mismatch is a credibility risk under the 10% "Reward & Training Pipeline" criterion.

### 🟡 Important: GAE Without a Value Function is Degenerate

In `train_defender_ppo.py` lines 477-507, the GAE implementation sets V(s) ≈ 0 for all states:

```python
# Since we don't have a value function, approximate V(s) ≈ 0
# This gives δ_t = r_t (clipped)
```

With V(s) = 0, GAE reduces to exponentially-weighted sums of raw rewards. This is better than pure Monte Carlo returns (which the prior `train_ppo.py` used), but it's not true GAE — it's a discount-weighted reward accumulator. This is fine for a hackathon, but if a judge asks "how does your GAE baseline work?", the answer "we don't have a baseline" reveals that the advantage estimates have high variance. Not a blocker, but be prepared to defend it.

### 🟡 Important: The `KERNEL_MODES` Finite Set is a Design Strength — Highlight It

The prior review missed this entirely. In `state_manager.py` lines 235-242:

```python
KERNEL_MODES: set = {
    "enhanced_due_diligence",
    "structuring_detection",
    "trade_based_ml_detection",
    "sanctions_screening",
    "mule_ring_detection",
    "high_risk_jurisdiction",
}
```

This is a **closed set** of valid kernel modes. The agent cannot inject arbitrary text into the system prompt — it must choose from these 6 modes. This is a deliberate design decision to prevent prompt injection as a reward hack. This is exactly the kind of "hard to game" reward design the rubric explicitly asks for. You should call this out in the README.

### 🟡 Important: The Terminal Reward Double-Counting Fix is Critical and Undocumented

In `train_defender_ppo.py` lines 403-408:

```python
# P1-3 FIX: The terminal obs.reward is the grader's composite total
# which already includes all accumulated per-step micro-rewards.
# Subtract the sum of prior step rewards so GAE counts each reward
# unit exactly once (otherwise step rewards are counted 2x).
prior_step_reward_sum = sum(s.reward for s in steps[:-1])
steps[-1].reward = reward - prior_step_reward_sum
```

This is a **crucial correctness fix** that prevents the PPO trainer from double-counting step rewards in the terminal signal. Without this, the agent would receive inflated returns on episodes where it uses many OS mechanics (each step reward counted once in the trajectory AND once in the terminal composite). This fix should be mentioned in `TRAINING.md` as a stability engineering feature — it demonstrates deep understanding of the reward pipeline.

### 🟢 Strength Not Highlighted: The Launderer Validation Gate

The `launderer_env.py` `validate_scenario()` function (lines 161-213) implements a 9-check validation gate:

1. Top-level keys present
2. Ground truth structure
3. `is_suspicious` must be True
4. `correct_decision` must be `file_sar`
5. Typology must be valid (uses `TypologyEnum`)
6. Non-empty entities
7. Non-empty findings
8. Alert must have `alert_id`
9. At least one transaction and customer profile

This prevents the Launderer from gaming the reward by generating trivial or malformed scenarios. Combined with the shaped reward (-2.0 for parse fail, -1.0 for schema fail, +0.1 to +1.0 for valid scenarios), this creates a well-designed curriculum where the Launderer must first learn to produce valid JSON, then learn to make it evasive. This is excellent environment design that should be showcased.

---

## Part 3: Final Verified Rubric-Aligned Assessment

### Environment Innovation (40%) — Score: 9/10

**Strengths:**
- The OS-mechanic metaphor (RAM eviction, async interrupts, kernel updates) is genuinely novel. No other OpenEnv submission will have this.
- 18 tools with meaningful semantic differences (not 18 variations of "move left")
- Procedural scenario generation with anti-memorization guarantees (unique IDs per episode)
- Multi-agent self-play where the adversary itself is a trainable LLM
- Closed kernel mode set prevents prompt injection — demonstrates security-aware environment design
- Reward farming hard caps (3 disk writes, 2 kernel injections) prevent degenerate policies
- 3 typologies × 3 difficulties × procedural generation = effectively infinite training scenarios

**Weakness:**
- The environment is complex enough that a 3-minute README scan may not convey the innovation. The README leads with "The Problem" (good) but then immediately shows an architecture diagram with 4 boxes (training infrastructure detail that should come later).

### Storytelling & Presentation (30%) — Score: 6/10

**Strengths:**
- The "Can an LLM run its own Operating System?" tagline is compelling
- The 1MDB demo scenario is a brilliant narrative anchor
- The AGUI frontend (5-panel tactical dashboard) makes the abstract OS mechanics visible
- The `PROJECT_CONTEXT.md` is one of the most thorough hackathon docs I've ever seen

**Weaknesses:**
- No video or blog post linked from README (minimum requirement)
- No embedded reward plots (minimum requirement)
- README has 310 lines — too long for judges. The core story (OS mechanics) gets diluted by training details
- The README reward table doesn't match the code (credibility risk)
- The `aml_environment.py` docstring says 15 tools when there are 18

### Showing Improvement in Rewards (20%) — Score: 1/10

**This is the crisis category.**

- No reward curve images in the repo
- No before/after comparison
- No WandB run links
- The training scripts log all the right metrics (`os/page_faults`, `os/successful_pages`, `defender/entity_f1`), but there is zero evidence that training was ever run to completion
- The `colab_selfplay_train.py` has `USE_WANDB = False` by default

> [!CAUTION]
> **This category alone can sink an otherwise brilliant submission.** You must run training and commit the evidence before the deadline.

### Reward & Training Pipeline (10%) — Score: 8/10

**Strengths:**
- Dense per-step rewards (not just 0/1 at the end) — exactly what the rubric asks for
- Anti-gaming: hard caps, closed kernel modes, unique IDs
- The `AMLGrader` docstring explicitly proves lazy policies score lower: `E[R_always_SAR] = 0.475 < E[R_reasonable] ≈ 0.68`
- 10 stability features documented in TRAINING.md (auto-revert, entropy heartbeat, ratio clamping)
- The terminal reward double-counting fix (P1-3 FIX) shows deep RL engineering awareness
- Shaped Launderer rewards with 3-tier penalty structure

**Weaknesses:**
- README reward weights don't match `grader.py` weights
- GAE baseline is V(s)=0 (acknowledged in code comments, but still a limitation)

---

## Part 4: Exact Action Items (Priority-Ordered)

### P0 — Must Do Before Submission (Affects Minimum Requirements)

| # | Action | Category | Est. Time |
|---|--------|----------|-----------|
| 1 | **Run PPO training** (even 10 iterations of `train_defender_ppo.py` with `--dry-run` off, WandB on). Export reward curves as `.png`. Commit to `plots/` directory. | 20% Showing Improvement | 2-3 hours |
| 2 | **Embed plots in README.md** under a new `## Training Results` section. Include `reward/mean`, `os/page_faults` (decreasing), `os/successful_pages` (increasing). Add one-line captions. | 20% Showing Improvement | 15 min |
| 3 | **Record a <2 min video** or write a mini-blog on HuggingFace. Link from README. | Minimum Requirement | 30 min |
| 4 | **Verify HF Space URL** is at the top of README and is live. Currently listed at line 293: `https://huggingface.co/spaces/MuazTPM/aml_investigation_env` | Minimum Requirement | 5 min |

### P1 — Should Do (Affects Scoring)

| # | Action | Category | Est. Time |
|---|--------|----------|-----------|
| 5 | **Fix README reward table** to match `grader.py` actual weights (detect=1.0, entity_f1=0.5, typology=0.3, efficiency=0.2, os=0.2). | 10% Pipeline | 10 min |
| 6 | **Fix `aml_environment.py` docstring** — change "15 tools" to "18 tools". | Code Quality | 2 min |
| 7 | **Create `judge_quick_train.ipynb`** — a minimal Colab notebook (Cells 1-5 from TRAINING.md, hardcoded to 5 iterations, WandB on, one inline `matplotlib` plot). | 10% Pipeline | 45 min |
| 8 | **Add a "Before vs. After" section** to README showing qualitative behavior change (table of untrained vs. trained agent behavior on OS mechanics). | 20% Showing Improvement | 15 min |

### P2 — Nice to Have (Polish)

| # | Action | Category | Est. Time |
|---|--------|----------|-----------|
| 9 | **Highlight the closed `KERNEL_MODES` set** in README as an anti-gaming feature. | 40% Environment | 5 min |
| 10 | **Mention the P1-3 terminal reward fix** in TRAINING.md stability features table. | 10% Pipeline | 5 min |
| 11 | **Shorten README** to ~200 lines. Move VRAM budgets, 70B details, and DPO pipeline to TRAINING.md. Keep README focused on: Problem → Environment → Reward → Results → Quick Start. | 30% Storytelling | 30 min |
| 12 | **Add the Launderer validation gate** to README as a subsection under Reward System. Show the 9 checks. | 40% Environment | 10 min |

---

## Final Verdict

This is one of the strongest OpenEnv submissions I've reviewed. The OS-mechanic POMDP is genuinely novel — it creates a transferable set of skills (memory management, async handling, self-improvement) that no other AML or grid-world benchmark tests. The self-play pipeline with a trainable adversarial Launderer puts this in multi-agent territory (Theme #1) while the 25-step investigation horizon firmly covers long-horizon planning (Theme #2). The AML domain with FinCEN 4-pillar tools is real professional world modeling (Theme #3.1).

The engineering quality is exceptional. The 14+ stability fixes in the PPO trainer, the VRAM-safe model swapping, the reward farming hard caps, and the closed kernel mode set all demonstrate a team that understands RL failure modes at a production level.

**The submission will live or die on one thing: whether reward plots are committed to the repo before the deadline.** Everything else is polish. Fix P0 items 1-4, and this is a contender for first place.
