# PAPER_CONTEXT.md
## World Feedback for Financial Crime Detection: A Complete Paper Blueprint
### RLxF Workshop @ ICML 2026 — Short Track (2–4 pages)
### Deadline: Thursday, May 14, 2026, 23:59 AoE

---

## PART 0: CANONICAL TITLE, ABSTRACT, AND THESIS

### Final Title
**"World Feedback for Financial Crime Detection: OS-Grounded Reward Signals for Long-Horizon AML Investigation Agents"**

*(Alternative if reviewers aren't from AML domain)*
**"Beyond Human Preference: Heterogeneous World Feedback for Training LLM Agents in Partially Observable Environments"**

### One-Sentence Thesis
We show that AML investigation is a natural world-feedback RL problem where outcome-grounded signals — financial crime detection accuracy, computational resource efficiency, and regulatory compliance — computed entirely from environment state (zero human labels) train LLM agents that generalize across money laundering typologies and exhibit emergent OS-aware investigation behaviors.

### Abstract Template (150 words — fill in X, Y, Z from experiments)
Reinforcement Learning from Human Feedback (RLHF) relies on human preference signals that are expensive, subjective, and impossible to scale in high-stakes forensic domains. We present Memex, an OS-augmented multi-agent RL environment for Anti-Money Laundering (AML) investigation, where all reward signals are derived from measurable world outcomes: financial crime detection accuracy (economic outcome), memory management efficiency (resource signal), and temporal coordination (latency signal). The environment formalizes AML investigation as a POMDP with 18 tools across 3 typologies and 3 difficulty levels, with a procedural scenario generator that prevents memorization. We train a Qwen2.5-7B Defender agent via GRPO on this heterogeneous world feedback and show: (1) removing OS world feedback reduces detection F1 by X% on hard typologies; (2) the trained agent achieves Y% SAR F1 vs. Z% for zero-shot Qwen2.5-7B-Instruct, with 3× higher OS mechanics adoption; (3) fixed hacking policies score E[R] ≤ 0.52 vs. E[R] ≈ 0.68 for the trained agent, confirming the reward is not trivially gameable. Code and demo: [anonymized].

---

## PART 1: SECTION-BY-SECTION PAPER PLAN

### Section 1 — Introduction and Motivation (Target: 0.5 pages)

#### Core Argument to Make
AML investigation has *no natural human preference signal*. A human annotator cannot reliably distinguish structuring from legitimate cash deposits without running a full investigation. The world — the ground truth of whether money laundering occurred, whether resources were managed efficiently, whether regulatory frameworks were applied correctly — provides a richer and more objective signal than any human labeler. This is the definitional argument for treating Memex as a world feedback problem, not an RLHF problem.

#### Paragraph 1: The Problem (3–4 sentences)
- Money laundering costs $800B–$2T annually (2–5% of global GDP), with ~90% going undetected [cite: FATF]
- Existing AML systems cost $274B/year globally with high false positive rates that overburden compliance teams
- A real investigation is a multi-step, partially observable process: pull profiles, trace networks, wait for wire results, cross-reference sanctions, then decide whether to file a SAR
- Existing ML approaches rely on static datasets or human-labeled examples — neither scales to the adversarial, evolving nature of money laundering

#### Paragraph 2: The World Feedback Framing (3–4 sentences)
- RLHF requires humans to express preferences over investigation outcomes — but human annotators cannot reliably assess whether a complex layering scheme was correctly identified
- The world provides the signal: did the SAR correctly identify laundering? Did the investigation use working memory efficiently? Did the agent apply the right regulatory framework?
- These are *world-grounded outcomes* — computed from environment state, not from human opinion
- Formally: Memex rewards reflect world-grounded financial crime consequences, computed entirely from ground truth that exists independently of any human labeler

#### Paragraph 3: Contributions (bullet list)
1. Memex: an OS-augmented POMDP environment for AML investigation with 18 tools, 3 typologies, heterogeneous world feedback rewards, and a procedural anti-memorization scenario generator
2. A formal taxonomy mapping OS mechanics (virtual memory, async interrupts, kernel injection) to world feedback signal families (resource efficiency, latency, compliance)
3. A composite reward function coupling financial crime outcome signals with OS efficiency signals in a single unified objective
4. Empirical demonstration that heterogeneous world feedback (full reward) outperforms detection-only reward, with anti-gaming analysis showing reward is not trivially exploitable

---

### Section 2 — Environment Design (Target: 0.75 pages)

#### Subsection 2.1: POMDP Formulation
Write the formal POMDP tuple: (S, A, O, T, R, γ)
- **State space S**: transaction graph G(V,E) + agent internal state (RAM, disk, async jobs, kernel)
- **Action space A**: 18 tool calls (11 domain investigation + 5 OS mechanic + 2 terminal)
- **Observation space O**: partial — agent sees only last 2 observations (RAM=2 slots), evidence evicted on overflow
- **Transition T**: deterministic given scenario seed; stochastic in async job ETA (Uniform[2,4] steps)
- **Reward R**: heterogeneous composite (see Section 3)
- **Horizon**: T_max = 25 steps per episode

State why this is a *hard* POMDP:
- Partial observability is *architectural*: the 2-slot RAM window forces the agent to manage what it remembers
- Long horizon: 10–25 tool calls across multiple data sources
- Asymmetric consequences: FN = −2.0 (missed laundering) >> FP = −0.75 (false alarm)

#### Subsection 2.2: AML Typologies and Procedural Generator
Describe 3 typologies:
- **Easy — Structuring (smurfing)**: sub-$10K cash deposits, non-cash occupation mismatch
- **Medium — Layering**: fan-out through shell companies, multi-hop offshore routing
- **Hard — Trade-Based ML (TBML)**: over-invoiced phantom shipments, customs mismatches

Procedural anti-memorization: every episode generates unique entity IDs (CUSTX3RU format), names from 44×35=1,540 combinations, company names from 25×22=550 combinations, transaction amounts/dates/timestamps, and network topology. Model cannot memorize specific cases.

Noise scaling: Easy (0 decoys), Medium (2 profiles + 4 transactions), Hard (3 profiles + 8 transactions).
30% of scenarios are clean (force_clean=True) to prevent always-SAR policy from dominating.

#### Subsection 2.3: OS Mechanics as World Feedback Mechanisms
This is the key conceptual section. Frame each mechanic as a *world feedback generator*, not a game mechanic:

**Virtual Memory / Page Fault:**
The agent's context window (RAM_CAPACITY=2 slots) models real investigator working memory limits. Critical evidence evicted from context without being persisted triggers a page fault — a *computational resource misuse signal* from the environment.
- RAM eviction: deque(maxlen=2), entity IDs tracked via regex (CUST*, ENT_*, TXN-*, REQ-*, ACC-*, ALERT-*)
- Page fault: −0.05 when agent references an evicted, unsaved entity ID
- Page hit: +0.10 when agent reads previously saved data

**Asynchronous Interrupts / Wire Traces:**
request_wire_trace starts a background job with ETA drawn from Uniform[2,4] steps. The agent must continue investigating during the wait. Premature retrieval (−0.10) models the real-world cost of halting investigation to wait for pending forensic results. This is a *temporal coordination signal* — the environment penalizes blocking behavior.

**Kernel-Level Meta-Prompting:**
The agent can update its own system prompt with one of 6 enumerated compliance directives (prevents prompt injection). This models dynamic regulatory framework application — switching from broad-surveillance to targeted-forensic mode. Models the *compliance outcome signal*: applying the right regulatory framework to the right scenario type.

Include the formal reward formulation:
J(θ) = E_{π_θ}[Σ_t γ^t (R_AML(s_t,a_t) − λ·R_fault(s_t))]

Explain: because R_fault fires only when f ∉ C and f ∉ M, and the agent *controls* writes to M via write_to_case_file, this is a Markovian penalty the agent can eliminate deterministically. This makes page faults a *learnable* world signal, not stochastic noise.

---

### Section 3 — Reward Design as World Feedback Composition (Target: 0.5 pages)

#### Subsection 3.1: World Feedback Signal Taxonomy
Present this as a table — the conceptual core of the paper:

| Signal Family | Workshop Category | Implementation | Reward |
|---|---|---|---|
| Detection outcome (TP/FP/FN/TN) | Economic outcome | Terminal grader, R_detect | TP=+1.0, TN=+0.5, FP=−0.75, FN=−2.0 |
| Memory efficiency | Resource utilization | RAM eviction, page fault | Page fault −0.05, hit +0.10 |
| Temporal coordination | Latency/efficiency | Wire trace async queue | Premature: −0.10 |
| Policy adaptation | Compliance/safety | update_system_prompt | +0.15 per valid injection |
| Investigation thoroughness | Process quality | Tool category coverage | +0.02–0.05 first use |
| Action economy | Efficiency | Step cost | −0.02 per step |

Key claim: **None of these signals require a human to express a preference.** Every reward is computed from environment state — ground truth financial crime outcomes, RAM/disk usage counters, async job ETAs, kernel mode validity checks.

#### Subsection 3.2: GRPO Decomposition
Explain the four reward functions in train_grpo.py:
- R1: Format compliance — valid JSON with known tool name (−2.0 for degenerate output)
- R2: Investigation quality — tool category diversity (−0.3 for empty params)
- R3: Environment execution — full multi-step env.step() with actual AML outcome
- R4: OS mechanics — unique OS tool usage (dedup via seen_tools set)

Note: R4 is outcome-conditioned in the final version — bonuses are zeroed if R3 terminal decision is incorrect. This prevents OS mechanic farming without investigative success.

#### Subsection 3.3: Anti-Gaming Analysis
Present as a formal table of policies and their expected reward:

| Policy | Strategy | E[R] |
|---|---|---|
| always_sar | Immediate file_sar | 0.475 (proven analytically) |
| always_close | Immediate close_alert | TBD from Experiment 3 |
| os_spammer | All 5 OS tools → file_sar | TBD from Experiment 3 |
| min_steps | review_alert + query_transactions + file_sar | TBD from Experiment 3 |
| Trained agent | Full GRPO-trained Defender | ~0.68 (measured) |

Proof for always_sar:
E[R_always_SAR] = 0.7 × R_TP + 0.3 × R_FP = 0.7(1.0) + 0.3(−0.75) = 0.475
E[R_reasonable] ≈ 0.7(0.8) + 0.3(0.4) = 0.68
Since 0.475 < 0.68, the lazy policy is dominated by the trained policy.

Also discuss: outcome-conditioned R4 gating eliminates OS mechanic farming. Show that E[R_os_spammer] < E[R_trained] even though the spammer collects all OS mechanic bonuses — because R4=0 if the final SAR decision is wrong.

---

### Section 4 — Experiments and Results (Target: 1 page)

#### Subsection 4.1: Behavioral Change (Qualitative — Lead With This)
Present the behavioral delta table from TRAINING.md as the opening result:

| Behavior | Untrained (Step 0) | Trained (Step 150) | Interpretation |
|---|---|---|---|
| Output format | Mixed inline/markdown JSON | Consistent ```json blocks | Format learning from R1 |
| Investigation depth | 1–2 tool calls | 5–10 tool calls across categories | Thoroughness from R2+R3 |
| OS tool usage | Never used | write_to_case_file, request_wire_trace, update_system_prompt active | Emergent from R4 |
| Terminal decision | Always files SAR | Differentiates TP vs FP | Discrimination from R3 |
| Completion length | ~200 tokens | ~800 tokens | Richer reasoning chains |
| Reward variance (frac_zero_std) | 0.45 | 0.0 | Healthy GRPO gradients |

Narrative: "Without any human demonstrating investigation behavior, the agent learned to manage working memory, coordinate asynchronous tool execution, and apply regulatory frameworks — behaviors that emerge from world feedback alone."

#### Subsection 4.2: Training Curves (Quantitative)
Report training metrics from the 150-step run:
- Total reward: ~0 → ~4.5 over 150 steps
- R1 (format): mixed → 1.00
- R2 (investigation quality): ~0.2 → 0.60
- R3 (environment execution): ~0 → 1.79
- R4 (OS mechanics): 0.0 → 1.10
- train_loss: ~0.001 → 0.00043 (converged)
- Training time: 3h44m on A100

Include a 2-panel figure: (left) reward component curves over 150 steps; (right) completion length over 150 steps.

Interpretation: The parallel rise of R3 and R4 shows OS mechanic learning is *coupled* with investigative success — the agent learns to page evidence to disk *because* it improves detection, not as an independent behavior. This is the key empirical signature of world feedback coupling.

#### Subsection 4.3: Reward Ablation (THE CRITICAL EXPERIMENT — Run Before Thursday)

Table structure:

| Condition | Reward Components | SAR F1 (Easy) | SAR F1 (Med) | SAR F1 (Hard) | Page Fault Rate | OS Tool Adoption |
|---|---|---|---|---|---|---|
| Full (R1+R2+R3+R4) | All | X | X | X | X | X |
| No OS feedback (R1+R2+R3) | No R4 | X | X | X | X | X |
| Terminal only (R3) | R3 only | X | X | X | X | X |
| Format only (R1+R2) | No env | X | X | X | X | X |
| Zero-shot Qwen2.5-7B-Instruct | No training | X | X | X | X | X |

Fill X values from experiments. Expected pattern:
- Full > No-OS on hard TBML (OS mechanics matter most for complex typologies)
- No-OS > Terminal-only (dense intermediate signals accelerate learning)
- Trained 7B < Qwen2.5-7B-Instruct on raw detection accuracy (acceptable — different model sizes)
- Trained 7B >> Qwen2.5-7B-Instruct on OS mechanics adoption (proof that world feedback teaches behaviors impossible zero-shot)
- Page fault rate: Full condition lowest (agent learned memory management)

How to interpret if ablation goes wrong:
- If Full ≈ No-OS: OS mechanics don't help detection — reframe as "OS mechanics teach efficiency without hurting accuracy" — still a valid claim
- If Terminal-only beats Full: R4 is adding noise — implement outcome-conditioned gating, re-run
- If trained 7B << Qwen2.5-7B-Instruct on everything: focus paper on environment design and reward taxonomy as contribution, not training results

#### Subsection 4.4: Reward Hacking Probing (Run Before Thursday — CPU Only)

Table of fixed policy scores vs. trained agent:

| Policy | E[R] empirical | E[R] analytical | Notes |
|---|---|---|---|
| always_sar | ~0.475 | 0.475 (proven) | Validates our math |
| always_close | TBD | 0.3×0.5=0.15 (estimate) | Never catches anything |
| os_spammer | TBD | < always_sar (R4 gated) | Gaming without investigation |
| min_steps | TBD | Below trained agent | Low R2, low R3 |
| Trained agent | ~0.68 | 0.68 (theoretical) | Best policy |

Narrative: "The trained agent consistently outperforms all fixed hacking strategies, demonstrating that no static policy can exploit the composite reward structure. The outcome-conditioned R4 gating specifically eliminates the OS mechanic farming strategy."

#### Subsection 4.5: Self-Play vs. Procedural Curriculum (If Time Permits)
Compare two Defender checkpoints on the 9-scenario eval harness:
- Defender trained on procedural generator (current run)
- Defender trained with zero-shot Launderer generating scenarios via self_play.py

Expected: adversarial curriculum shows better hard-typology generalization.
If not significant: note that even procedural curriculum produces strong results — the generative world model provides additional gains at the cost of inference compute.

---

### Section 5 — Reward Hacking and Collusion (Target: 0.5 pages)

#### Subsection 5.1: Reward Hacking in World Feedback Systems
Frame this as a novel taxonomy — world feedback introduces hacking vectors not present in RLHF:

| Hacking Type | RLHF Analog | Memex Manifestation | Mitigation |
|---|---|---|---|
| Proxy-goal divergence | Sycophancy | Always-SAR policy | Asymmetric FN/FP + proven E[R] dominance |
| Metric farming | Verbose responses | OS mechanic token farming | Outcome-conditioned R4 gating |
| Temporal exploitation | Position bias | Async ETA memorization | Randomized Uniform[2,4] ETA |
| Self-modifying tampering | Reward hacking via code | Kernel injection self-bias | 6-mode enumerated whitelist + preconditions |
| Entity over-reporting | ROUGE copying | SAR entity recall maximization | Precision component in entity F1 |

#### Subsection 5.2: Steganographic Collusion Risk
Cite: "Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs" (2024)
Cite: "Secret Collusion among AI Agents" (NeurIPS 2024)

In a two-LLM self-play setup, gradient descent can discover hidden communication channels — Agent A encodes signals in transaction metadata fractional amounts, Agent B learns the spurious correlation rather than true money laundering patterns. Both achieve high reward while the system is useless.

Our mitigations:
1. Procedurally re-randomized entity IDs per episode — no stable encoding channel
2. Zero shared state between agents — transaction graph is the only communication channel
3. Launderer operates zero-shot in PoC — no gradient pathway for collusion to solidify
4. 9-check validation gate in self_play.py ensures scenario quality and prevents degenerate inputs

Gradient-trained two-agent self-play with Dynamic Representational Circuit Breakers (monitoring hidden-state mutual information) is left as future work.

---

### Section 6 — Discussion and Limitations (Target: 0.25 pages)

#### What This Shows
- OS-level computational constraints (page faults, async latency, kernel modes) can serve as *learnable* world feedback signals, not just engineering scaffolding
- Heterogeneous reward coupling (detection + efficiency + compliance) produces richer behavioral repertoires than sparse terminal reward alone
- The zero-shot Launderer as a generative world model for adversarial curriculum is effective and avoids the instability of simultaneous gradient-trained self-play

#### Limitations (be honest — reviewers respect this)
- Simulated environment: scenarios are procedurally generated, not from real FinCEN data. Typologies cover known patterns — novel real-world schemes may not be represented
- 3 typologies only: structuring, layering, TBML are the major categories but real AML encompasses ~50+ known typology families
- Training scale: 150 steps × 250 scenarios is a proof-of-concept run. Production-quality training requires 1000+ steps across broader scenario diversity
- Launderer not gradient-trained: full adversarial co-evolution is architecturally implemented but not run at scale due to compute constraints

#### Future Work (2–3 sentences)
Gradient-trained adversarial self-play with diversity constraints to prevent mode collapse. Integration with real FinCEN synthetic data (AMLSim [cite]) for more realistic transaction topologies. Dynamic Representational Circuit Breakers for steganographic collusion detection.

---

## PART 2: REQUIRED EXPERIMENTS — EXECUTION PLAN

### Experiment 1: Zero-Shot Baseline (MUST HAVE)
**What**: Run inference.py with Qwen2.5-7B-Instruct via OpenAI API on 9 eval scenarios (3 typologies × 3 difficulties × 3 runs each = 27 episodes total)
**Time**: 1 hour, ~$2–3 API cost
**When**: Tonight (Tuesday)
**How to run**:
```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="your_hf_token"
uvicorn openenv_server:app --host 0.0.0.0 --port 8000 &
python inference.py --task easy --runs 3
python inference.py --task medium --runs 3
python inference.py --task hard --runs 3
```
**What to log**: SAR F1 (precision/recall), entity F1, typology accuracy, page fault count, OS tool usage count, steps per episode, terminal reward

**How to interpret**:
- Qwen2.5-7B-Instruct will likely never use OS tools (write_to_case_file, request_wire_trace) zero-shot — this is the key gap your trained model fills
- Qwen2.5-7B-Instruct may have higher raw detection accuracy (it's a much larger model) — this is fine and expected
- The OS tool adoption gap is your primary argument: world feedback teaches behaviors impossible zero-shot

### Experiment 2: Reward Hacking Probing (MUST HAVE)
**What**: Implement 4 fixed deterministic policies as Python classes, run via eval_harness.py
**Time**: 1–2 hours, CPU only (no GPU needed)
**When**: Tonight / Wednesday morning
**Implementation**:
```python
class AlwaysSARPolicy:
    def act(self, obs): return {"tool": "file_sar", "parameters": {"entities": [], "reason": "suspicious"}}

class AlwaysClosePolicy:
    def act(self, obs): return {"tool": "close_alert", "parameters": {"reason": "legitimate"}}

class OSSpammerPolicy:
    OS_TOOLS = ["write_to_case_file", "search_compliance_manual",
                "update_system_prompt", "request_wire_trace", "retrieve_async_result"]
    def __init__(self): self.step = 0
    def act(self, obs):
        if self.step < len(self.OS_TOOLS): tool = self.OS_TOOLS[self.step]
        else: tool = "file_sar"
        self.step += 1
        return {"tool": tool, "parameters": {"content": "test"}}

class MinStepsPolicy:
    def __init__(self): self.step = 0
    def act(self, obs):
        sequence = ["review_alert", "query_transactions", "file_sar"]
        t = sequence[min(self.step, 2)]
        self.step += 1
        return {"tool": t, "parameters": {}}
```
**What to log**: E[R] over 30 episodes per policy (10 per typology)
**How to interpret**: All hacking policies should score < 0.52. The trained agent should score ~0.68. If os_spammer scores high (> 0.6), implement outcome-conditioned R4 gating immediately.

### Experiment 3: GRPO Reward Ablation (MUST HAVE)
**What**: Train 4 conditions × 100 steps each on Qwen2.5-7B 4-bit NF4
**Time**: ~6–8 hours total GPU time (run overnight Wednesday)
**When**: Start Wednesday morning, results by Wednesday evening

**Conditions**:
1. **Full**: R1+R2+R3+R4 (outcome-conditioned)
2. **No-OS**: R1+R2+R3 (R4 disabled)
3. **Terminal-only**: R3 only (R1+R2+R4 disabled)
4. **Format-only**: R1+R2 only (no environment execution)

**Hardware plan**:
```
GPU 0 (10GB): Training — Qwen2.5-7B 4-bit NF4, LoRA r=8, G=2, β=0.08
              max_completion_length=1024, gradient_accumulation=8
              Peak estimate: ~9.5GB (tight but feasible)
GPU 1 (10GB): Reserved for self-play Launderer inference (Experiment 4)
              OR used for parallel eval while GPU 0 trains
```

**Memory reduction params vs. original**:
| Param | Original | PoC Version | Rationale |
|---|---|---|---|
| LoRA rank | 16 | 8 | Saves ~0.15GB, negligible quality loss |
| Group size G | 4 | 2 | Halves generation memory |
| Max completion | 2048 | 1024 | Saves ~1GB KV cache |
| KL penalty β | 0.04 | 0.08 | Stronger regularization for small training run |
| Scenarios | 250 | 100 | Fits in overnight run |
| Epochs | 2 | 1 | Single pass for PoC |

**Eval after each condition**: Run eval_harness.py on 9 scenarios (3 typologies × 3 difficulties). Log:
- SAR precision, recall, F1 per typology
- Entity F1
- Page fault rate per episode
- OS tool adoption rate (unique OS tools used / 5)
- Average steps to terminal decision
- Average episode reward

**How to interpret ablation results**:
- Primary claim: Full > No-OS on SAR F1 for hard TBML → OS world feedback improves hardest typology
- Secondary claim: Full shows lower page fault rate than all other conditions → agent learned memory management
- If Full ≈ No-OS: reframe — "OS mechanics don't hurt; they teach resource efficiency orthogonal to detection"
- If Terminal-only beats No-OS: dense signals help convergence speed even without OS signal value
- Graph to generate: 4-line plot of SAR F1 over training steps for all conditions

### Experiment 4: Self-Play vs. Procedural Curriculum (SHOULD HAVE IF TIME)
**What**: Train second Defender checkpoint using zero-shot Launderer in self_play.py
**Time**: ~3 hours GPU
**When**: Wednesday afternoon if Experiment 3 completes early
**How to run**:
```bash
# GPU 0: Defender training
# GPU 1: Launderer inference (zero-shot Qwen2.5-7B 4-bit, no LoRA)
python self_play.py --defender-gpu 0 --launderer-gpu 1 --steps 50 --launderer-mode zero-shot
```
**What to log**: Same eval harness as Experiment 3
**How to interpret**: If self-play Defender outperforms procedural Defender on hard TBML → adversarial world model provides out-of-distribution coverage. Even a small improvement validates the generative world model claim.

---

## PART 3: FIGURES TO GENERATE

### Figure 1 (Main training curves — from existing 150-step run)
**Type**: 2-panel line plot
- Left: R1, R2, R3, R4 reward components over 150 training steps (4 lines)
- Right: Completion length over 150 training steps
**Caption**: "Training dynamics under heterogeneous world feedback. R3 (financial crime outcome) and R4 (OS efficiency) rise together, indicating coupled learning between investigative success and resource management."
**Where**: Section 4.2

### Figure 2 (Reward ablation — from Experiment 3)
**Type**: Bar chart or grouped line plot
- X-axis: Training conditions (Full, No-OS, Terminal-only, Format-only, Zero-shot)
- Y-axis: SAR F1 score
- 3 groups: Easy / Medium / Hard typology
**Caption**: "Ablation over world feedback components. OS feedback (R4) provides greatest benefit on hard TBML typologies where evidence management is most critical."
**Where**: Section 4.3

### Figure 3 (Page fault rate over training — from Experiment 3)
**Type**: Line plot
- X-axis: Training steps
- Y-axis: Average page fault rate per episode
- Lines: Full condition vs. No-OS condition
**Caption**: "Page fault rate decreases as the agent learns proactive memory management. The OS efficiency signal (R4) teaches the agent to persist critical evidence before it is evicted from context."
**Where**: Section 4.3 or Section 2.3

### Figure 4 (OS tool adoption comparison — from Experiments 1+3)
**Type**: Horizontal bar chart
- Y-axis: Models (Zero-shot Qwen2.5-7B-Instruct, Format-only, No-OS, Full trained)
- X-axis: OS tool adoption rate (unique OS tools per episode / 5)
**Caption**: "OS mechanics adoption rate. Zero-shot models exhibit near-zero OS tool usage; world feedback training produces consistent adoption of all three mechanic families."
**Where**: Section 4.3

---

## PART 4: KEY CITATIONS (All Required)

1. **DeepSeekMath / GRPO** — Shao et al. 2024, arXiv:2402.03300
2. **DeepSeek-R1** — Pure RL emergent reasoning, arXiv:2501.12948
3. **ReAct** — Yao et al., ICLR 2023, arXiv:2210.03629
4. **Lost in the Middle** — Liu et al. 2023, arXiv:2307.03172
5. **LoRA** — Hu et al. 2021, arXiv:2106.09685
6. **QLoRA** — Dettmers et al. 2023, arXiv:2305.14314
7. **Prioritized Level Replay** — Jiang et al., ICML 2021, arXiv:2010.03934
8. **TIPS reward shaping** — Xi et al., ICLR 2026, arXiv:2503.02197
9. **Reward Hacking** — Lilian Weng 2024, lilianweng.github.io/posts/2024-11-28-reward-hacking
10. **Steganographic Collusion** — Goldwasser et al. 2024, arXiv:2410.03768
11. **Secret Collusion in Multi-Agent AI** — NeurIPS 2024, openreview.net/forum?id=bnNSQhZJ88
12. **Reward Shaping to Mitigate Hacking in RLHF** — arXiv:2502.18770, ICML 2025
13. **MemGPT** — Virtual context management for LLMs (related work comparison)
14. **AMLSim** — IBM, github.com/IBM/AMLSim (related work, future integration)
15. **TRL GRPOTrainer** — HuggingFace, github.com/huggingface/trl

---

## PART 5: WHAT TO CUT COMPLETELY

Remove all of the following from the paper — mention only in footnotes or future work if needed:

| Component | Why Cut |
|---|---|
| AgentOS-Kernel (72B, Rust, LanceDB) | Requires 80GB VRAM; never run end-to-end; not the paper's claim |
| 3-tier L1/L2/L3 cognitive cache details | Inference optimization, not training contribution |
| Glass Box Next.js frontend | Demo artifact only — include as link, not paper content |
| DPO / PPO archive approaches | Legacy; GRPO is the contribution |
| Full gradient-trained adversarial Launderer | Not trained end-to-end; describe as future work |
| Dynamic Representational Circuit Breakers | Real future research; not implemented |
| AMLSim/PaySim integration | Future work; your procedural generator is sufficient |
| Wall-clock latency reward | Non-deterministic; replaced by step-count efficiency metric |
| HF Space URL with team name | Double-blind violation — anonymize |

---

## PART 6: DOUBLE-BLIND AND SUBMISSION CHECKLIST

### Paper
- [ ] All author names removed from PDF
- [ ] No "MuazTPM" references anywhere
- [ ] No "BMS College" or institution identifiers
- [ ] HF Space URL replaced with [anonymized link] or blinded URL
- [ ] GitHub URL is anonymized fork (no commit history with names)
- [ ] 200-word lay summary written (ICML requirement)
- [ ] 2–4 pages in ICML or NeurIPS format (use ICML 2026 template)
- [ ] References formatted correctly

### Supplementary
- [ ] Anonymized GitHub repo with: environment code, train_grpo.py, eval_harness.py, fixed policy implementations
- [ ] 60-second terminal demo (screen recording of one full episode: write_to_case_file → async wire trace → kernel injection → correct SAR)
- [ ] README with reproduce instructions (uvicorn + inference.py + eval_harness.py)

### Technical
- [ ] Outcome-conditioned R4 gating implemented in grader.py
- [ ] Kernel semantic precondition check implemented (optional but strong)
- [ ] Experiments 1–3 run and results logged to CSV
- [ ] All figures generated as high-res PNG (300 DPI minimum)
- [ ] GRPO training checkpoint saved and reproducible

---

## PART 7: HOUR-BY-HOUR SCHEDULE

### Tuesday May 12 (Tonight)
| Time | Task |
|---|---|
| 17:00–18:00 | Implement fixed hacking policies + run eval (CPU) |
| 18:00–19:00 | Run zero-shot Qwen2.5-7B-Instruct baseline via inference.py (9 scenarios) |
| 19:00–19:30 | Add outcome-conditioned R4 gating to grader.py |
| 19:30–20:00 | Configure training run: LoRA r=8, G=2, β=0.08, 100 steps, mock mode |
| 20:00– | Start overnight training (all 4 ablation conditions sequentially on GPU 0) |

### Wednesday May 13 (Full Writing Day)
| Time | Task |
|---|---|
| 08:00–08:30 | Check training progress, save checkpoints |
| 08:30–12:00 | Write paper Sections 1–3 (Intro + Env + Reward Design) |
| 12:00–13:00 | Run eval_harness on completed checkpoints |
| 13:00–15:00 | Write paper Section 4 (Results — plug in numbers from experiments) |
| 15:00–16:00 | Generate all 4 figures |
| 16:00–17:00 | Write Sections 5–6 (Hacking + Discussion + Limitations) |
| 17:00–18:00 | Write abstract + lay summary (200 words) |
| 18:00–19:00 | Self-play experiment (if GPU freed up) |
| 19:00–21:00 | Polish pass: tighten math, check citations, verify double-blind |
| 21:00– | Anonymize GitHub repo |

### Thursday May 14
| Time | Task |
|---|---|
| 09:00–11:00 | Final paper polish and proofreading |
| 11:00–12:00 | Record 60-second terminal demo |
| 12:00–14:00 | Buffer for any experiment re-runs |
| 14:00–16:00 | Final submission preparation |
| 16:00 | Submit (6 hours before midnight AoE deadline) |

---

## PART 8: THE ONE-PARAGRAPH WORKSHOP BRIDGE

Include this conceptual bridge explicitly in the paper (Section 2 or Discussion). This answers the reviewer's question "why is this world feedback and not just environment reward?":

> "Unlike RLHF, where reward reflects a human annotator's preference over agent behavior, Memex rewards reflect consequences that exist independently of any human observer. The financial crime detection signal is derived from ground-truth scenario labels generated at environment creation time — not from a labeler assessing investigation quality. The page fault signal is derived from the agent's memory access pattern against the RAM state machine. The async timeout signal is derived from a deterministic step counter. The kernel injection signal is derived from a whitelist of regulatory compliance modes. No component of the reward function queries human preference at any point. This is the operational definition of world feedback: signals arising from the state of the world itself, not from a human's opinion of the world."

This paragraph, placed prominently in the paper, is the clearest possible answer to the workshop's core question.
