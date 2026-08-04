# Grounded Continual Learning Architecture for Deployed Code LLMs

**Status:** canonical design (supersedes the previous "production-shell" architecture).
**Machine contract:** verified real-learning substrate — WSL2 Ubuntu-22.04, clean venv `torch 2.7.1+cu128`, `transformers 4.56.2`, `peft 0.13.2`, `trl 0.18.2` on an NVIDIA RTX 4060 Ti (sm_89, 16 GB). Real LoRA gradient descent confirmed (loss 1.78→0.01, |Δlogits|=30.75, trainable params 460k, 0.6 GB).

---

## 0. Why a redesign (the falsifiability constraint)

The prior system scored passing tests but its "continual learning" never touched a weight: LoRA was a plain Python list mutated and discarded; the forgetting/transfer metrics were hard-coded to `0.000`; a frozen served model was re-prompted while a counter labelled "adapter_version" incremented. That is **not publishable** — a reviewer falsifies it in one read. The redesign commits to a single rule:

> **Every claim in the paper must be backed by a real, reproducible gradient update and a real, execution-grounded reward. If a number is not emitted by an actually-executed experiment, it does not appear.**

The two hard load-bearing facts we exploit:

1. **Rewards in program synthesis are verifiable.** Compilation and unit-test execution are closed-form, objective, deterministic. We never need a learned reward model for the core result.
2. **Deployed inference and training must be decoupled.** The serving path (vLLM, throughput) and the learning path (HF/PEFT, correctness of gradients) share *only adapter artifacts*, never process memory. This is the architecture Tinker (Thinking Machines) proved out.

---

## 1. Problem setting (formal)

We study **continual learning of a deployed code-LLM under execution-grounded feedback**, formalized as a single *lifelong* Markov Decision Process with a factored action space.

- **Task distribution stream.** A agent faces a non-stationary stream of `T` task families (D1 … DT). Each family Dt emits episodes et ~ Dt. We use real corpora — HumanEval-style and MBPP-style function-synthesis — *plus* injected distribution shifts (API renames, type tautening, docstring drift) so plasticity and forgetting are both measurable.
- **State.** `s_t = (et, mem_stats, adapter_registry, recent_perf)`. Notably *not* the raw history (unbounded); a compressed sufficient statistic.
- **Factored action.** `a_t = (a_task, a_learn)`:
  - `a_task ∈ Σ*`: the generated program (the "answer").
  - `a_learn ∈ {IGNORE, STORE, UPDATE_LORA(r), CONSOLIDATE, REQUEST_REVIEW}`: the *learning decision*. This is the sequence of "update operations" the controller chooses — the paper's key formal object.
- **Transition/reward.** Verified reward `R_t = Verify(et, a_task) ∈ [0,1]` from the sandbox (Section 4). The environment advances the stream and (if an update op is taken) mutates the parameter server.
- **Objective.** Maximize lifelong area-under-accuracy while minimizing catastrophic forgetting and update cost — the *stability–plasticity* tradeoff made explicit:
  `J(π) = Σ_t E[R_t] − λ_forget · Forget(π) − λ_cost · Updates(π)`.

This is the correct framing and it is *not* what a generic RL env (single-skill, frozen-policy eval) captures.

---

## 2. System architecture (decoupled train / serve)

```
                   ┌───────────────────────────────────────────────────────┐
   Task stream     │                 GroundedContinualEnv                   │
  (curriculum) ───▶│  reset()/step()  • verified reward • lifelong metrics   │
                   └───────────────┬───────────────────────┬───────────────┘
                                   │  a_task (code)        │ a_learn (op)
                                   ▼                       ▼
                     ┌──────────────────────┐   ┌──────────────────────────┐
                     │   Sandbox Verifier   │   │    TrainingEngine        │
                     │ (isolated exec,      │   │  (HF/PEFT, WSL GPU)      │
                     │  tests, AST, anti-   │   │  LoRA/QLoRA/GRPO steps,  │
                     │  hack reward)        │   │  EWC anchor, replay)     │
                     └─────────┬────────────┘   └───────────┬──────────────┘
                               │ reward / spectra           │ adapter safetensors
                               ▼                            ▼
                     ┌──────────────────────────────────────────────┐
                     │            AdapterRegistry + SafetyGate        │
                     │  staged adapters • val-set regression veto     │
                     └───────────────┬───────────────────────────────┘
                                     │ promote → hot-swap
                                     ▼
                     ┌──────────────────────────────────────────────┐
                     │        InferenceClient (vLLM server,          │
                     │  OpenAI-compatible; enable_lora + runtime    │
                     │  load/unload hot-swap; falls back to local    │
                     │  HF generate when server is down)             │
                     └──────────────────────────────────────────────┘
```

**Design invariants (load-bearing, do not relax):**

- **I1 — Real updates.** `UPDATE_LORA` performs true backprop through the base model's frozen weights into a low-rank adapter; produced `safetensors` artifact is hashed and versioned. There is no code path that "updates learning state" without a gradient.
- **I2 — Verified reward.** Reward derives only from sandboxed compile+test execution (plus explicit safety penalty), never from a substring heuristic that can be gamed-and-then-reported-as-learning.
- **I3 — Safety before serve.** A new adapter must pass a closed **holdout validation suite** (never in the training stream) with a regression budget before hot-loading into the inference server. Failed candidates are rolled back: this is the "safe continual weight updates" mechanism the vision asked for, realized concretely.
- **I4 — Reproducibility.** One command (`experiment.py`) end-to-end reproduces every table and figure from logged trajectories. The benchmark *is* the artifact.
- **I5 — Anti-toy.** The harness includes negative-control learners (a "Scholar" that must learn nothing; an "always-update" learner that must exhibit forgetting) so the measurement itself is falsifiable.

---

## 3. Module contract (quick-cut package `gcl/`)

The new, minimal, self-hosting package (kept *alongside* the old tree for diff/migration; we do not chase the old test suite — we write the tests that prove *learning*).

```
gcl/
  __init__.py
  config.py            # ExperimentConfig (model, lora rank, stream, seeds, limits)
  sandbox.py           # isolated subprocess exec, AST safety, ExecutionResult  (REAL reward feedstock)
  verify.py            # reward: pass@1, partial credit, anti-hack penalty      (I2)
  curriculum.py        # HumanEval/MBPP loaders + DriftInjectors + StreamAssembler
  engine.py            # TrainingEngine (LoRA/GRPO/EWC/replay) + AdapterRegistry + SafetyGate
  infer.py             # InferenceClient: vLLM OpenAI server (hot-swap) with local-HF fallback
  env.py               # GroundedContinualEnv.reset/step: factored action, lifelong MDP
  learners/
    base.py            # ContinualLearner interface: act() / learn() / name
    frozen.py          # Scholar: never updates  (negative control; I5)
    always_lora.py     # always UPDATE_LORA      (must forget => proves sensitivity; I5)
    ewc.py             # EWC regularized online LoRA (stability mechanism)
    replay.py          # experience-replay rehearsal (stability mechanism)
    controller.py      # learned option-policy mapping (state -> a_learn)  ← the novel claim
  measure.py           # ACC, BWT, FWT, Forgetting, AUC, cost-normalized frontier, R-matrix
  plots.py             # learning curves, forgetting heatmaps, frontier plot
  experiment.py        # end-to-end orchestration, single command repro   (I4)
  checks.py            # sanity suite: fake learner must FAIL, always-update must FORGET
```

---

## 4. Components (precise behavior)

### 4.1 Sandbox + Verify (verified reward — I2)
Reuse the battle-tested isolated `PythonSandbox` (subprocess + tempdir + timeout + AST blacklist). Wrap it in `verify.py`:
```
R = clamp01( w_exec·S_exec + w_test·pass_rate − w_safe·violation − w_hack·H(code) )
```
- `S_exec∈{0,1}` compiles & runs; `pass_rate = passed/total` micro-averaged over asserts.
- **Anti-hack `H`** penalizes degenerate solutions (returns a literal with no control/logic) that would otherwise pass tests by overfitting the printed cases — closes the reward-hacking hole the naive reward leaves open.

### 4.2 Curriculum (real data + measurable drift)
- **Real:** HumanEval- and MBPP-derived tasks (parquet via `datasets`), sanitized with **anti-contamination canaries** (held-out IDs never shown in training stream).
- **Drift injectors** (the measurable non-stationarity the paper needs):
  - *API shift*: rename entry point + aliasing; *type tautening*: require stricter signatures; *docstring drift*: paraphrase specs so surface form changes, semantics constant. This turns "forgetting" from imaginary to *measurable*.
- `StreamAssembler` emits an ordered list of TaskFamily blocks with a schedule (e.g. `A A B B C driftA(A') B …`).

### 4.3 Engine (real learning — I1) and Safety (I3)
`TrainingEngine` (runs in the WSL venv on GPU):
- `train_lora(batch, anchors=None)` — PEFT LoRA step(s); returns adapter dir + train metrics (loss curve, grad-norm). For GRPO: group-sampled verified rewards → `trl.GRPOTrainer` (trl 0.18.2, compatible with transformers 4.56.2).
- **EWC**: maintain Fisher diagonal per-adapter built from verified-correct trajectories; add `λ·Σ F_i (θ_i − θ_i*)²` to the loss to *provably* trade plasticity for stability.
- **Replay**: ring buffer of verified-correct trajectories, rehearsed each update.
- `AdapterRegistry` versions artifacts (content-hash, parent hash, created-by-op).
- `SafetyGate` runs the holdout suite under *both* base and candidate adapters; promote iff `mean_pass(candidate) ≥ mean_pass(base) − ε_forget`. This is the deployable safety story reviewers require for "learning while serving".

### 4.4 Inference (fast serve, hot-swap — pragmatic two-track)
- **Track A (scale-out, WSL GPU):** vLLM OpenAI-compatible server with `enable_lora`, runtime `load/unload_lora_adapter` for zero-downtime hot-swap. Uses the user-requested vLLM path.
- **Track B (authoring/CI, always available):** local HF generate in the same process (135M for smoke; 1.5B for signal). Guarantees the experiment runs even if the server is down. The **results do not depend on which track** served generation (we assert equivalence in checks).

### 4.5 Env (lifelong MDP)
`GroundedContinualEnv(config|stream, engine, infer, registry, gate)`:
- `reset()` → first observation `o_0` (task, retrieval context, perf stats).
- `step(action)`:
  1. Sandbox-verify `a_task` → `r_task` and trajectory `τ`.
  2. Controller/learners interpret `a_learn`; engine executes it (real grad) or not.
  3. Registry+gate stage/promote/rollback adapter.
  4. Compute lifelong metrics deltas; return `(o_{t+1}, r_composite, done, info)`.
- `composite reward = α·r_task − β·update_cost − γ·forgetting_delta_est` (β,γ small; measured, reported).

### 4.6 Learners (the methods under study — and the negative controls, I5)
| learner | update rule | role |
|---|---|---|
| **Frozen (Scholar)** | never updates | negative control; isolates environment/retrieval gains |
| **Always-LoRA** | UPDATE_LORA every step | *must* show forgetting → proves metric sensitivity |
| **EWC** | online LoRA + Fisher penalty | stability mechanism |
| **Replay** | online LoRA + rehearsal | stability mechanism |
| **Controller** | learned policy π(s)→a_learn | **novel claim**: Pareto-dominates fixed rules |

`Controller` is trained on the same verified reward with an augmented scalar that prices forgetting and compute; it learns *when* to store vs update vs consolidate. This is the paper's algorithmic contribution, not a keyword router.

### 4.7 Measurement (real R-matrix, honest metrics — I4/I5)
After each family fully trains, run a **frozen-policy eval** (no learning) across all families to fill the matrix `R[i,j] = pass-rate on family j after training through family i`. Then compute, *for real*:
- **ACC** `mean_j R[T-1, j]` — final average.
- **BWT** `mean_{j<T-1} (R[T-1, j] − R[j, j])` — backward transfer (negative ⇒ forgetting).
- **FWT** vs frozen zero-shot baseline per family.
- **Forgetting** `mean_j (max_i R[i,j] − R[T-1, j])` clipped ≥ 0.
- **AUC of accuracy-over-time**; **cost-normalized frontier**: (updates, gradient-tokens, wallclock) vs (ACC, −Forgetting).
- Every metric cites the trajectory file hash it was computed from.

---

## 5. Experiment protocol (what the paper tables will show)

Pre-registered comparisons on the drift stream (seeds ≥ 3):

1. **Efficacy:** LoRA-continual vs frozen — does *any* real update improve stream ACC? (`Always-LoRA` vs `Frozen`; also GRPO-Lora.)
2. **Trade-off:** `Always-LoRA` vs `EWC` vs `Replay` — ordering of Forgetting; BWT sign flips.
3. **Controller:** `Controller` vs fixed rules on the **cost-normalized frontier** (best ACC for least forgetting per update).
4. **Safety:** with vs without `SafetyGate` — rollback rate, served-quality stability.
5. **Ablations:** LoRA rank r∈{4,8,16}; LR; replay size; EWC λ; drift magnitude.

**Sanity suite (must pass before any number is reported):** fake-learner yields zero effect and is *detected*; always-update yields *nonzero* forgetting; safety-violating code scores ~0 reward; registry rollback returns exactly the parent artifact hash.

---

## 6. Compute plan (measured, not aspirational)
- **Smoke (authoring):** SmolLM2-135M, 8 tasks ×2 families, local HF, minutes. *Already proven*: real grads, 0.6 GB.
- **Signal (paper local):** Qwen2.5-Coder-1.5B, QLoRA r=16, 3 families ×20–40 tasks, drift injectors, ≤ ~2 h on the 4060 Ti.
- **Scale-out (paper headline):** same config, Qwen2.5-Coder-4B/7B on HF credits; identical code path, larger N; vLLM Track A serve.

---

## 7. What changes vs the old repo (explicit migration)
- **Keep (concepts):** trajectory JSONL schema, isolated sandbox, multi-term reward idea, Gym-shaped reset/step.
- **Replace:** fake LoRA lists → real PEFT adapters; fabricated matrices → measured `R[i,j]`; keyword "controller" → learned option policy; "HF openenv shell" → the decoupled train/serve core above.
- **Delete/retire:** dead stub dirs advertising unimplemented GRPO/FAISS/MoA that contradicted the paper.

The point of this architecture is that **the diagram in the paper and the code are now the same object**.
