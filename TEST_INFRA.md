# OpenContinualEnv — E2E Testing Infrastructure Specification (`TEST_INFRA.md`)

## 1. Overview & Architecture Philosophy

`OpenContinualEnv` is an open, Gymnasium-compliant research environment designed for evaluating lifelong and continual learning algorithms in deployed Large Language Models (LLMs) on coding and reasoning tasks.

To ensure strict quality assurance, architectural alignment, and behavioral correctness across the full system lifecycle, the end-to-end (E2E) testing framework provides an **opaque-box evaluation methodology**. The test suite treats the environment, sandbox, experience store, controller, baselines, and benchmark tools as integrated modules, validating their external interfaces and system dynamics without altering internal implementations.

### Key Testing Principles:
1. **Opaque-Box E2E Design**: Verification strictly targets public API contracts, data schemas, state transitions, and file outputs.
2. **Multi-Tier Hierarchy**: Structured into 4 distinct test tiers, ranging from isolated feature validation (Tier 1) to multi-episode lifelong learning scenarios (Tier 4).
3. **No-Cheating Integrity Guarantee**: Tests perform real logic execution, filesystem I/O, mathematical verification, and artifact validation. Fake assertions, hardcoded mock returns, and dummy facades are prohibited.
4. **Pytest & UV Integration**: Test suites are natively executable via `uv run pytest tests/e2e`.

---

## 2. Feature Coverage Architecture (F1 – F8)

The test suite covers all eight (8) core system features defined in the project architecture:

```
+---------------------------------------------------------------------------------------------------+
|                                     OpenContinualEnv Architecture                                  |
+---------------------------------------------------------------------------------------------------+
|  F1: Farama Gymnasium Interface  <--->  F2: Python Execution Sandbox  <--->  F4: Configurable Reward|
|          (reset, step, eval)                 (code exec, errors, tests)           (success, safety)|
|                                                        |                                          |
|                                                        v                                          |
|  F6: Learning Controller        <--->  F3: Experience Trajectory Store  <--->  F5: Continual Baselines|
|      (Ignore, Memory, LoRA, Base)            (JSON/JSONL, query, feedback)        (Replay, LoRA, Hybrid)|
|                                                        |                                          |
|                                                        v                                          |
|  F8: Experiment Runner & Plots   <--->  F7: Comprehensive Metrics Suite                          |
|      (benchmarks, curves, matrices)          (Success, Speed, Forgetting, Transfer, Stability)    |
+---------------------------------------------------------------------------------------------------+
```

### Feature 1: Farama Gymnasium Interface & Hugging Face OpenEnv Architecture (F1)
- **Module**: `open_continual_env.env.core_env.OpenContinualEnv` & `OpenContinualGymWrapper`
- **Specification**:
  - `OpenContinualEnv` subclasses Hugging Face `openenv` core architecture (`openenv.core.Environment`).
  - HF OpenEnv schema dataclasses: `OpenContinualObservation` (inherits `openenv.core.Observation`), `OpenContinualAction` (inherits `openenv.core.Action`), `OpenContinualState` (inherits `openenv.core.State`).
  - Compatibility layer: `OpenContinualGymWrapper(gymnasium.Env)` for Farama Gymnasium compatibility.
  - `reset(seed=None, options=None)` -> returns `(observation: OpenContinualObservation, info: dict)`
  - `step(action: OpenContinualAction | str | dict)` -> returns `(observation: OpenContinualObservation, reward: float, terminated: bool, truncated: bool, info: dict)`
  - `evaluate(model=None, test_suite=None)` -> returns evaluation metrics dictionary.
  - Verification of Gymnasium state consistency, HF openenv integration, seed reproducibility, and step reward integration.

### Feature 2: Real Python Execution Sandbox & Error Capture (F2)
- **Module**: `open_continual_env.env.sandbox.PythonSandbox`
- **Specification**:
  - `execute(code: str, test_code: str = None, timeout: float = 5.0)` -> returns `ExecutionResult`
  - Captures `stdout`, `stderr`, `exit_code`, `tests_passed`, `tests_total`, `pass_rate`, `execution_time`, `error_type`.
  - Safely intercepts `SyntaxError`, `TimeoutError`, `ZeroDivisionError`, `TypeError`, and runtime exceptions.

### Feature 3: Trajectory Experience Store & Query Engine (F3)
- **Module**: `open_continual_env.trajectory.schema.Trajectory` & `open_continual_env.trajectory.store.ExperienceStore`
- **Specification**:
  - `Trajectory` schema fields: `trajectory_id`, `prompt`, `model_response`, `reasoning_notes`, `generated_code`, `execution_output`, `feedback`, `reward`, `regression_results`, `timestamp`.
  - `ExperienceStore` methods: `add()`, `query(filter_fn)`, `save_json()`, `load_json()`, `save_jsonl()`, `load_jsonl()`, `get_all()`, `clear()`.
  - Guarantees schema preservation across serialization/deserialization cycles.

### Feature 4: Configurable Reward Pipeline (F4)
- **Module**: `open_continual_env.env.rewards.RewardEngine`
- **Specification**:
  - `compute_reward(execution_result, code="", metadata=None)` -> returns `float` reward in `[0.0, 1.0]`.
  - Configurable sub-components:
    - `execution_weight`: Bonus for non-zero exit code / success.
    - `unit_test_weight`: Proportional reward based on test pass rate.
    - `efficiency_weight`: Length/speed optimization score.
    - `safety_penalty`: Deductions for unsafe AST constructs or forbidden imports (`os.system`, `subprocess`, etc.).

### Feature 5: Continual Learning Baselines (F5)
- **Module**: `open_continual_env.baselines` (`MemoryReplayAgent`, `LoRAOnlineAgent`, `HybridContinualAgent`)
- **Specification**:
  - Standardized interface `BaseContinualAgent`: `train_step(trajectory)`, `predict(prompt)`, `save_checkpoint(path)`, `load_checkpoint(path)`.
  - Replay buffer sampling, LoRA gradient update emulation/step, and combined Hybrid memory+adapter retention.

### Feature 6: Learning Controller Policy (F6)
- **Module**: `open_continual_env.controller.learning_controller.LearningController` & `ControllerAction`
- **Specification**:
  - `ControllerAction` Enum: `IGNORE = 0`, `STORE_MEMORY = 1`, `UPDATE_LORA = 2`, `UPDATE_BASE = 3`.
  - `decide(trajectory: Trajectory, model_state: dict = None)` -> returns `ControllerAction`.
  - Policy heuristics based on reward thresholds, task difficulty, error taxonomy, and regression risk.

### Feature 7: Metrics Suite (F7)
- **Module**: `open_continual_env.benchmark.metrics`
- **Specification**:
  - Functions: `task_success_rate()`, `learning_speed()`, `catastrophic_forgetting()`, `backward_transfer()`, `forward_transfer()`, `weight_stability()`, `compute_all_metrics()`.
  - Standard mathematical formulas for accuracy matrices $R_{i,j}$ and weight stability metrics.

### Feature 8: Benchmark Experiment Runner & Plotting Tools (F8)
- **Module**: `open_continual_env.benchmark.runner.BenchmarkRunner` & `open_continual_env.benchmark.plots`
- **Specification**:
  - `BenchmarkRunner.run_benchmark(agent, env, num_episodes)` -> executes benchmark evaluation and returns experiment summary dict.
  - Visualization functions: `plot_learning_curve(history, output_path)` and `plot_forgetting_matrix(matrix, output_path)` generating valid image files (PNG/SVG).

---

## 3. Test Suite Hierarchy & Structure

The tests are located under `tests/e2e/` and divided into four functional tiers:

```
tests/e2e/
├── test_tier1_feature_coverage.py     # Tier 1: Basic feature validation (>=5 tests per feature)
├── test_tier2_boundary_corner.py      # Tier 2: Edge cases, errors, boundaries (>=5 tests per feature)
├── test_tier3_cross_feature.py         # Tier 3: Pairwise interaction & workflow tests
└── test_tier4_real_world.py            # Tier 4: Multi-episode lifelong learning scenario benchmarks
```

### Test Count Requirements Matrix

| Tier | Test File | Target Coverage | Min Test Count |
|---|---|---|---|
| **Tier 1** | `test_tier1_feature_coverage.py` | Full feature API happy-path validation (F1 - F8) | 40 tests (5 per feature) |
| **Tier 2** | `test_tier2_boundary_corner.py` | Edge cases, invalid inputs, timeouts, error bounds | 40 tests (5 per feature) |
| **Tier 3** | `test_tier3_cross_feature.py` | Pairwise component interactions (F1xF3, F2xF4, F3xF6, etc.) | 16 tests |
| **Tier 4** | `test_tier4_real_world.py` | Complete multi-episode continuous learning scenarios | 10 tests |
| **Total** | | | **>= 106 E2E Tests** |

---

## 4. Test Invocation & Verification Protocol

Tests are invoked using standard `pytest` under `uv`:

```bash
# Run all E2E tests with verbose output
uv run pytest tests/e2e -v

# Run specific tier
uv run pytest tests/e2e/test_tier1_feature_coverage.py -v
```

### Test Robustness Strategy
To prevent test collection failures during incremental baseline/environment development:
1. Tests dynamically verify availability of `open_continual_env` modules using `importorskip` or try-except safeguards where appropriate.
2. Direct instantiation tests assert on full API signatures, return types, dictionary schemas, and behavior.
3. File I/O tests use isolated `tmp_path` fixtures to ensure safe execution without side effects on the project workspace.

---

## 5. Audit & Integrity Standards

All test cases adhere to the following mandatory standards:
- **No Mock Hardcoding**: Tests execute real operations (e.g. running Python code inside the sandbox, writing JSONL files to disk, computing matrix math for metrics).
- **Explicit Assertions**: Statements verify exact data structures, range boundaries, type specifications, and state changes.
- **Traceability**: Every test function name clearly reflects its target feature and test tier (e.g. `test_f1_tier1_reset_returns_valid_observation`).
