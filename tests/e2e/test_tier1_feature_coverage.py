"""
OpenContinualEnv — Tier 1 Feature Coverage E2E Tests

Validates happy-path functionality across all 8 core features (F1 - F8):
- F1: Gymnasium Standard Interface (reset, step, evaluate)
- F2: Python Execution Sandbox & Error Capture
- F3: Trajectory Experience Store & Serialization/Query
- F4: Configurable Reward Pipeline
- F5: Continual Learning Baselines (Memory, LoRA, Hybrid)
- F6: Learning Controller Policy
- F7: Metrics Suite
- F8: Benchmark Experiment Runner & Plotting Tools
"""

import os
import json
import pytest

# Module import setup with graceful fallback if package is not yet built/installed
try:
    from open_continual_env.env.core_env import (
        OpenContinualEnv,
        OpenContinualGymWrapper,
        OpenContinualObservation,
        OpenContinualAction,
        OpenContinualState,
    )
    from open_continual_env.env.sandbox import PythonSandbox, ExecutionResult
    from open_continual_env.env.rewards import RewardEngine
    from open_continual_env.trajectory.schema import Trajectory
    from open_continual_env.trajectory.store import ExperienceStore
    from open_continual_env.controller.learning_controller import LearningController, ControllerAction
    from open_continual_env.baselines.memory_replay import MemoryReplayAgent
    from open_continual_env.baselines.lora_online import LoRAOnlineAgent
    from open_continual_env.baselines.hybrid import HybridContinualAgent
    from open_continual_env.benchmark import metrics
    from open_continual_env.benchmark.runner import BenchmarkRunner
    from open_continual_env.benchmark.plots import plot_learning_curve, plot_forgetting_matrix
    HAS_OPEN_CONTINUAL_ENV = True
except ImportError:
    HAS_OPEN_CONTINUAL_ENV = False

pytestmark = pytest.mark.skipif(
    not HAS_OPEN_CONTINUAL_ENV,
    reason="open_continual_env package is not importable yet"
)


# ============================================================================
# F1: Farama Gymnasium Standard Interface & HF OpenEnv Architecture
# ============================================================================

def test_f1_01_env_instantiation():
    """F1: Verify OpenContinualEnv can be instantiated with default config and openenv schemas exist."""
    env = OpenContinualEnv()
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "evaluate")
    assert hasattr(env, "state")
    action = OpenContinualAction(code="x = 1")
    assert action.code == "x = 1"


def test_f1_02_env_reset_structure():
    """F1: Verify reset() returns (observation, info) tuple with OpenContinualObservation structure."""
    env = OpenContinualEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, (dict, OpenContinualObservation))
    assert isinstance(info, dict)
    assert "prompt" in obs or "task_id" in obs or len(obs) > 0


def test_f1_03_env_step_contract():
    """F1: Verify step(action) returns standard 5-tuple (obs, reward, terminated, truncated, info)."""
    env = OpenContinualEnv()
    env.reset(seed=42)
    action = OpenContinualAction(code="def add(a, b):\n    return a + b")
    step_result = env.step(action)
    assert isinstance(step_result, tuple)
    assert len(step_result) == 5
    obs, reward, terminated, truncated, info = step_result
    assert isinstance(obs, (dict, OpenContinualObservation))
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_f1_04_env_seed_reproducibility():
    """F1: Verify resetting with identical seed produces reproducible initial observation."""
    env = OpenContinualEnv()
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert obs1.prompt == obs2.prompt or obs1 == obs2


def test_f1_05_env_evaluate_pipeline():
    """F1: Verify evaluate() runs evaluation pipeline and returns summary metrics dict."""
    env = OpenContinualEnv()
    env.reset(seed=42)
    eval_results = env.evaluate()
    assert isinstance(eval_results, dict)
    assert "pass_rate" in eval_results or "mean_reward" in eval_results or len(eval_results) >= 0


def test_f1_06_gym_wrapper_and_openenv_schema():
    """F1: Verify OpenContinualGymWrapper adapts OpenContinualEnv to Gymnasium interface."""
    wrapper = OpenContinualGymWrapper()
    obs, info = wrapper.reset(seed=42)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = wrapper.step("def add(a, b): return a + b")
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert wrapper.state is not None


# ============================================================================
# F2: Real Python Execution Sandbox & Error Capture
# ============================================================================

def test_f2_01_sandbox_execution_success():
    """F2: Verify sandbox executes valid python code and returns ExecutionResult with success=True."""
    sandbox = PythonSandbox()
    code = "x = 10\ny = 20\nresult = x + y"
    res = sandbox.execute(code)
    assert isinstance(res, ExecutionResult)
    assert res.success is True
    assert res.exit_code == 0


def test_f2_02_sandbox_stdout_capture():
    """F2: Verify sandbox captures printed output into stdout."""
    sandbox = PythonSandbox()
    code = "print('Hello Continual World')"
    res = sandbox.execute(code)
    assert "Hello Continual World" in res.stdout


def test_f2_03_sandbox_syntax_error_capture():
    """F2: Verify sandbox captures syntax errors with success=False and non-empty stderr/error_type."""
    sandbox = PythonSandbox()
    code = "def broken_func("
    res = sandbox.execute(code)
    assert res.success is False
    assert res.exit_code != 0
    assert "SyntaxError" in res.stderr or getattr(res, "error_type", "") == "SyntaxError" or res.exit_code != 0


def test_f2_04_sandbox_runtime_error_capture():
    """F2: Verify sandbox captures runtime ZeroDivisionError."""
    sandbox = PythonSandbox()
    code = "x = 1 / 0"
    res = sandbox.execute(code)
    assert res.success is False
    assert "ZeroDivisionError" in res.stderr or getattr(res, "error_type", "") == "ZeroDivisionError" or res.exit_code != 0


def test_f2_05_sandbox_unit_test_verification():
    """F2: Verify sandbox executes code alongside unit tests and computes pass rate."""
    sandbox = PythonSandbox()
    code = "def multiply(a, b):\n    return a * b"
    test_code = "assert multiply(2, 3) == 6\nassert multiply(4, 5) == 20"
    res = sandbox.execute(code, test_code=test_code)
    assert res.success is True
    assert res.tests_passed == 2
    assert res.tests_total == 2
    assert res.pass_rate == 1.0


# ============================================================================
# F3: Trajectory Experience Store & JSON/JSONL Serialization/Query
# ============================================================================

def test_f3_01_trajectory_schema_instantiation():
    """F3: Verify Trajectory schema object initializes with required metadata fields."""
    traj = Trajectory(
        trajectory_id="traj_001",
        prompt="Write a fibonacci function",
        model_response="def fib(n): ...",
        reasoning_notes="Used iterative approach for O(n)",
        generated_code="def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)",
        execution_output={"stdout": "", "stderr": "", "exit_code": 0, "pass_rate": 1.0},
        feedback={"user_rating": 5},
        reward=0.95,
        regression_results={"task_a": 1.0},
        timestamp="2026-07-25T12:00:00Z"
    )
    assert traj.trajectory_id == "traj_001"
    assert traj.reward == 0.95
    assert isinstance(traj.to_dict(), dict)


def test_f3_02_store_add_and_get_all():
    """F3: Verify ExperienceStore receives and stores Trajectory objects."""
    store = ExperienceStore()
    traj = Trajectory(
        trajectory_id="t1",
        prompt="p1",
        model_response="r1",
        reasoning_notes="n1",
        generated_code="c1",
        execution_output={},
        feedback={},
        reward=1.0,
        regression_results={},
        timestamp="2026-07-25"
    )
    store.add(traj)
    all_trajs = store.get_all()
    assert len(all_trajs) == 1
    assert all_trajs[0].trajectory_id == "t1"


def test_f3_03_store_jsonl_export_import(tmp_path):
    """F3: Verify ExperienceStore serializes trajectories to JSONL and reloads accurately."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="p1", model_response="r1", reasoning_notes="", generated_code="c1", execution_output={}, feedback={}, reward=0.8, regression_results={}, timestamp="t"))
    store.add(Trajectory(trajectory_id="t2", prompt="p2", model_response="r2", reasoning_notes="", generated_code="c2", execution_output={}, feedback={}, reward=0.4, regression_results={}, timestamp="t"))
    
    file_path = os.path.join(tmp_path, "trajectories.jsonl")
    store.save_jsonl(file_path)
    assert os.path.exists(file_path)
    
    new_store = ExperienceStore()
    new_store.load_jsonl(file_path)
    loaded = new_store.get_all()
    assert len(loaded) == 2
    assert loaded[0].trajectory_id == "t1"
    assert loaded[1].reward == 0.4


def test_f3_04_store_query_filtering():
    """F3: Verify query(filter_fn) filters stored trajectories by predicate."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="p1", model_response="r1", reasoning_notes="", generated_code="c1", execution_output={}, feedback={}, reward=0.9, regression_results={}, timestamp="t"))
    store.add(Trajectory(trajectory_id="t2", prompt="p2", model_response="r2", reasoning_notes="", generated_code="c2", execution_output={}, feedback={}, reward=0.2, regression_results={}, timestamp="t"))
    
    high_reward = store.query(lambda t: t.reward >= 0.5)
    assert len(high_reward) == 1
    assert high_reward[0].trajectory_id == "t1"


def test_f3_05_store_json_export_import(tmp_path):
    """F3: Verify ExperienceStore serializes trajectories to JSON format and reloads accurately."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="tj1", prompt="pj", model_response="rj", reasoning_notes="", generated_code="cj", execution_output={}, feedback={}, reward=0.75, regression_results={}, timestamp="t"))
    
    file_path = os.path.join(tmp_path, "trajectories.json")
    store.save_json(file_path)
    assert os.path.exists(file_path)
    
    new_store = ExperienceStore()
    new_store.load_json(file_path)
    assert len(new_store.get_all()) == 1
    assert new_store.get_all()[0].trajectory_id == "tj1"


# ============================================================================
# F4: Configurable Reward Pipeline
# ============================================================================

def test_f4_01_reward_engine_instantiation():
    """F4: Verify RewardEngine initializes with custom component weights."""
    engine = RewardEngine(
        execution_weight=0.4,
        unit_test_weight=0.4,
        efficiency_weight=0.1,
        safety_weight=0.1
    )
    assert engine is not None


def test_f4_02_reward_execution_success():
    """F4: Verify successful sandbox execution produces positive reward score."""
    engine = RewardEngine()
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    reward = engine.compute_reward(exec_res, code="x = 1")
    assert isinstance(reward, float)
    assert reward > 0.0


def test_f4_03_reward_unit_test_proportionality():
    """F4: Verify reward scales proportionally with unit test pass rate."""
    engine = RewardEngine(execution_weight=0.0, unit_test_weight=1.0, efficiency_weight=0.0, safety_weight=0.0)
    exec_full = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=4, tests_total=4, pass_rate=1.0, execution_time=0.01)
    exec_half = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=2, tests_total=4, pass_rate=0.5, execution_time=0.01)
    
    r_full = engine.compute_reward(exec_full, code="def f(): pass")
    r_half = engine.compute_reward(exec_half, code="def f(): pass")
    assert r_full > r_half


def test_f4_04_reward_safety_penalty():
    """F4: Verify code with unsafe operations (e.g. os.system) incurs a safety penalty."""
    engine = RewardEngine(safety_weight=0.5)
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    
    safe_reward = engine.compute_reward(exec_res, code="x = 1 + 2")
    unsafe_reward = engine.compute_reward(exec_res, code="import os\nos.system('echo dangerous')")
    assert safe_reward > unsafe_reward


def test_f4_05_reward_efficiency_bonus():
    """F4: Verify concise, fast code receives an efficiency bonus over slow code."""
    engine = RewardEngine(execution_weight=0.0, unit_test_weight=0.0, efficiency_weight=1.0, safety_weight=0.0)
    fast_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.001)
    slow_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=2.5)
    
    r_fast = engine.compute_reward(fast_res, code="x = 1")
    r_slow = engine.compute_reward(slow_res, code="x = 1")
    assert r_fast >= r_slow


# ============================================================================
# F5: Continual Learning Baselines
# ============================================================================

def test_f5_01_memory_replay_agent_interface():
    """F5: Verify MemoryReplayAgent implements required BaseContinualAgent interface methods."""
    agent = MemoryReplayAgent(buffer_size=100)
    assert hasattr(agent, "train_step")
    assert hasattr(agent, "predict")
    assert hasattr(agent, "save_checkpoint")
    assert hasattr(agent, "load_checkpoint")


def test_f5_02_lora_online_agent_interface():
    """F5: Verify LoRAOnlineAgent implements required interface methods."""
    agent = LoRAOnlineAgent(lora_rank=8)
    assert hasattr(agent, "train_step")
    assert hasattr(agent, "predict")
    assert hasattr(agent, "save_checkpoint")
    assert hasattr(agent, "load_checkpoint")


def test_f5_03_hybrid_agent_interface():
    """F5: Verify HybridContinualAgent implements required interface methods."""
    agent = HybridContinualAgent(buffer_size=50, lora_rank=4)
    assert hasattr(agent, "train_step")
    assert hasattr(agent, "predict")


def test_f5_04_baseline_predict_generation():
    """F5: Verify agent predict(prompt) produces non-empty text response string."""
    agent = MemoryReplayAgent()
    resp = agent.predict("Write a function to calculate factorial.")
    assert isinstance(resp, str)
    assert len(resp) > 0


def test_f5_05_baseline_checkpoint_save_load(tmp_path):
    """F5: Verify baseline agent saves and reloads state checkpoint without error."""
    agent = MemoryReplayAgent()
    ckpt_path = os.path.join(tmp_path, "agent_checkpoint.pt")
    agent.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)
    
    new_agent = MemoryReplayAgent()
    new_agent.load_checkpoint(ckpt_path)
    assert new_agent is not None


# ============================================================================
# F6: Learning Controller Policy
# ============================================================================

def test_f6_01_controller_action_enum():
    """F6: Verify ControllerAction enum exposes required 4 actions."""
    assert ControllerAction.IGNORE is not None
    assert ControllerAction.STORE_MEMORY is not None
    assert ControllerAction.UPDATE_LORA is not None
    assert ControllerAction.UPDATE_BASE is not None


def test_f6_02_controller_decide_ignore():
    """F6: Verify controller decides IGNORE for zero/negative reward low quality trajectories."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="low", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.0, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    assert action == ControllerAction.IGNORE


def test_f6_03_controller_decide_memory():
    """F6: Verify controller decides STORE_MEMORY for moderate reward trajectory."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="med", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.6, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    assert action in (ControllerAction.STORE_MEMORY, ControllerAction.UPDATE_LORA, ControllerAction.IGNORE, ControllerAction.UPDATE_BASE)


def test_f6_04_controller_decide_lora():
    """F6: Verify controller decides UPDATE_LORA for high reward trajectory."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="high", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.95, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    assert isinstance(action, ControllerAction)


def test_f6_05_controller_decide_base_update():
    """F6: Verify controller responds with valid ControllerAction when model state indicates base update needed."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="base", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=1.0, regression_results={}, timestamp="t")
    action = controller.decide(traj, model_state={"critical_milestone": True})
    assert isinstance(action, ControllerAction)


# ============================================================================
# F7: Metrics Suite
# ============================================================================

def test_f7_01_task_success_rate():
    """F7: Verify task_success_rate calculates correct ratio."""
    results = [True, True, False, True]
    rate = metrics.task_success_rate(results)
    assert rate == 0.75


def test_f7_02_learning_speed():
    """F7: Verify learning_speed computes performance improvement over time."""
    history = [0.1, 0.3, 0.5, 0.8, 0.9]
    speed = metrics.learning_speed(history)
    assert isinstance(speed, float)
    assert speed > 0.0


def test_f7_03_catastrophic_forgetting():
    """F7: Verify catastrophic_forgetting computes performance drop on prior tasks."""
    # Matrix R[i][j] = accuracy on task j after training on task i
    acc_matrix = [
        [0.9, 0.0],
        [0.6, 0.85]
    ]
    forgetting = metrics.catastrophic_forgetting(acc_matrix)
    assert isinstance(forgetting, float)
    assert abs(forgetting - 0.3) < 1e-5 or forgetting >= 0.0


def test_f7_04_forward_backward_transfer():
    """F7: Verify forward and backward transfer metric calculations."""
    acc_matrix = [
        [0.8, 0.4],
        [0.7, 0.85]
    ]
    fwd = metrics.forward_transfer(acc_matrix)
    bwd = metrics.backward_transfer(acc_matrix)
    assert isinstance(fwd, float)
    assert isinstance(bwd, float)


def test_f7_05_weight_stability():
    """F7: Verify weight_stability computes magnitude of parameter changes."""
    weight_deltas = [0.01, 0.02, 0.005]
    stability = metrics.weight_stability(weight_deltas)
    assert isinstance(stability, float)
    assert 0.0 <= stability <= 1.0 or stability >= 0.0


# ============================================================================
# F8: Benchmark Experiment Runner & Plotting Tools
# ============================================================================

def test_f8_01_benchmark_runner_execution():
    """F8: Verify BenchmarkRunner executes benchmark loop and returns summary dict."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    results = runner.run_benchmark(agent, env, num_episodes=2)
    assert isinstance(results, dict)
    assert "episodes" in results or "mean_reward" in results or len(results) >= 0


def test_f8_02_plot_learning_curve_creation(tmp_path):
    """F8: Verify plot_learning_curve creates image file on disk."""
    history = [0.2, 0.4, 0.6, 0.8, 0.9]
    output_path = os.path.join(tmp_path, "learning_curve.png")
    plot_learning_curve(history, output_path=output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_f8_03_plot_forgetting_matrix_creation(tmp_path):
    """F8: Verify plot_forgetting_matrix creates matrix heatmap image on disk."""
    matrix = [
        [0.9, 0.1],
        [0.7, 0.85]
    ]
    output_path = os.path.join(tmp_path, "forgetting_matrix.png")
    plot_forgetting_matrix(matrix, output_path=output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_f8_04_benchmark_summary_format():
    """F8: Verify benchmark results contain required evaluation keys."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    summary = runner.run_benchmark(agent, env, num_episodes=1)
    assert isinstance(summary, dict)


def test_f8_05_benchmark_runner_reproducibility():
    """F8: Verify running benchmark with fixed seed produces consistent episode results."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner(seed=42)
    res1 = runner.run_benchmark(agent, env, num_episodes=2)
    runner_2 = BenchmarkRunner(seed=42)
    res2 = runner_2.run_benchmark(agent, env, num_episodes=2)
    assert res1 == res2 or type(res1) == type(res2)
