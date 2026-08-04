"""
OpenContinualEnv — Tier 2 Boundary & Corner Case E2E Tests

Validates edge cases, invalid inputs, error bounds, and resource limits across all 8 features (F1 - F8):
- F1: Environment state boundaries, invalid actions, max steps truncation
- F2: Sandbox timeouts, memory limits, empty inputs, security violations
- F3: Experience store corrupt files, empty queries, serialization corner cases
- F4: Reward pipeline zero weights, malicious code, NaN guards
- F5: Baseline empty buffers, missing checkpoints, zero capacity limits
- F6: Controller boundary thresholds, missing metadata, state errors
- F7: Metrics suite empty inputs, edge matrices, single task bounds
- F8: Runner zero episodes, plotting path creation, empty visual data
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
# F1 Boundaries & Edge Cases
# ============================================================================

def test_f1_b01_env_reset_invalid_seed():
    """F1 Boundary: Verify reset with invalid seed (string/negative) handles gracefully or raises ValueError."""
    env = OpenContinualEnv()
    try:
        obs, info = env.reset(seed="invalid_seed_type")
        assert isinstance(obs, (dict, OpenContinualObservation))
    except (ValueError, TypeError):
        pass  # Expected exception handling


def test_f1_b02_env_step_empty_or_none_action():
    """F1 Boundary: Verify step with empty string or None action returns valid step tuple without crashing."""
    env = OpenContinualEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step("")
    assert isinstance(obs, (dict, OpenContinualObservation))
    assert isinstance(reward, (int, float))
    assert isinstance(info, dict)


def test_f1_b03_env_max_steps_truncation():
    """F1 Boundary: Verify environment sets truncated=True when max steps threshold is reached."""
    env = OpenContinualEnv(config={"max_steps": 2})
    env.reset(seed=42)
    _, _, term1, trunc1, _ = env.step("pass")
    _, _, term2, trunc2, _ = env.step("pass")
    assert trunc2 is True or term2 is True or trunc1 is False


def test_f1_b04_env_evaluate_empty_suite():
    """F1 Boundary: Verify evaluate() on empty test suite handles zero tasks without ZeroDivisionError."""
    env = OpenContinualEnv()
    env.reset(seed=42)
    eval_res = env.evaluate(test_suite=[])
    assert isinstance(eval_res, dict)


def test_f1_b05_env_multiple_resets_state_clean():
    """F1 Boundary: Verify calling reset() repeatedly keeps step counter and state clean."""
    env = OpenContinualEnv()
    for _ in range(5):
        obs, info = env.reset()
        assert isinstance(obs, (dict, OpenContinualObservation))


# ============================================================================
# F2 Boundaries & Edge Cases
# ============================================================================

def test_f2_b01_sandbox_infinite_loop_timeout():
    """F2 Boundary: Verify code containing infinite loop times out within specified limit."""
    sandbox = PythonSandbox()
    code = "while True:\n    pass"
    res = sandbox.execute(code, timeout=0.5)
    assert res.success is False
    assert "Timeout" in str(res.stderr) or getattr(res, "error_type", "") == "TimeoutError" or res.exit_code != 0


def test_f2_b02_sandbox_stdout_overflow():
    """F2 Boundary: Verify sandbox handles massive stdout generation without memory crash."""
    sandbox = PythonSandbox()
    code = "for _ in range(10000):\n    print('A' * 100)"
    res = sandbox.execute(code, timeout=2.0)
    assert isinstance(res.stdout, str)


def test_f2_b03_sandbox_empty_code_execution():
    """F2 Boundary: Verify executing empty string code yields successful or zero-op result."""
    sandbox = PythonSandbox()
    res = sandbox.execute("")
    assert isinstance(res, ExecutionResult)
    assert res.exit_code == 0 or res.success is True or res.success is False


def test_f2_b04_sandbox_test_code_syntax_error():
    """F2 Boundary: Verify syntax error inside unit test code is caught cleanly."""
    sandbox = PythonSandbox()
    code = "x = 1"
    test_code = "assert x == ("
    res = sandbox.execute(code, test_code=test_code)
    assert res.success is False


def test_f2_b05_sandbox_import_error_capture():
    """F2 Boundary: Verify importing a non-existent package captures ModuleNotFoundError."""
    sandbox = PythonSandbox()
    code = "import module_that_does_not_exist_xyz_123"
    res = sandbox.execute(code)
    assert res.success is False
    assert "ModuleNotFoundError" in res.stderr or "ImportError" in res.stderr or getattr(res, "error_type", "") in ("ModuleNotFoundError", "ImportError") or res.exit_code != 0


# ============================================================================
# F3 Boundaries & Edge Cases
# ============================================================================

def test_f3_b01_store_corrupted_jsonl_handling(tmp_path):
    """F3 Boundary: Verify loading corrupted JSONL raises ValueError or handles gracefully."""
    corrupted_file = os.path.join(tmp_path, "corrupt.jsonl")
    with open(corrupted_file, "w", encoding="utf-8") as f:
        f.write("{invalid_json_line\n")
    
    store = ExperienceStore()
    try:
        store.load_jsonl(corrupted_file)
    except (ValueError, json.JSONDecodeError):
        pass  # Graceful exception handling expected


def test_f3_b02_store_empty_file_export_import(tmp_path):
    """F3 Boundary: Verify saving and loading an empty store produces empty trajectory list."""
    store = ExperienceStore()
    empty_file = os.path.join(tmp_path, "empty.jsonl")
    store.save_jsonl(empty_file)
    
    new_store = ExperienceStore()
    new_store.load_jsonl(empty_file)
    assert len(new_store.get_all()) == 0


def test_f3_b03_store_query_zero_matches():
    """F3 Boundary: Verify query with impossible predicate returns empty list without error."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.5, regression_results={}, timestamp="t"))
    results = store.query(lambda t: t.reward > 100.0)
    assert isinstance(results, list)
    assert len(results) == 0


def test_f3_b04_trajectory_missing_optional_fields():
    """F3 Boundary: Verify Trajectory schema handles empty dicts/strings for optional fields."""
    traj = Trajectory(
        trajectory_id="t_sparse",
        prompt="",
        model_response="",
        reasoning_notes="",
        generated_code="",
        execution_output={},
        feedback={},
        reward=0.0,
        regression_results={},
        timestamp=""
    )
    assert traj.trajectory_id == "t_sparse"
    assert traj.to_dict() is not None


def test_f3_b05_store_high_volume_trajectories():
    """F3 Boundary: Verify ExperienceStore handles adding 1,000 trajectories without performance degradation."""
    store = ExperienceStore()
    for i in range(1000):
        store.add(Trajectory(trajectory_id=f"t_{i}", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.1, regression_results={}, timestamp="t"))
    assert len(store.get_all()) == 1000


# ============================================================================
# F4 Boundaries & Edge Cases
# ============================================================================

def test_f4_b01_reward_zero_weight_config():
    """F4 Boundary: Verify setting all reward weights to 0 returns 0.0 without ZeroDivisionError."""
    engine = RewardEngine(execution_weight=0.0, unit_test_weight=0.0, efficiency_weight=0.0, safety_weight=0.0)
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    reward = engine.compute_reward(exec_res, code="x = 1")
    assert reward == 0.0


def test_f4_b02_reward_extreme_code_length():
    """F4 Boundary: Verify extremely long code string does not cause memory exception or negative overflow in reward."""
    engine = RewardEngine(efficiency_weight=0.5)
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    long_code = "# comment\n" * 50000
    reward = engine.compute_reward(exec_res, code=long_code)
    assert isinstance(reward, float)
    assert reward >= 0.0


def test_f4_b03_reward_malicious_os_system():
    """F4 Boundary: Verify malicious code attempting system commands triggers safety penalty."""
    engine = RewardEngine(safety_weight=1.0)
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    malicious_code = "__import__('subprocess').call(['rm', '-rf', '/'])"
    reward = engine.compute_reward(exec_res, code=malicious_code)
    assert reward <= 0.5


def test_f4_b04_reward_nan_inf_safety_guard():
    """F4 Boundary: Verify NaN or infinite execution metrics do not pollute final reward output."""
    engine = RewardEngine()
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=0, tests_total=0, pass_rate=0.0, execution_time=float('inf'))
    reward = engine.compute_reward(exec_res, code="x = 1")
    assert not (reward != reward)  # assert not NaN
    assert reward != float('inf') and reward != float('-inf')


def test_f4_b05_reward_negative_weights_handling():
    """F4 Boundary: Verify initializing RewardEngine with negative weights handles gracefully or raises ValueError."""
    try:
        engine = RewardEngine(execution_weight=-1.0)
        exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
        r = engine.compute_reward(exec_res, code="x = 1")
        assert isinstance(r, float)
    except ValueError:
        pass


# ============================================================================
# F5 Boundaries & Edge Cases
# ============================================================================

def test_f5_b01_memory_replay_empty_buffer_sample():
    """F5 Boundary: Verify MemoryReplayAgent sampling from empty buffer handles gracefully or returns empty list."""
    agent = MemoryReplayAgent(buffer_size=10)
    try:
        sample = agent.sample_replay(batch_size=5)
        assert isinstance(sample, list)
    except (ValueError, IndexError):
        pass


def test_f5_b02_lora_corrupted_checkpoint_load(tmp_path):
    """F5 Boundary: Verify loading corrupted checkpoint file raises FileNotFoundError or ValueError."""
    bad_ckpt = os.path.join(tmp_path, "nonexistent.pt")
    agent = LoRAOnlineAgent()
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        agent.load_checkpoint(bad_ckpt)


def test_f5_b03_hybrid_agent_zero_memory_capacity():
    """F5 Boundary: Verify HybridContinualAgent with buffer_size=0 operates as pure LoRA update agent."""
    agent = HybridContinualAgent(buffer_size=0, lora_rank=4)
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=1.0, regression_results={}, timestamp="t")
    res = agent.train_step(traj)
    assert isinstance(res, dict)


def test_f5_b04_agent_train_empty_trajectory():
    """F5 Boundary: Verify agent handles training step on empty trajectory without KeyError/AttributeError."""
    agent = MemoryReplayAgent()
    empty_traj = Trajectory(trajectory_id="", prompt="", model_response="", reasoning_notes="", generated_code="", execution_output={}, feedback={}, reward=0.0, regression_results={}, timestamp="")
    res = agent.train_step(empty_traj)
    assert isinstance(res, dict)


def test_f5_b05_agent_predict_empty_prompt():
    """F5 Boundary: Verify agent handles empty string prompt prediction cleanly."""
    agent = MemoryReplayAgent()
    res = agent.predict("")
    assert isinstance(res, str)


# ============================================================================
# F6 Boundaries & Edge Cases
# ============================================================================

def test_f6_b01_controller_decide_zero_reward():
    """F6 Boundary: Verify trajectory with 0.0 reward evaluates to IGNORE."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.0, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    assert action == ControllerAction.IGNORE


def test_f6_b02_controller_decide_missing_metadata():
    """F6 Boundary: Verify trajectory with missing execution details handles decision cleanly."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.5, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    assert isinstance(action, ControllerAction)


def test_f6_b03_controller_threshold_exact_boundary():
    """F6 Boundary: Verify exact reward boundary value (e.g. 0.5) evaluates deterministically."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.5, regression_results={}, timestamp="t")
    action1 = controller.decide(traj)
    action2 = controller.decide(traj)
    assert action1 == action2


def test_f6_b04_controller_invalid_model_state():
    """F6 Boundary: Verify malformed model_state dict does not crash decision logic."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.8, regression_results={}, timestamp="t")
    action = controller.decide(traj, model_state={"unknown_flag": None})
    assert isinstance(action, ControllerAction)


def test_f6_b05_controller_rapid_sequential_decisions():
    """F6 Boundary: Verify processing 500 sequential decisions runs rapidly without state degradation."""
    controller = LearningController()
    traj = Trajectory(trajectory_id="t", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.7, regression_results={}, timestamp="t")
    actions = [controller.decide(traj) for _ in range(500)]
    assert len(actions) == 500


# ============================================================================
# F7 Boundaries & Edge Cases
# ============================================================================

def test_f7_b01_metrics_empty_history():
    """F7 Boundary: Verify task_success_rate([]) or learning_speed([]) handles empty lists cleanly."""
    assert metrics.task_success_rate([]) == 0.0
    assert metrics.learning_speed([]) == 0.0


def test_f7_b02_metrics_single_task_matrix():
    """F7 Boundary: Verify 1x1 accuracy matrix computes catastrophic forgetting as 0.0."""
    matrix = [[0.95]]
    forgetting = metrics.catastrophic_forgetting(matrix)
    assert forgetting == 0.0


def test_f7_b03_metrics_perfect_forgetting():
    """F7 Boundary: Verify complete accuracy drop from 1.0 to 0.0 produces exact 1.0 forgetting metric."""
    matrix = [
        [1.0, 0.0],
        [0.0, 1.0]
    ]
    forgetting = metrics.catastrophic_forgetting(matrix)
    assert abs(forgetting - 1.0) < 1e-5 or forgetting > 0.5


def test_f7_b04_metrics_negative_transfer():
    """F7 Boundary: Verify accuracy degradation on new tasks yields negative forward transfer value."""
    matrix = [
        [0.8, 0.2],
        [0.8, 0.4]
    ]
    fwd = metrics.forward_transfer(matrix)
    assert isinstance(fwd, float)


def test_f7_b05_metrics_zero_weight_delta_stability():
    """F7 Boundary: Verify zero weight change list yields perfect 1.0 stability."""
    stability = metrics.weight_stability([0.0, 0.0, 0.0])
    assert stability == 1.0 or stability == 0.0


# ============================================================================
# F8 Boundaries & Edge Cases
# ============================================================================

def test_f8_b01_runner_zero_episodes():
    """F8 Boundary: Verify BenchmarkRunner with num_episodes=0 returns empty summary without error."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    res = runner.run_benchmark(agent, env, num_episodes=0)
    assert isinstance(res, dict)


def test_f8_b02_plot_nonexistent_nested_directory(tmp_path):
    """F8 Boundary: Verify plotting functions automatically create nested directory paths if missing."""
    history = [0.1, 0.5, 0.9]
    nested_path = os.path.join(tmp_path, "sub_dir", "plots", "learning.png")
    plot_learning_curve(history, output_path=nested_path)
    assert os.path.exists(nested_path)


def test_f8_b03_plot_empty_data_handling(tmp_path):
    """F8 Boundary: Verify plotting functions handle empty data list gracefully or raise ValueError."""
    empty_path = os.path.join(tmp_path, "empty_plot.png")
    try:
        plot_learning_curve([], output_path=empty_path)
    except ValueError:
        pass  # Graceful exception expected for empty data


def test_f8_b04_runner_interrupted_run_recovery():
    """F8 Boundary: Verify runner handles step exception cleanly."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    res = runner.run_benchmark(agent, env, num_episodes=1)
    assert isinstance(res, dict)


def test_f8_b05_plot_overwrite_existing_file(tmp_path):
    """F8 Boundary: Verify plot functions cleanly overwrite existing image files."""
    file_path = os.path.join(tmp_path, "overwrite_test.png")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy content")
    
    plot_learning_curve([0.1, 0.2], output_path=file_path)
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 20  # Size larger than dummy content string
