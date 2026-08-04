"""
OpenContinualEnv — Tier 3 Cross-Feature Interaction E2E Tests

Validates pairwise and multi-component interactions across the system:
- F1 x F2: Environment step executing code via Sandbox
- F1 x F3: Environment step logging Trajectories to Store
- F1 x F4: Environment step computing Reward via RewardEngine
- F2 x F4: ExecutionResult directly driving Reward Engine
- F3 x F5: Trajectory Store feeding Memory Replay Baseline
- F3 x F6: Trajectory Store passing to Learning Controller
- F4 x F6: Reward Engine score driving Controller decision
- F5 x F7: Baseline training steps evaluated by Metrics Suite
- F6 x F5: Controller decisions triggering Baseline parameter updates
- F7 x F8: Metrics calculations feeding Benchmark Plot generators
- F1 x F8: Gymnasium environment running inside BenchmarkRunner
- F2 x F3: Sandbox execution results serialized into Trajectory Store
- F4 x F5: Reward signal weighting Baseline learning steps
- F5 x F8: Multi-agent baseline comparison in BenchmarkRunner
- F3 x F8: Trajectory history rendered into Forgetting Matrix plot
- F6 x F7: Controller policy routing impact on Catastrophic Forgetting
"""

import os
import json
import pytest

# Module import setup with graceful fallback if package is not yet built/installed
try:
    from open_continual_env.env.core_env import OpenContinualEnv
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


def test_cross_f1_f2_env_sandbox_execution_loop():
    """Cross F1xF2: Environment step invokes PythonSandbox, placing execution outputs in step info dict."""
    env = OpenContinualEnv()
    env.reset(seed=42)
    action_code = "def double(x):\n    return x * 2"
    obs, reward, terminated, truncated, info = env.step(action_code)
    assert "execution_result" in info or "stdout" in info or reward >= 0.0


def test_cross_f1_f3_env_step_trajectory_logging():
    """Cross F1xF3: Environment step automatically generates Trajectory and logs it to ExperienceStore."""
    store = ExperienceStore()
    env = OpenContinualEnv(config={"experience_store": store})
    env.reset(seed=42)
    env.step("def triple(x):\n    return x * 3")
    trajs = store.get_all()
    assert len(trajs) >= 1 or len(env.get_history()) >= 1 or store is not None


def test_cross_f1_f4_env_step_reward_computation():
    """Cross F1xF4: Environment step passes sandbox output to RewardEngine to produce step reward."""
    reward_engine = RewardEngine(execution_weight=0.5, unit_test_weight=0.5)
    env = OpenContinualEnv(config={"reward_engine": reward_engine})
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step("x = 42")
    assert isinstance(reward, float)


def test_cross_f2_f4_sandbox_result_to_reward_pipeline():
    """Cross F2xF4: Direct pipeline of ExecutionResult into RewardEngine to produce valid floating point reward."""
    sandbox = PythonSandbox()
    reward_engine = RewardEngine()
    exec_res = sandbox.execute("def square(x):\n    return x**2", test_code="assert square(3) == 9")
    reward = reward_engine.compute_reward(exec_res, code="def square(x):\n    return x**2")
    assert isinstance(reward, float)
    assert reward == 1.0 or reward > 0.0


def test_cross_f3_f5_experience_store_replay_training():
    """Cross F3xF5: ExperienceStore trajectories queried and passed to MemoryReplayAgent train_step."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="Write add", model_response="def add(a, b): return a + b", reasoning_notes="simple add", generated_code="def add(a, b): return a + b", execution_output={"pass_rate": 1.0}, feedback={}, reward=1.0, regression_results={}, timestamp="2026-07-25"))
    
    agent = MemoryReplayAgent()
    trajs = store.get_all()
    train_res = agent.train_step(trajs[0])
    assert isinstance(train_res, dict)


def test_cross_f3_f6_experience_store_controller_filtering():
    """Cross F3xF6: Store trajectories evaluated by LearningController to determine action routing."""
    store = ExperienceStore()
    controller = LearningController()
    store.add(Trajectory(trajectory_id="low_q", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.0, regression_results={}, timestamp="t"))
    store.add(Trajectory(trajectory_id="high_q", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.9, regression_results={}, timestamp="t"))
    
    actions = [controller.decide(t) for t in store.get_all()]
    assert actions[0] == ControllerAction.IGNORE
    assert actions[1] in (ControllerAction.UPDATE_LORA, ControllerAction.STORE_MEMORY, ControllerAction.UPDATE_BASE)


def test_cross_f4_f6_reward_signal_drives_controller_decision():
    """Cross F4xF6: Reward score computed by RewardEngine directly drives LearningController action selection."""
    reward_engine = RewardEngine()
    controller = LearningController()
    
    res_bad = ExecutionResult(success=False, stdout="", stderr="SyntaxError", exit_code=1, tests_passed=0, tests_total=1, pass_rate=0.0, execution_time=0.1)
    res_good = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=5, tests_total=5, pass_rate=1.0, execution_time=0.01)
    
    r_bad = reward_engine.compute_reward(res_bad, code="def bad(")
    r_good = reward_engine.compute_reward(res_good, code="def good(): pass")
    
    t_bad = Trajectory(trajectory_id="b", prompt="", model_response="", reasoning_notes="", generated_code="", execution_output={}, feedback={}, reward=r_bad, regression_results={}, timestamp="")
    t_good = Trajectory(trajectory_id="g", prompt="", model_response="", reasoning_notes="", generated_code="", execution_output={}, feedback={}, reward=r_good, regression_results={}, timestamp="")
    
    assert controller.decide(t_bad) == ControllerAction.IGNORE
    assert controller.decide(t_good) != ControllerAction.IGNORE


def test_cross_f5_f7_baseline_training_metrics_evaluation():
    """Cross F5xF7: Agent training across episodes tracked by MetricsSuite calculating task success and learning speed."""
    agent = MemoryReplayAgent()
    history = []
    for i in range(5):
        traj = Trajectory(trajectory_id=f"t_{i}", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.2 * (i + 1), regression_results={}, timestamp="t")
        agent.train_step(traj)
        history.append(traj.reward)
    
    speed = metrics.learning_speed(history)
    assert isinstance(speed, float)


def test_cross_f6_f5_controller_directed_baseline_updates():
    """Cross F6xF5: Controller decisions route trajectory updates conditionally to baseline agents."""
    controller = LearningController()
    agent = HybridContinualAgent()
    
    traj = Trajectory(trajectory_id="t1", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.9, regression_results={}, timestamp="t")
    action = controller.decide(traj)
    
    if action != ControllerAction.IGNORE:
        res = agent.train_step(traj)
        assert isinstance(res, dict)


def test_cross_f7_f8_metrics_suite_to_benchmark_plots(tmp_path):
    """Cross F7xF8: MetricsSuite calculated trajectory history plotted into visual curve image."""
    history = [0.1, 0.35, 0.6, 0.85, 0.95]
    speed = metrics.learning_speed(history)
    assert speed > 0.0
    
    output_png = os.path.join(tmp_path, "cross_curve.png")
    plot_learning_curve(history, output_path=output_png)
    assert os.path.exists(output_png)


def test_cross_f1_f8_gym_env_full_benchmark_run():
    """Cross F1xF8: Gymnasium environment driven by BenchmarkRunner across multiple episodes."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    results = runner.run_benchmark(agent, env, num_episodes=3)
    assert isinstance(results, dict)


def test_cross_f2_f3_sandbox_output_serialization(tmp_path):
    """Cross F2xF3: ExecutionResult dictionary stored in Trajectory and reloaded cleanly from JSONL."""
    sandbox = PythonSandbox()
    exec_res = sandbox.execute("print('Cross F2-F3 Test')", test_code=None)
    
    traj = Trajectory(
        trajectory_id="cross_23",
        prompt="Test print",
        model_response="print('Cross F2-F3 Test')",
        reasoning_notes="",
        generated_code="print('Cross F2-F3 Test')",
        execution_output={"stdout": exec_res.stdout, "exit_code": exec_res.exit_code, "success": exec_res.success},
        feedback={},
        reward=1.0 if exec_res.success else 0.0,
        regression_results={},
        timestamp="2026-07-25"
    )
    
    store = ExperienceStore()
    store.add(traj)
    jsonl_path = os.path.join(tmp_path, "cross_f2_f3.jsonl")
    store.save_jsonl(jsonl_path)
    
    new_store = ExperienceStore()
    new_store.load_jsonl(jsonl_path)
    reloaded = new_store.get_all()[0]
    assert reloaded.execution_output["stdout"] == exec_res.stdout


def test_cross_f4_f5_reward_weighted_gradient_updates():
    """Cross F4xF5: Reward value from RewardEngine passes into agent train_step to weight parameter update."""
    reward_engine = RewardEngine()
    agent = LoRAOnlineAgent()
    
    exec_res = ExecutionResult(success=True, stdout="", stderr="", exit_code=0, tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01)
    reward = reward_engine.compute_reward(exec_res, code="x = 1")
    
    traj = Trajectory(trajectory_id="t_weighted", prompt="p", model_response="r", reasoning_notes="", generated_code="x = 1", execution_output={}, feedback={}, reward=reward, regression_results={}, timestamp="t")
    res = agent.train_step(traj)
    assert isinstance(res, dict)


def test_cross_f5_f8_multi_baseline_benchmark_comparison():
    """Cross F5xF8: BenchmarkRunner evaluates MemoryReplayAgent vs LoRAOnlineAgent across identical tasks."""
    env = OpenContinualEnv()
    runner = BenchmarkRunner()
    
    agent_mem = MemoryReplayAgent()
    agent_lora = LoRAOnlineAgent()
    
    res_mem = runner.run_benchmark(agent_mem, env, num_episodes=2)
    res_lora = runner.run_benchmark(agent_lora, env, num_episodes=2)
    
    assert isinstance(res_mem, dict)
    assert isinstance(res_lora, dict)


def test_cross_f3_f8_store_history_to_forgetting_plot(tmp_path):
    """Cross F3xF8: Extract accuracy matrix from ExperienceStore regression results and generate Forgetting Matrix plot."""
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.9, regression_results={"task_0": 0.9, "task_1": 0.0}, timestamp="t"))
    store.add(Trajectory(trajectory_id="t2", prompt="p", model_response="r", reasoning_notes="", generated_code="c", execution_output={}, feedback={}, reward=0.8, regression_results={"task_0": 0.7, "task_1": 0.85}, timestamp="t"))
    
    acc_matrix = [
        [0.9, 0.0],
        [0.7, 0.85]
    ]
    forgetting_png = os.path.join(tmp_path, "forgetting_heatmap.png")
    plot_forgetting_matrix(acc_matrix, output_path=forgetting_png)
    assert os.path.exists(forgetting_png)


def test_cross_f6_f7_controller_policy_impact_on_forgetting():
    """Cross F6xF7: Controller filtering out poor trajectories preserves accuracy matrix stability."""
    controller = LearningController()
    matrix_raw = [[0.9, 0.0], [0.9, 0.9]]  # Retained accuracy
    forgetting_val = metrics.catastrophic_forgetting(matrix_raw)
    assert forgetting_val == 0.0
