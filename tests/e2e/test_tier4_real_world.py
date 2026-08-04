"""
OpenContinualEnv — Tier 4 Real-World Lifelong Learning Scenario E2E Tests

Validates complete real-world deployment benchmarks and multi-episode lifelong learning scenarios:
- Scenario 1: Lifelong coding task progression across multiple domains
- Scenario 2: Learning Controller dynamic policy routing over experience stream
- Scenario 3: Experience store multi-session persistence and replay buffer recovery
- Scenario 4: Catastrophic forgetting mitigation benchmark (Task A -> Task B -> Task A)
- Scenario 5: End-to-end benchmark execution, metric calculation, and plot reporting
- Scenario 6: Reward pipeline safety and code efficiency tradeoff evaluation
- Scenario 7: Sandbox isolation and stability under adversarial code streams
- Scenario 8: Multi-task forward and backward transfer benchmark
- Scenario 9: Crash recovery, checkpoint resumption, and continuous training
- Scenario 10: Full 8-component synchronous environment lifecycle evaluation
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


def test_tier4_01_lifelong_coding_benchmark_scenario():
    """Scenario 1: Full multi-task sequential coding benchmark evaluating task success across sequential domains."""
    env = OpenContinualEnv()
    agent = HybridContinualAgent()
    
    tasks = [
        "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
        "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
        "def binary_search(arr, target):\n    l, r = 0, len(arr) - 1\n    while l <= r:\n        mid = (l + r) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: l = mid + 1\n        else: r = mid - 1\n    return -1"
    ]
    
    rewards = []
    for code in tasks:
        obs, reward, term, trunc, info = env.step(code)
        traj = Trajectory(
            trajectory_id=f"t_{len(rewards)}",
            prompt="Write function",
            model_response=code,
            reasoning_notes="",
            generated_code=code,
            execution_output=info,
            feedback={},
            reward=reward,
            regression_results={},
            timestamp="2026-07-25"
        )
        agent.train_step(traj)
        rewards.append(reward)
    
    assert len(rewards) == 3
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_tier4_02_learning_controller_dynamic_routing_scenario():
    """Scenario 2: Stream of varied quality execution trajectories dynamically routed by LearningController."""
    controller = LearningController()
    
    trajectories = [
        Trajectory("t0", "p", "r", "", "broken code (", {}, {}, 0.0, {}, "2026"),
        Trajectory("t1", "p", "r", "", "def f(): pass", {}, {}, 0.6, {}, "2026"),
        Trajectory("t2", "p", "r", "", "def f(): return 42", {}, {}, 0.95, {}, "2026"),
        Trajectory("t3", "p", "r", "", "print('hello')", {}, {}, 0.4, {}, "2026")
    ]
    
    decisions = [controller.decide(t) for t in trajectories]
    assert len(decisions) == 4
    assert decisions[0] == ControllerAction.IGNORE
    assert any(d != ControllerAction.IGNORE for d in decisions[1:])


def test_tier4_03_experience_store_session_persistence_scenario(tmp_path):
    """Scenario 3: Lifelong learning session serializes 50 trajectories to disk, reloaded in a new session."""
    store = ExperienceStore()
    for i in range(50):
        store.add(Trajectory(
            trajectory_id=f"session_1_t{i}",
            prompt=f"Prompt {i}",
            model_response=f"Response {i}",
            reasoning_notes=f"Notes {i}",
            generated_code=f"code_{i} = {i}",
            execution_output={"pass_rate": 1.0 if i % 2 == 0 else 0.5},
            feedback={},
            reward=1.0 if i % 2 == 0 else 0.5,
            regression_results={},
            timestamp="2026-07-25"
        ))
    
    storage_file = os.path.join(tmp_path, "session_history.jsonl")
    store.save_jsonl(storage_file)
    assert os.path.getsize(storage_file) > 0
    
    # Simulate fresh deployment session
    fresh_store = ExperienceStore()
    fresh_store.load_jsonl(storage_file)
    reloaded_trajs = fresh_store.get_all()
    assert len(reloaded_trajs) == 50
    
    high_value_trajs = fresh_store.query(lambda t: t.reward == 1.0)
    assert len(high_value_trajs) == 25


def test_tier4_04_continual_learning_catastrophic_forgetting_mitigation():
    """Scenario 4: Benchmark evaluating Catastrophic Forgetting metric on Task A -> Task B -> Task A sequence."""
    acc_matrix = [
        [0.90, 0.00],  # After Task A
        [0.85, 0.88]   # After Task B (retained Task A at 85%)
    ]
    
    forgetting = metrics.catastrophic_forgetting(acc_matrix)
    bwd_transfer = metrics.backward_transfer(acc_matrix)
    
    # Low forgetting rate indicates effective catastrophic forgetting mitigation
    assert forgetting <= 0.10
    assert isinstance(bwd_transfer, float)


def test_tier4_05_end_to_end_benchmark_runner_reporting_and_plotting(tmp_path):
    """Scenario 5: Complete experiment pipeline: Gym Env -> Agent -> Metrics -> Plot Files -> Result Summary."""
    env = OpenContinualEnv()
    agent = MemoryReplayAgent()
    runner = BenchmarkRunner()
    
    benchmark_summary = runner.run_benchmark(agent, env, num_episodes=3)
    assert isinstance(benchmark_summary, dict)
    
    curve_png = os.path.join(tmp_path, "e2e_learning_curve.png")
    matrix_png = os.path.join(tmp_path, "e2e_forgetting_matrix.png")
    
    history_rewards = [0.2, 0.5, 0.8]
    acc_matrix = [[0.9, 0.1], [0.8, 0.85]]
    
    plot_learning_curve(history_rewards, output_path=curve_png)
    plot_forgetting_matrix(acc_matrix, output_path=matrix_png)
    
    assert os.path.exists(curve_png)
    assert os.path.exists(matrix_png)


def test_tier4_06_reward_pipeline_safety_and_efficiency_tradeoff():
    """Scenario 6: Evaluate reward pipeline tradeoff between code safety and execution efficiency."""
    reward_engine = RewardEngine(execution_weight=0.3, unit_test_weight=0.3, efficiency_weight=0.2, safety_weight=0.2)
    sandbox = PythonSandbox()
    
    code_safe_efficient = "def f(): return sum(range(100))"
    code_unsafe = "import os\ndef f(): os.system('echo hi')\nreturn 0"
    
    res_safe = sandbox.execute(code_safe_efficient)
    res_unsafe = sandbox.execute(code_unsafe)
    
    r_safe = reward_engine.compute_reward(res_safe, code=code_safe_efficient)
    r_unsafe = reward_engine.compute_reward(res_unsafe, code=code_unsafe)
    
    assert r_safe > r_unsafe


def test_tier4_07_sandbox_isolation_under_adversarial_code():
    """Scenario 7: Verify sandbox isolation and non-crashing behavior under a stream of adversarial inputs."""
    sandbox = PythonSandbox()
    adversarial_inputs = [
        "def syntax_break(",
        "while True: pass",
        "1 / 0",
        "import os\nos.system('ls')",
        "x = [1] * (10**7)",
        "raise Exception('Adversarial exception')"
    ]
    
    for code in adversarial_inputs:
        res = sandbox.execute(code, timeout=0.5)
        assert isinstance(res, ExecutionResult)
        assert isinstance(res.stdout, str)
        assert isinstance(res.stderr, str)


def test_tier4_08_multi_task_forward_backward_transfer_benchmark():
    """Scenario 8: Benchmark calculating Forward Transfer (Task C) and Backward Transfer (Task A)."""
    acc_matrix = [
        [0.80, 0.20, 0.10],
        [0.75, 0.85, 0.15],
        [0.72, 0.82, 0.88]
    ]
    
    fwd_transfer = metrics.forward_transfer(acc_matrix)
    bwd_transfer = metrics.backward_transfer(acc_matrix)
    forgetting = metrics.catastrophic_forgetting(acc_matrix)
    
    assert isinstance(fwd_transfer, float)
    assert isinstance(bwd_transfer, float)
    assert isinstance(forgetting, float)


def test_tier4_09_checkpoint_recovery_and_continual_resumption(tmp_path):
    """Scenario 9: Mid-training checkpoint save, crash simulation, recovery, and training resumption."""
    agent = LoRAOnlineAgent()
    traj1 = Trajectory("t1", "p1", "r1", "", "c1", {}, {}, 0.8, {}, "2026")
    agent.train_step(traj1)
    
    ckpt_path = os.path.join(tmp_path, "resumption_checkpoint.pt")
    agent.save_checkpoint(ckpt_path)
    
    # Simulate crash & restart
    resumed_agent = LoRAOnlineAgent()
    resumed_agent.load_checkpoint(ckpt_path)
    
    traj2 = Trajectory("t2", "p2", "r2", "", "c2", {}, {}, 0.9, {}, "2026")
    step_res = resumed_agent.train_step(traj2)
    assert isinstance(step_res, dict)


def test_tier4_10_full_system_lifecycle_e2e(tmp_path):
    """Scenario 10: Complete synchronous 8-component lifecycle execution."""
    # 1. Environment (F1)
    env = OpenContinualEnv()
    obs, info = env.reset(seed=42)
    
    # 2. Execution Sandbox (F2)
    sandbox = PythonSandbox()
    code = "def solve(n):\n    return n * (n + 1) // 2"
    test_code = "assert solve(5) == 15"
    exec_res = sandbox.execute(code, test_code=test_code)
    
    # 3. Reward Pipeline (F4)
    reward_engine = RewardEngine()
    reward = reward_engine.compute_reward(exec_res, code=code)
    
    # 4. Step Environment
    obs, reward, term, trunc, step_info = env.step(code)
    
    # 5. Experience Store (F3)
    store = ExperienceStore()
    traj = Trajectory(
        trajectory_id="lifecycle_t1",
        prompt=obs.get("prompt", "Solve sum"),
        model_response=code,
        reasoning_notes="Gauss summation formula",
        generated_code=code,
        execution_output={"stdout": exec_res.stdout, "pass_rate": exec_res.pass_rate},
        feedback={"user_eval": "good"},
        reward=reward,
        regression_results={"solve_task": 1.0},
        timestamp="2026-07-25"
    )
    store.add(traj)
    
    # 6. Learning Controller (F6)
    controller = LearningController()
    action = controller.decide(traj)
    
    # 7. Continual Baseline Agent (F5)
    agent = HybridContinualAgent()
    if action != ControllerAction.IGNORE:
        agent.train_step(traj)
    
    # 8. Metrics Suite & Benchmark Runner & Plots (F7 & F8)
    history = [reward]
    learning_sp = metrics.learning_speed(history)
    assert isinstance(learning_sp, float)
    
    plot_path = os.path.join(tmp_path, "lifecycle_plot.png")
    plot_learning_curve(history, output_path=plot_path)
    assert os.path.exists(plot_path)
    
    jsonl_path = os.path.join(tmp_path, "lifecycle_store.jsonl")
    store.save_jsonl(jsonl_path)
    assert os.path.getsize(jsonl_path) > 0
