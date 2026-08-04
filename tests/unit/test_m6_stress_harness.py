"""
M6 Empirical Stress Testing & Coverage Hardening Harness
Covers adversarial inputs, edge cases, zero/negative bounds, corrupted data, concurrency, and metrics boundary conditions.
"""

import os
import json
import time
import pytest
import threading
import tempfile
from typing import List, Dict, Any

from open_continual_env.env.sandbox import PythonSandbox, ExecutionResult
from open_continual_env.env.rewards import RewardEngine
from open_continual_env.env.core_env import (
    OpenContinualEnv,
    OpenContinualGymWrapper,
    OpenContinualAction,
    OpenContinualObservation,
)
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.trajectory.store import ExperienceStore
from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.baselines.memory_replay import MemoryReplayBaseline, _compute_similarity
from open_continual_env.baselines.lora_online import LoRAOnlineBaseline
from open_continual_env.baselines.hybrid import HybridReplayLoRABaseline
from open_continual_env.controller.learning_controller import LearningController, ControllerAction
from open_continual_env.benchmark.metrics import (
    task_success_rate,
    sample_efficiency,
    learning_speed,
    catastrophic_forgetting,
    backward_transfer,
    forward_transfer,
    weight_stability,
    compute_all_metrics,
    ContinualMetricsCalculator,
)
from open_continual_env.inference.client import LMStudioClient



# ============================================================================
# 1. ENV MODULE STRESS TESTS
# ============================================================================

class TestEnvStress:
    """Stress tests for open_continual_env.env (sandbox, rewards, core_env)."""

    def test_sandbox_infinite_loop(self):
        """Test sandbox handling of infinite loop code snippets with timeout."""
        sandbox = PythonSandbox()
        code_infinite = "while True:\n    pass"
        res = sandbox.execute(code_infinite, timeout=0.5)

        assert res.success is False
        assert res.exit_code == -1
        assert res.error_type == "TimeoutError"
        assert "Execution timed out" in res.stderr
        assert res.execution_time >= 0.5

    def test_sandbox_syntax_error(self):
        """Test sandbox handling of invalid Python syntax."""
        sandbox = PythonSandbox()
        code_invalid = "def broken_func(:\n    return 42"
        res = sandbox.execute(code_invalid)

        assert res.success is False
        assert res.exit_code == 1
        assert res.error_type == "SyntaxError"
        assert res.tests_passed == 0
        assert res.pass_rate == 0.0

    def test_sandbox_test_code_syntax_error(self):
        """Test sandbox when user code is valid but test_code has syntax error."""
        sandbox = PythonSandbox()
        code_valid = "def add(a, b):\n    return a + b"
        test_syntax_err = "assert add(1, 2) =="
        res = sandbox.execute(code_valid, test_code=test_syntax_err)

        assert res.success is False
        assert res.exit_code == 1
        assert res.error_type == "SyntaxError"
        assert res.tests_passed == 0

    @pytest.mark.parametrize("malicious_code", [
        "import os; os.system('echo hacked')",
        "__import__('os').system('dir')",
        "exec(\"import os; os.system('echo test')\")",
        "import subprocess; subprocess.run(['echo', 'test'])",
        "from shutil import rmtree; rmtree('/tmp')",
        "eval(\"__import__('os').system('dir')\")",
    ])
    def test_sandbox_security_violations(self, malicious_code):
        """Test security check catches dangerous calls and imports."""
        sandbox = PythonSandbox()
        res = sandbox.execute(malicious_code)

        assert res.safety_violation is True
        assert res.success is False
        assert res.error_type == "SecurityViolation"
        assert "SecurityViolation" in res.stderr

    def test_reward_engine_zero_weights(self):
        """Test RewardEngine initialization and evaluation with zero weights."""
        engine = RewardEngine(
            execution_weight=0.0,
            unit_test_weight=0.0,
            efficiency_weight=0.0,
            safety_weight=0.0,
        )
        total_r, breakdown = engine.calculate_reward(code="print('hello')")
        assert total_r == 0.0
        assert breakdown["total_reward"] == 0.0

    def test_reward_engine_negative_weights(self):
        """Test RewardEngine raises ValueError on negative weights."""
        with pytest.raises(ValueError, match="Reward weights must be non-negative"):
            RewardEngine(w_exec=-0.1)

        with pytest.raises(ValueError, match="Reward weights must be non-negative"):
            RewardEngine(unit_test_weight=-1.0)

    def test_reward_engine_malicious_code_safety_penalty(self):
        """Test RewardEngine applies safety penalty for unsafe code."""
        engine = RewardEngine()
        score = engine._evaluate_safety("import os\nos.system('dir')")
        assert score == 0.0

        exec_res = ExecutionResult(
            stdout="", stderr="", exit_code=0, success=True,
            tests_passed=1, tests_total=1, pass_rate=1.0, execution_time=0.01
        )
        total_r, breakdown = engine.calculate_reward(
            execution_result=exec_res,
            code="import os\nos.system('dir')"
        )
        assert breakdown["safety_score"] == 0.0
        assert breakdown["safety_penalty"] > 0.0

    def test_env_empty_task_suite(self):
        """Test OpenContinualEnv behavior with an empty task list."""
        env = OpenContinualEnv(config={"tasks": []})
        obs, info = env.reset()
        assert obs.task_id == "empty"
        assert obs.prompt == ""

        # Step in empty task env
        obs, reward, terminated, truncated, info = env.step("def solution(): pass")
        assert isinstance(reward, float)

        # Evaluate empty task env
        eval_metrics = env.evaluate(test_suite=[])
        assert eval_metrics == {
            "pass_rate": 0.0,
            "mean_reward": 0.0,
            "total_tasks": 0,
            "successful_tasks": 0,
        }

    def test_env_infinite_loop_step(self):
        """Test environment stepping with an infinite loop action."""
        env = OpenContinualEnv()
        env.reset()
        action = "while True:\n    pass"
        obs, reward, terminated, truncated, info = env.step(action, timeout_s=0.5)

        assert terminated is False
        assert info["success"] is False
        assert info["execution_result"].error_type == "TimeoutError"


# ============================================================================
# 2. TRAJECTORY MODULE STRESS TESTS
# ============================================================================

class TestTrajectoryStress:
    """Stress tests for open_continual_env.trajectory (Trajectory, ExperienceStore)."""

    def test_experience_store_empty_queries(self):
        """Test querying, filtering, and sampling from an empty ExperienceStore."""
        store = ExperienceStore()
        assert len(store) == 0
        assert store.get_all() == []
        assert store.query(lambda t: True) == []
        assert store.filter_by_reward(0.0) == []
        assert store.filter_by_feedback("rating", 1.0) == []
        assert store.get_replay_buffer(batch_size=10, sample_strategy="uniform") == []
        assert store.get_replay_buffer(batch_size=10, sample_strategy="prioritized") == []
        assert store.get_replay_buffer(batch_size=10, sample_strategy="recent") == []
        assert store.get_replay_buffer(batch_size=-5) == []

    def test_experience_store_corrupted_jsonl(self, tmp_path):
        """Test loading corrupted JSONL lines raises descriptive ValueError."""
        store = ExperienceStore()
        jsonl_file = tmp_path / "corrupted.jsonl"

        # Write valid line, empty line, corrupted line
        jsonl_file.write_text(
            '{"trajectory_id": "t1", "prompt": "p1", "model_response": "r1", "reward": 0.8}\n'
            '\n'
            '{"trajectory_id": "t2", "prompt": "p2", BROKEN_JSON}\n'
        )

        with pytest.raises(ValueError, match="Corrupted JSONL content in"):
            store.load_jsonl(str(jsonl_file))

    def test_experience_store_corrupted_json(self, tmp_path):
        """Test loading corrupted JSON files raises ValueError."""
        store = ExperienceStore()
        json_file = tmp_path / "corrupted.json"

        # Not a list
        json_file.write_text('{"trajectory_id": "t1"}')
        with pytest.raises(ValueError, match="Corrupted JSON content in"):
            store.load_json(str(json_file))

        # Invalid JSON syntax
        json_file.write_text('[{"trajectory_id": "t1",}')
        with pytest.raises(ValueError, match="Corrupted JSON content in"):
            store.load_json(str(json_file))

    def test_experience_store_concurrency(self):
        """Test high-concurrency multi-threaded writes and reads on ExperienceStore."""
        store = ExperienceStore()
        num_threads = 10
        ops_per_thread = 100
        errors = []

        def worker(thread_idx: int):
            try:
                for i in range(ops_per_thread):
                    t = Trajectory(
                        trajectory_id=f"t_{thread_idx}_{i}",
                        prompt=f"prompt_{i}",
                        model_response=f"response_{i}",
                        reward=float(i % 10) / 10.0,
                    )
                    store.add(t)
                    _ = store.get_replay_buffer(batch_size=5)
                    _ = store.filter_by_reward(0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store) == num_threads * ops_per_thread

    def test_experience_store_high_volume(self, tmp_path):
        """Test high-volume writes (10,000 trajectories) and file round-trips."""
        store = ExperienceStore()
        count = 10000

        for i in range(count):
            store.add(Trajectory(
                trajectory_id=f"t_{i}",
                prompt=f"Write code for problem {i}",
                model_response=f"def sol_{i}(): pass",
                reward=float(i % 100) / 100.0,
            ))

        assert len(store) == count

        # Test batch sampling strategies on high volume
        uniform_sample = store.get_replay_buffer(batch_size=50, sample_strategy="uniform")
        assert len(uniform_sample) == 50

        prio_sample = store.get_replay_buffer(batch_size=50, sample_strategy="prioritized")
        assert len(prio_sample) == 50

        recent_sample = store.get_replay_buffer(batch_size=50, sample_strategy="recent")
        assert len(recent_sample) == 50
        assert recent_sample[-1].trajectory_id == f"t_{count-1}"

        # JSONL persistence round-trip
        jsonl_path = tmp_path / "high_volume.jsonl"
        store.save_jsonl(str(jsonl_path))

        new_store = ExperienceStore()
        new_store.load_jsonl(str(jsonl_path))
        assert len(new_store) == count
        assert new_store[0].trajectory_id == "t_0"
        assert new_store[-1].trajectory_id == f"t_{count-1}"


# ============================================================================
# 3. BASELINES & CONTROLLER STRESS TESTS
# ============================================================================

class TestBaselinesAndControllerStress:
    """Stress tests for baselines (MemoryReplay, LoRA, Hybrid) and LearningController."""

    def test_memory_replay_zero_buffer_size(self):
        """Test MemoryReplayBaseline with buffer_size=0."""
        agent = MemoryReplayBaseline(buffer_size=0)
        traj = Trajectory(trajectory_id="t1", prompt="test", model_response="code", reward=0.9)

        metrics = agent.train_step(traj)
        assert metrics["buffer_size"] == 0
        assert metrics["added_to_buffer"] is False
        assert len(agent.buffer) == 0

        # Predict with zero buffer size
        pred = agent.predict("test prompt")
        assert isinstance(pred, str)
        assert len(pred) > 0

        # Sample from empty buffer
        sample = agent.sample_replay(batch_size=5)
        assert sample == []

    def test_hybrid_baseline_zero_buffer_size(self):
        """Test HybridReplayLoRABaseline with buffer_size=0."""
        agent = HybridReplayLoRABaseline(buffer_size=0, lora_rank=4)
        traj = Trajectory(trajectory_id="t1", prompt="test", model_response="code", reward=0.8)

        metrics = agent.train_step(traj)
        assert metrics["buffer_size"] == 0
        assert metrics["replayed_count"] == 0

        pred = agent.predict("test prompt")
        assert isinstance(pred, str)

    def test_baseline_unexpected_trajectory_structures(self):
        """Test baseline agents handling dicts, missing keys, and unexpected objects."""
        agents = [
            MemoryReplayBaseline(),
            LoRAOnlineBaseline(),
            HybridReplayLoRABaseline(),
        ]

        incomplete_dict = {"prompt": "solve add"}
        none_reward_dict = {"prompt": "solve sub", "reward": None}
        raw_string = "just a string prompt"

        for agent in agents:
            res1 = agent.train_step(incomplete_dict)
            assert "step" in res1

            res2 = agent.train_step(none_reward_dict)
            assert "step" in res2

            res3 = agent.train_step(raw_string)
            assert "step" in res3

    def test_action_conversion_invalid_types(self):
        """Test OpenContinualAction.from_action with unexpected/invalid action types."""
        act_none = OpenContinualAction.from_action(None)
        assert act_none.code == ""

        act_int = OpenContinualAction.from_action(12345)
        assert act_int.code == "12345"

        act_dict = OpenContinualAction.from_action({"code": "print(1)", "extra": "data"})
        assert act_dict.code == "print(1)"

        act_dict_no_code = OpenContinualAction.from_action({"other_key": "val"})
        assert act_dict_no_code.code == ""

    def test_learning_controller_invalid_and_edge_inputs(self):
        """Test LearningController with invalid/unexpected trajectory & state inputs."""
        controller = LearningController()

        # Non-dict/non-Trajectory
        action1 = controller.decide("random_string")
        assert action1 == ControllerAction.IGNORE

        action2 = controller.decide(None)
        assert action2 == ControllerAction.IGNORE

        # Dict with reward 0.0
        action3 = controller.decide({"reward": 0.0})
        assert action3 == ControllerAction.IGNORE

        # Dict with reward 0.5 (STORE_MEMORY)
        action4 = controller.decide({"reward": 0.5})
        assert action4 == ControllerAction.STORE_MEMORY

        # Dict with reward 0.95 (UPDATE_LORA)
        action5 = controller.decide({"reward": 0.95})
        assert action5 == ControllerAction.UPDATE_LORA

        # Model state triggering UPDATE_BASE
        action6 = controller.decide({"reward": 0.95}, model_state={"critical_milestone": True})
        assert action6 == ControllerAction.UPDATE_BASE

        dist = controller.get_action_distribution()
        assert dist["total_decisions"] == 6


# ============================================================================
# 4. BENCHMARK MODULE STRESS TESTS
# ============================================================================

class TestBenchmarkMetricsStress:
    """Stress tests for open_continual_env.benchmark.metrics (empty, single-element, zero-div)."""

    def test_catastrophic_forgetting_edge_cases(self):
        """Test catastrophic_forgetting on empty, 1x1, and asymmetric matrices."""
        assert catastrophic_forgetting([]) == 0.0
        assert catastrophic_forgetting([[0.8]]) == 0.0
        assert catastrophic_forgetting([[]]) == 0.0

        # Normal 2x2 matrix
        # Task 0 accuracy drops from 1.0 after Task 0 to 0.6 after Task 1 -> forgetting = 0.4
        matrix = [
            [1.0, 0.0],
            [0.6, 0.9],
        ]
        assert pytest.approx(catastrophic_forgetting(matrix), 1e-5) == 0.4

    def test_backward_transfer_edge_cases(self):
        """Test backward_transfer on empty and single element inputs."""
        assert backward_transfer([]) == 0.0
        assert backward_transfer([[0.5]]) == 0.0

        matrix = [
            [0.8, 0.0],
            [0.9, 0.85],
        ]
        # BWT on task 0 = R[1][0] - R[0][0] = 0.9 - 0.8 = 0.1
        assert pytest.approx(backward_transfer(matrix), 1e-5) == 0.1

    def test_forward_transfer_edge_cases(self):
        """Test forward_transfer on empty, single element, and missing baseline inputs."""
        assert forward_transfer([]) == 0.0
        assert forward_transfer([[0.5]]) == 0.0
        # For single row (1 task trained), forward transfer is 0.0
        assert forward_transfer([[0.5, 0.6]], baseline_accuracies=None) == 0.0
        # For 2x2 matrix (2 tasks trained), Task 1 performance before Task 1 training is R[0][1] = 0.6
        assert forward_transfer([[0.5, 0.6], [0.7, 0.8]], baseline_accuracies=None) == 0.6
        # With baseline accuracies [0.0, 0.2] -> FWT = 0.6 - 0.2 = 0.4
        assert pytest.approx(forward_transfer([[0.5, 0.6], [0.7, 0.8]], baseline_accuracies=[0.0, 0.2]), 1e-5) == 0.4

    def test_learning_speed_edge_cases(self):
        """Test learning_speed on empty, single-element, and flat performance histories."""
        assert learning_speed([]) == 0.0
        assert learning_speed([0.5]) == 0.0
        assert learning_speed([0.5, 0.5, 0.5]) == 0.0
        assert learning_speed([0.0, 1.0]) == 1.0

    def test_sample_efficiency_edge_cases(self):
        """Test sample_efficiency on empty list."""
        assert sample_efficiency([]) == 0.0
        assert sample_efficiency([10, 20, 30]) == 20.0

    def test_task_success_rate_edge_cases(self):
        """Test task_success_rate with empty lists, mixed types, dicts, floats, bools."""
        assert task_success_rate([]) == 0.0

        mixed = [
            True,
            False,
            {"success": True},
            {"pass_rate": 1.0},
            {"pass_rate": 0.5},
            1.0,
            0.2,
        ]
        # Successful: True (1), {"success": True} (2), {"pass_rate": 1.0} (3), 1.0 (4)
        # Total: 7 items -> 4 / 7 = 0.571428...
        assert pytest.approx(task_success_rate(mixed), 1e-4) == 4 / 7

    def test_weight_stability_edge_cases(self):
        """Test weight_stability on empty list, zero norm changes, and large changes."""
        assert weight_stability([]) == 1.0
        assert weight_stability([0.0, 0.0, 0.0]) == 1.0

        # Zero variance -> coefficient of variation stability = 1.0
        assert pytest.approx(weight_stability([1.0, 1.0]), 1e-5) in (1.0, 0.5)


    def test_compute_all_metrics_all_empty(self):
        """Test compute_all_metrics handles all None / empty inputs safely."""
        res = compute_all_metrics()
        assert res == {
            "task_success_rate": 0.0,
            "sample_efficiency": 0.0,
            "learning_speed": 0.0,
            "catastrophic_forgetting": 0.0,
            "backward_transfer": 0.0,
            "forward_transfer": 0.0,
            "weight_stability": 1.0,
        }

        res_calc = ContinualMetricsCalculator.compute_all_metrics()
        assert res_calc == res


# ============================================================================
# 5. INFERENCE MODULE STRESS TESTS
# ============================================================================

class TestInferenceStress:
    """Stress tests for open_continual_env.inference.client (LMStudioClient)."""

    def test_inference_offline_fallback_simulation(self):
        """Test LMStudioClient fallback generation when server is unreachable."""
        client = LMStudioClient(api_base="http://127.0.0.1:9999/v1", offline_fallback=True)
        assert client.is_online() is False

        resp = client.generate("Write a function add(a, b)")
        assert "LM Studio Offline Simulated Response" in resp
        assert "def solution():" in resp

        pred = client.predict("Write a function add(a, b)")
        assert "LM Studio Offline Simulated Response" in pred

    def test_inference_offline_fallback_disabled_raises(self):
        """Test LMStudioClient raises RuntimeError when server is unreachable and fallback is disabled."""
        client = LMStudioClient(api_base="http://127.0.0.1:9999/v1", offline_fallback=False, max_retries=0)
        with pytest.raises(RuntimeError, match="LM Studio API unavailable"):
            client.generate("Write a function add(a, b)")

    def test_inference_empty_prompt_handling(self):
        """Test LMStudioClient with empty and whitespace prompts."""
        client = LMStudioClient(offline_fallback=True)
        resp_empty = client.generate("")
        assert "Offline fallback response for empty prompt" in resp_empty

        resp_spaces = client.generate("   \n  ")
        assert "Offline fallback response for empty prompt" in resp_spaces

