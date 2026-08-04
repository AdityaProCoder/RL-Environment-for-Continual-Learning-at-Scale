"""
Unit test suite verifying Part A critical bug fixes, Part B moderate fixes,
and Part C/D SOTA techniques & novel mathematical contributions.
"""

import pytest
import numpy as np
from open_continual_env.env.core_env import OpenContinualEnv, LearningDecision
from open_continual_env.env.rewards import RewardEngine
from open_continual_env.benchmark.metrics import ContinualMetrics
from open_continual_env.training.online_trainer import OnlineTrainer
from open_continual_env.utils.novelty_gate import NoveltyGate
from open_continual_env.trajectory.store import ExperienceStore
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.baselines.jitrl_baseline import JitRLBaseline
from open_continual_env.training.grpo_trainer import GRPOTrainer
from open_continual_env.utils.prompt_optimizer import PromptOptimizer
from open_continual_env.benchmark.interference import GradientInterferenceMatrix
from open_continual_env.utils.itls import ITLSSignal
from open_continual_env.utils.geometry_conflict import GeometryConflict


def test_bug_1_and_2_online_trainer_formatting():
    trainer = OnlineTrainer()
    t = Trajectory(trajectory_id="t1", prompt="Add", generated_code="def add(a,b): return a+b", reward=1.0)
    info = trainer.queue("c1", t, min_batch_size=1)
    assert info["status"] in ("queued", "submitting")


def test_bug_3_evaluate_math_domain():
    env = OpenContinualEnv(config={
        "tasks": [
            {
                "task_id": "m1",
                "domain": "math",
                "prompt": "What is 2+3?",
                "reference_answer": "5",
            }
        ]
    })
    res = env.evaluate(model=lambda p: "#### 5")
    assert res["pass_rate"] == 1.0
    assert res["successful_tasks"] == 1


def test_bug_4_weight_stability_cv():
    # Constant rewards -> 0 variation -> stability 1.0
    s1 = ContinualMetrics.weight_stability([1.0, 1.0, 1.0, 1.0])
    assert s1 == 1.0

    # High variation -> lower stability
    s2 = ContinualMetrics.weight_stability([1.0, 0.0, 1.0, 0.0])
    assert s2 < 1.0


def test_b1_gsm8k_fraction_and_comma():
    engine = RewardEngine()
    r1, info1 = engine.compute_math_reward("The answer is 1,000", "1000")
    assert info1["match"] is True

    r2, info2 = engine.compute_math_reward("Result is 3/4", "0.75")
    assert info2["match"] is True


def test_b2_novelty_gate_high_novelty_override():
    gate = NoveltyGate(memory=None)  # memory=None gives novelty=1.0
    should_up = gate.should_update(prompt="Unseen novel hard task", reward=0.1)
    assert should_up is True


def test_c1_jitrl_baseline():
    agent = JitRLBaseline()
    action = agent.predict_action("Write sort function")
    assert action.learning_decision == LearningDecision.STORE_MEMORY

    t = Trajectory(trajectory_id="t1", prompt="Write sort", generated_code="def sort(): pass", reward=1.0)
    res = agent.train_step(t)
    assert res["zero_gradient"] is True


def test_c2_freshness_aware_per():
    store = ExperienceStore()
    store.add(Trajectory(trajectory_id="t1", prompt="Task 1", generated_code="", reward=0.5))
    store.add(Trajectory(trajectory_id="t2", prompt="Task 2", generated_code="", reward=0.9))

    batch = store.get_replay_buffer(batch_size=1, sample_strategy="freshness_aware")
    assert len(batch) == 1


def test_c3_and_c7_grpo_and_frpo():
    grpo = GRPOTrainer(group_size=3)
    res = grpo.train_step(
        prompt="Write add",
        generated_codes=["code1", "code2", "code3"],
        rewards=[1.0, 0.5, 0.0]
    )
    assert res["status"] == "success"
    assert len(res["advantages"]) == 3
    assert res["advantages"][0] > res["advantages"][2]


def test_c5_prompt_optimizer():
    opt = PromptOptimizer()
    prefix = opt.optimize_prefix("t1", "Write code", {"success": False, "stderr": "SyntaxError"})
    assert "SyntaxError" in prefix


def test_d1_gradient_interference_matrix():
    gim = GradientInterferenceMatrix(num_tasks=3)
    g_i = np.array([1.0, 0.0, 0.0])
    g_j = np.array([-1.0, 0.0, 0.0])
    cos_val = gim.compute_gradient_cosine(g_i, g_j)
    assert cos_val == -1.0

    gim.update_pair(0, 1, cos_val)
    score = gim.get_interference_score(1)
    assert score == 1.0


def test_d2_learning_efficiency_frontier():
    auc = ContinualMetrics.compute_learning_efficiency_frontier([0.2, 0.5, 0.8], [0.5, 0.7, 0.9])
    assert auc > 0.0


def test_d3_itls_signal():
    signal = ITLSSignal.compute_itls([-0.1, -0.2], [0.5, 0.6])
    assert 0.0 <= signal <= 1.0


def test_c9_geometry_conflict():
    cov1 = np.eye(3, dtype=np.float32)
    cov2 = np.eye(3, dtype=np.float32)
    dist = GeometryConflict.compute_wasserstein_distance(cov1, cov2)
    assert dist == 0.0
