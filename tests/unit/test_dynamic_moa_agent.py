"""
Unit tests for DynamicMoABaseline agent and learning decision policy.
"""

import os
import tempfile
from unittest.mock import patch
import pytest
from open_continual_env.baselines.dynamic_moa import DynamicMoABaseline
from open_continual_env.env.core_env import LearningDecision, OpenContinualAction
from open_continual_env.trajectory.schema import Trajectory


def test_dynamic_moa_initialization():
    agent = DynamicMoABaseline(agent_name="TestMoA")
    assert agent.agent_name == "TestMoA"
    assert agent.memory is not None
    assert agent.router is not None


def test_dynamic_moa_decide_learning():
    """Cover all 3 branches of decide_learning() by controlling novelty via mock."""
    agent = DynamicMoABaseline()

    # Branch 1: High reward (>=0.5) + high novelty (>=0.4) → UPDATE_ADAPTER
    with patch.object(agent.novelty_gate, 'compute_novelty', return_value=0.5):
        d1 = agent.decide_learning("Task A", reward=0.6)
    assert d1 == LearningDecision.UPDATE_ADAPTER

    # Branch 2: Moderate reward (>=0.3) + high novelty (>=0.6) → REQUEST_REVIEW
    with patch.object(agent.novelty_gate, 'compute_novelty', return_value=0.7):
        d2 = agent.decide_learning("Task B", reward=0.4)
    assert d2 == LearningDecision.REQUEST_REVIEW

    # Branch 3: Low novelty / low reward → STORE_MEMORY (default)
    with patch.object(agent.novelty_gate, 'compute_novelty', return_value=0.2):
        d3 = agent.decide_learning("Task C", reward=0.2)
    assert d3 == LearningDecision.STORE_MEMORY


def test_dynamic_moa_predict_action():
    agent = DynamicMoABaseline()
    action = agent.predict_action("Write a function to sum numbers")

    assert isinstance(action, OpenContinualAction)
    assert isinstance(action.code, str)
    assert "cluster_id" in action.metadata


def test_dynamic_moa_train_step():
    agent = DynamicMoABaseline()
    traj = Trajectory(
        trajectory_id="t_test",
        prompt="Write a fibonacci function in Python",
        generated_code="def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)",
        reward=1.0,
    )

    step_res = agent.train_step(traj)
    assert "decision" in step_res
    assert step_res["step"] == 1


def test_dynamic_moa_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = DynamicMoABaseline()
        traj = Trajectory(trajectory_id="t1", prompt="Write add", generated_code="def add(a,b): return a+b", reward=1.0)
        agent.memory.add(traj)

        agent.save_checkpoint(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "memory", "trajectories.json"))

        agent2 = DynamicMoABaseline()
        agent2.load_checkpoint(tmpdir)
        assert len(agent2.memory) == 1
