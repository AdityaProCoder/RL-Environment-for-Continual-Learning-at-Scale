"""
Unit tests for 2D Action Space, Extended Observation, observe(), reward(), update(), and step() in core_env.py.
"""

import pytest
from open_continual_env.env.core_env import (
    OpenContinualEnv,
    OpenContinualAction,
    OpenContinualObservation,
    LearningDecision,
    OpenContinualGymWrapper,
)


def test_learning_decision_enum():
    assert LearningDecision.ANSWER_ONLY.value == "answer_only"
    assert LearningDecision.STORE_MEMORY.value == "store_memory"
    assert LearningDecision.UPDATE_ADAPTER.value == "update_adapter"
    assert LearningDecision.SKIP_UPDATE.value == "skip_update"
    assert LearningDecision.REQUEST_REVIEW.value == "request_review"


def test_action_from_action_parsing():
    # String parsing
    a1 = OpenContinualAction.from_action("def foo(): pass")
    assert a1.code == "def foo(): pass"
    assert a1.learning_decision == LearningDecision.ANSWER_ONLY

    # Tuple parsing (code, decision)
    a2 = OpenContinualAction.from_action(("def bar(): pass", "store_memory"))
    assert a2.code == "def bar(): pass"
    assert a2.learning_decision == LearningDecision.STORE_MEMORY

    # Dict parsing
    a3 = OpenContinualAction.from_action({
        "code": "def baz(): return 42",
        "learning_decision": "update_adapter"
    })
    assert a3.code == "def baz(): return 42"
    assert a3.learning_decision == LearningDecision.UPDATE_ADAPTER

    # Direct OpenContinualAction
    a4 = OpenContinualAction(code="x=1", learning_decision=LearningDecision.SKIP_UPDATE)
    assert OpenContinualAction.from_action(a4) is a4


def test_env_observe():
    env = OpenContinualEnv()
    env.reset()
    obs_info = env.observe()

    assert "memory_size" in obs_info
    assert "cluster_distribution" in obs_info
    assert "adapter_versions" in obs_info
    assert "recent_rewards" in obs_info
    assert "suggested_action" in obs_info


def test_env_composite_reward():
    env = OpenContinualEnv()
    r1 = env.reward(task_reward=1.0, learning_decision=LearningDecision.ANSWER_ONLY)
    assert r1 == 1.0

    r2 = env.reward(task_reward=1.0, learning_decision=LearningDecision.UPDATE_ADAPTER)
    # 1.0 - 0.01 * 0.1 compute cost = 0.999
    assert r2 < 1.0


def test_env_update_execution():
    env = OpenContinualEnv()
    env.reset()

    class MockTraj:
        prompt = "Write add function"
        reward = 1.0

    traj = MockTraj()
    res1 = env.update(traj, LearningDecision.STORE_MEMORY)
    assert res1.get("status") == "stored"

    res2 = env.update(traj, LearningDecision.SKIP_UPDATE)
    assert res2.get("status") == "skipped"


def test_env_step_2d_flow():
    env = OpenContinualEnv()
    obs, info = env.reset()

    action = ("def add(a, b):\n    return a + b\n", LearningDecision.STORE_MEMORY)
    obs, reward, terminated, truncated, step_info = env.step(action)

    assert obs.prompt != ""
    assert "composite_reward" in step_info
    assert step_info["learning_decision_taken"] == "store_memory"
    assert "update_result" in step_info


def test_gym_wrapper_compatibility():
    gym_env = OpenContinualGymWrapper()
    obs, info = gym_env.reset()
    assert isinstance(obs, dict)

    action = {"code": "def add(a, b): return a + b", "learning_decision": 1}
    obs, reward, terminated, truncated, step_info = gym_env.step(action)
    assert "learning_decision_taken" in step_info
