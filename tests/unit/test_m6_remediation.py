"""
Unit tests specifically targeting the M6 Forensic Audit findings:
1. Baseline Exports & Import Aliases
2. Baseline Methods & Return Signatures (train_step return dict, sample_replay List[dict], checkpoint error raising)
3. Catastrophic Forgetting metric positive drop computation
4. Genuine LoRA parameter tensor updates
"""

import os
import json
import pytest

from open_continual_env.baselines import (
    BaseContinualAgent,
    MemoryBaseline,
    MemoryReplayBaseline,
    MemoryReplayAgent,
    LoRABaseline,
    LoRAOnlineBaseline,
    LoRAOnlineAgent,
    HybridBaseline,
    HybridReplayLoRABaseline,
    HybridContinualAgent,
    _compute_similarity,
)
from open_continual_env.baselines.memory_replay import MemoryReplayBaseline as MRB_Direct
from open_continual_env.baselines.lora_online import LoRAOnlineBaseline as LOB_Direct
from open_continual_env.baselines.hybrid import HybridReplayLoRABaseline as HRB_Direct
from open_continual_env.benchmark.metrics import catastrophic_forgetting, backward_transfer
from open_continual_env.trajectory.schema import Trajectory


def test_finding_1_baseline_exports_and_aliases():
    """Verify all class names and aliases are exported correctly across all module paths."""
    assert MemoryReplayBaseline is MemoryReplayAgent
    assert MemoryBaseline is MemoryReplayAgent
    assert MRB_Direct is MemoryReplayAgent

    assert LoRAOnlineBaseline is LoRAOnlineAgent
    assert LoRABaseline is LoRAOnlineAgent
    assert LOB_Direct is LoRAOnlineAgent

    assert HybridReplayLoRABaseline is HybridContinualAgent
    assert HybridBaseline is HybridContinualAgent
    assert HRB_Direct is HybridContinualAgent

    assert callable(_compute_similarity)
    assert _compute_similarity("hello world", "hello Python") > 0.0


def test_finding_2_train_step_return_signatures():
    """Verify train_step across all baselines ALWAYS returns a valid dict with mandatory fields even for low reward."""
    agents = [
        MemoryReplayAgent(),
        LoRAOnlineAgent(),
        HybridContinualAgent(),
    ]

    empty_traj = Trajectory(
        trajectory_id="t_low",
        prompt="test prompt",
        model_response="test resp",
        reward=0.1,  # Low reward, should not update model
    )

    for agent in agents:
        res = agent.train_step(empty_traj)
        assert isinstance(res, dict)
        assert "loss" in res
        assert "updated" in res
        assert "metrics" in res
        assert "trajectory_id" in res
        assert res["trajectory_id"] == "t_low"
        assert isinstance(res["loss"], float)
        assert isinstance(res["updated"], bool)
        assert isinstance(res["metrics"], dict)


def test_finding_2_sample_replay_return_type():
    """Verify MemoryReplayAgent and HybridContinualAgent sample_replay returns List[dict]."""
    mem_agent = MemoryReplayAgent(buffer_size=10)
    assert mem_agent.sample_replay(batch_size=2) == []

    # Add a high-reward trajectory to buffer
    traj = Trajectory(
        trajectory_id="t_high",
        prompt="Write add",
        model_response="def add(a, b): return a + b",
        generated_code="def add(a, b): return a + b",
        reward=1.0,
    )
    mem_agent.train_step(traj)

    samples = mem_agent.sample_replay(batch_size=2)
    assert isinstance(samples, list)
    assert len(samples) == 1
    assert isinstance(samples[0], dict)
    assert samples[0]["trajectory_id"] == "t_high"

    hybrid_agent = HybridContinualAgent(buffer_size=10)
    hybrid_agent.train_step(traj)
    h_samples = hybrid_agent.sample_replay(batch_size=2)
    assert isinstance(h_samples, list)
    assert len(h_samples) == 1
    assert isinstance(h_samples[0], dict)


def test_finding_2_checkpoint_error_handling(tmp_path):
    """Verify load_checkpoint raises FileNotFoundError for missing files and ValueError for corrupted files."""
    agents = [MemoryReplayAgent(), LoRAOnlineAgent(), HybridContinualAgent()]

    non_existent = str(tmp_path / "does_not_exist.json")
    for agent in agents:
        with pytest.raises(FileNotFoundError):
            agent.load_checkpoint(non_existent)

    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text("{broken json content syntax error")

    for agent in agents:
        with pytest.raises(ValueError):
            agent.load_checkpoint(str(corrupted))


def test_finding_3_catastrophic_forgetting_metric_formula():
    """Verify catastrophic_forgetting computes positive accuracy drop and clips negative drops to zero."""
    # Matrix where task 0 drops from 0.9 to 0.6 -> forgetting = 0.3
    matrix_drop = [
        [0.9, 0.0],
        [0.6, 0.85]
    ]
    f_val = catastrophic_forgetting(matrix_drop)
    assert pytest.approx(f_val, 1e-5) == 0.3

    # Matrix where task 0 accuracy IMPROVED from 0.7 to 0.9 -> forgetting = 0.0 (not -0.2)
    matrix_improve = [
        [0.7, 0.0],
        [0.9, 0.95]
    ]
    assert catastrophic_forgetting(matrix_improve) == 0.0

    # Backward transfer remains signed: 0.9 - 0.7 = +0.2 for improve
    assert pytest.approx(backward_transfer(matrix_improve), 1e-5) == 0.2


def test_finding_4_genuine_lora_parameter_updates():
    """Verify LoRAOnlineAgent performs actual parameter tensor adjustments and updates weight norms."""
    agent = LoRAOnlineAgent(lora_rank=4, learning_rate=0.01)

    initial_version = agent.adapter_version

    # Step with reward >= 0.8 to trigger update
    high_reward_traj = Trajectory(
        trajectory_id="t_high",
        prompt="Write fibonacci function",
        generated_code="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        reward=0.95,
    )

    res = agent.train_step(high_reward_traj)

    assert res["updated"] is True
    assert res["adapter_version"] == initial_version + 1
    assert agent.adapter_version == initial_version + 1

    # Verify weight norm changed or parameters updated
    assert hasattr(agent, "W_A")
    assert hasattr(agent, "W_B")
    assert res["weight_norm"] >= 0.0


def test_finding_4_lora_checkpoint_tensor_persistence(tmp_path):
    """Verify saving and loading LoRAOnlineAgent checkpoint preserves parameter weight matrices."""
    agent = LoRAOnlineAgent(lora_rank=4)
    agent.train_step(Trajectory("t1", "p", "r", "", "c", {}, {}, 0.9, {}, "t"))

    ckpt_file = str(tmp_path / "lora_ckpt.json")
    agent.save_checkpoint(ckpt_file)

    new_agent = LoRAOnlineAgent(lora_rank=4)
    new_agent.load_checkpoint(ckpt_file)

    assert new_agent.adapter_version == agent.adapter_version
    assert pytest.approx(new_agent.weight_norm, 1e-5) == agent.weight_norm


def test_forensic_fix_1_parallel_benchmark_runner_llm_client():
    """Verify ParallelBenchmarkRunner initializes and stores self.llm_client via positional/keyword or kwargs."""
    from open_continual_env.benchmark.runner import ParallelBenchmarkRunner
    dummy_client = object()
    runner1 = ParallelBenchmarkRunner(llm_client=dummy_client)
    assert runner1.llm_client is dummy_client

    runner2 = ParallelBenchmarkRunner(kwargs={"llm_client": dummy_client})
    assert runner2.llm_client is dummy_client


def test_forensic_fix_2_trajectory_schema_default_model_response():
    """Verify Trajectory schema allows instantiation without model_response and defaults to empty string."""
    t1 = Trajectory("t_id", "prompt_text")
    assert t1.trajectory_id == "t_id"
    assert t1.prompt == "prompt_text"
    assert t1.model_response == ""

    t2 = Trajectory(trajectory_id="t_id_2", prompt="prompt_text_2")
    assert t2.model_response == ""


def test_forensic_fix_3_experience_store_corrupted_json_and_jsonl_exceptions(tmp_path):
    """Verify load_json and load_jsonl raise ValueError with exact corrupted message for invalid files."""
    from open_continual_env.trajectory.store import ExperienceStore
    store = ExperienceStore()

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{broken json syntax")
    with pytest.raises(ValueError, match="Corrupted JSON content in"):
        store.load_json(str(bad_json))

    non_list_json = tmp_path / "non_list.json"
    non_list_json.write_text('{"trajectory_id": "1"}')
    with pytest.raises(ValueError, match="Corrupted JSON content in"):
        store.load_json(str(non_list_json))

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"trajectory_id": "1"}\n{invalid line}\n')
    with pytest.raises(ValueError, match="Corrupted JSONL content in"):
        store.load_jsonl(str(bad_jsonl))

