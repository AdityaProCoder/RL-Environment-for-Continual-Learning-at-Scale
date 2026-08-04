"""
Unit tests for LMStudioClient / OpenAICompatibleClient inference module.
Verifies client initialization, environment variable overrides, fallback generation,
and integration with baseline agents and benchmark runners.
"""

import os
import pytest
from open_continual_env.inference.client import LMStudioClient, OpenAICompatibleClient
from open_continual_env.baselines.memory_replay import MemoryReplayBaseline
from open_continual_env.baselines.lora_online import LoRAOnlineBaseline
from open_continual_env.baselines.hybrid import HybridReplayLoRABaseline
from open_continual_env.benchmark.runner import BenchmarkRunner


def test_default_initialization():
    """Verify default initialization parameters of LMStudioClient."""
    client = LMStudioClient()
    assert client.api_base in ["http://127.0.0.1:1234/v1", "http://localhost:1234/v1"]
    assert client.model_name in ["google/gemma-4-e4b", "local-model"]
    assert client.api_key == "lm-studio"
    assert client.timeout in [10.0, 120.0]
    assert client.max_retries == 2
    assert client.offline_fallback is True



def test_explicit_config_overrides():
    """Verify explicit parameters override default configuration."""
    client = LMStudioClient(
        api_base="http://192.168.1.100:8000/v1",
        model_name="qwen2.5-coder-7b-instruct",
        api_key="test-secret-key",
        timeout=15.5,
        max_retries=4,
        offline_fallback=False,
    )
    assert client.api_base == "http://192.168.1.100:8000/v1"
    assert client.model_name == "qwen2.5-coder-7b-instruct"
    assert client.api_key == "test-secret-key"
    assert client.timeout == 15.5
    assert client.max_retries == 4
    assert client.offline_fallback is False


def test_environment_variable_overrides(monkeypatch):
    """Verify environment variables override defaults when no explicit params given."""
    monkeypatch.setenv("LM_STUDIO_API_BASE", "http://env-server:1234/v1")
    monkeypatch.setenv("LM_STUDIO_MODEL_NAME", "gemma-3-27b-it")
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")

    client = LMStudioClient()
    assert client.api_base == "http://env-server:1234/v1"
    assert client.model_name == "gemma-3-27b-it"
    assert client.api_key == "env-api-key"


def test_openai_compatible_alias():
    """Verify OpenAICompatibleClient is an alias of LMStudioClient."""
    assert OpenAICompatibleClient is LMStudioClient
    client = OpenAICompatibleClient()
    assert isinstance(client, LMStudioClient)


def test_offline_fallback_generation():
    """Verify fallback response generation when server is offline."""
    client = LMStudioClient(api_base="http://127.0.0.1:99999/v1", offline_fallback=True)
    
    # Test predict / generate
    res = client.generate("Write a function to compute factorial")
    assert isinstance(res, str)
    assert len(res) > 0
    assert "def solution():" in res

    res_pred = client.predict("Write a function to return fibonacci")
    assert isinstance(res_pred, str)
    assert len(res_pred) > 0


def test_empty_prompt_fallback():
    """Verify handling of empty prompts."""
    client = LMStudioClient()
    res = client.generate("")
    assert "empty prompt" in res.lower() or "def solution():" in res


def test_is_online_behavior():
    """Verify is_online returns a boolean without throwing exceptions."""
    client = LMStudioClient(api_base="http://127.0.0.1:99999/v1", timeout=0.5)
    online_status = client.is_online()
    assert isinstance(online_status, bool)


def test_baseline_agent_llm_client_integration():
    """Verify baseline agents integrate cleanly with LMStudioClient."""
    client = LMStudioClient()

    # Memory Replay Baseline with llm_client
    replay_agent = MemoryReplayBaseline(llm_client=client)
    pred1 = replay_agent.predict("Implement binary search")
    assert isinstance(pred1, str)
    assert len(pred1) > 0

    # LoRA Online Baseline with llm_client
    lora_agent = LoRAOnlineBaseline(llm_client=client)
    pred2 = lora_agent.predict("Implement quicksort")
    assert isinstance(pred2, str)
    assert len(pred2) > 0

    # Hybrid Baseline with llm_client
    hybrid_agent = HybridReplayLoRABaseline(llm_client=client)
    pred3 = hybrid_agent.predict("Implement merge sort")
    assert isinstance(pred3, str)
    assert len(pred3) > 0


def test_benchmark_runner_lm_studio_integration():
    """Verify BenchmarkRunner with LMStudioClient."""
    client = LMStudioClient()
    runner = BenchmarkRunner(llm_client=client)
    assert runner.llm_client is client

    runner_auto = BenchmarkRunner(use_lm_studio=True)
    assert isinstance(runner_auto.llm_client, LMStudioClient)

    # Test running a benchmark episode with LMStudioClient directly as agent
    from open_continual_env.env.core_env import OpenContinualEnv
    env = OpenContinualEnv()
    res = runner.run_benchmark(agent=client, env=env, num_episodes=1)
    assert "mean_reward" in res
    assert "trajectory_history" in res
    assert len(res["trajectory_history"]) > 0
