"""
Standalone Benchmark Execution Script for OpenContinualEnv.
Runs baseline comparisons across Memory Replay, LoRA Online, and Hybrid strategies,
and saves metrics JSON, Markdown tables, and visualization plots to benchmarks/results/.
"""

import os
import json
from typing import Dict, Any

from open_continual_env.env.core_env import OpenContinualEnv
from open_continual_env.baselines.memory_replay import MemoryReplayAgent
from open_continual_env.baselines.lora_online import LoRAOnlineAgent
from open_continual_env.baselines.hybrid import HybridContinualAgent
from open_continual_env.benchmark.runner import BenchmarkRunner
from open_continual_env.benchmark.plots import (
    plot_learning_curves,
    plot_forgetting_matrix,
    generate_metric_tables,
)


from benchmarks.run_empirical_benchmark import run_empirical_benchmark


def run_all_benchmarks(
    num_episodes: int = 3,
    output_dir: str = "benchmark_results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes empirical benchmark comparison across Memory Replay, LoRA Online, and Hybrid baselines.
    """
    api_base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    model_name = os.getenv("MODEL_NAME", "google/gemma-4-e4b")
    return run_empirical_benchmark(
        api_base=api_base,
        model_name=model_name,
        num_episodes=num_episodes,
        output_dir=output_dir,
    )


def main() -> None:
    run_all_benchmarks()


if __name__ == "__main__":
    main()
