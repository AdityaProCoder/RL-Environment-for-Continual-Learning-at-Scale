"""
OpenContinualEnv Benchmark Runner Script.

Executes multi-worker parallel benchmark evaluation comparing Memory, LoRA, and Hybrid baselines
connected to LM Studio local server endpoint.

Usage examples:
  python run_benchmark.py --num-workers 4
  python run_benchmark.py -w 8 -e 5 --model google/gemma-4-e4b
"""

import argparse
import os
import sys

# Load environment variables from .env if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip()

from benchmarks.run_empirical_benchmark import run_empirical_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenContinualEnv Multi-Worker Empirical Benchmark Evaluation"
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", "4")),
        help="Number of parallel environment workers (default: 4)",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=int(os.getenv("NUM_EPISODES", "3")),
        help="Number of episodes per benchmark task (default: 3)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
        help="LM Studio / OpenAI API base URL (default: http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B"),
        help="LM Studio / vLLM target model name (default: Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--local-gpu",
        action="store_true",
        help="Run real GPU PyTorch + PEFT/Unsloth inference and dynamic adapter hot-loading locally on GPU",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for benchmark results (default: results)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="all",
        help="Evaluation dataset: 'mbpp' (974 tasks), 'humaneval' (164 tasks), 'gsm8k' (1319 math tasks), or 'all' (default: all)",
    )
    parser.add_argument(
        "-n",
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum number of dataset tasks to evaluate (default: 100, set to 0 or 2500 for full dataset stream)",
    )
    parser.add_argument(
        "-i",
        "--eval-interval",
        type=int,
        default=10,
        help="Checkpoint interval for evaluation matrix calculation (default: 10)",
    )

    args = parser.parse_args()

    max_t = None if (args.max_tasks <= 0 or args.max_tasks >= 2500) else args.max_tasks

    print("=" * 70)
    print("Starting OpenContinualEnv Multi-Worker Empirical Benchmark Evaluation")
    print("=" * 70)
    print(f"Dataset Stream (-d / --dataset)       : {args.dataset.upper()}")
    print(f"Max Dataset Tasks (-n / --max-tasks)  : {max_t if max_t else 'FULL DATASET'}")
    print(f"Evaluation Interval (-i)              : Every {args.eval_interval} tasks")
    print(f"Parallel Workers (-w / --num-workers) : {args.num_workers}")
    print(f"Episodes per Task (-e / --episodes)    : {args.episodes}")
    print(f"Target Inference Endpoint              : {args.api_base}")
    print(f"Active Model Name                     : {args.model}")
    print(f"Local GPU PEFT Execution              : {args.local_gpu}")
    print(f"Output Directory                      : {args.output_dir}")
    print("=" * 70)

    run_empirical_benchmark(
        api_base=args.api_base,
        model_name=args.model,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        use_local_peft=args.local_gpu,
        dataset_name=args.dataset,
        max_tasks=max_t,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()
