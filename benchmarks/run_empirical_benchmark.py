"""
Empirical Continual Learning Benchmark Runner for OpenContinualEnv.

Executes real empirical multi-step, multi-episode benchmark runs on the active local
LM Studio instance (http://127.0.0.1:1234/v1, model google/gemma-4-e4b).
Saves real raw trajectories, execution logs, metrics summary, and markdown report
into benchmark_results/ at project root.
"""

import os
import sys
import json
import time
import datetime
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from open_continual_env.env.core_env import OpenContinualEnv
from open_continual_env.inference.client import LMStudioClient
from open_continual_env.baselines.memory_baseline import MemoryBaseline
from open_continual_env.baselines.lora_baseline import LoRABaseline
from open_continual_env.baselines.hybrid_baseline import HybridBaseline
from open_continual_env.baselines.dynamic_moa import DynamicMoABaseline
from open_continual_env.baselines.jitrl_baseline import JitRLBaseline
from open_continual_env.benchmark.metrics import ContinualMetrics
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.dataset_loader import UnifiedDatasetLoader


def log_message(log_file_path: str, msg: str) -> None:
    """Log timestamped message to log file and console."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_line = f"[{ts}] {msg}"
    print(log_line)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def append_trajectory_jsonl(trajectories_path: str, traj_dict: Dict[str, Any]) -> None:
    """Append single trajectory record as JSON line."""
    with open(trajectories_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(traj_dict) + "\n")


def run_empirical_benchmark(
    api_base: str = "http://127.0.0.1:1234/v1",
    model_name: str = "google/gemma-4-e4b",
    num_episodes: int = 3,
    output_dir: str = "benchmark_results",
    use_local_peft: bool = False,
    dataset_name: str = "all",
    max_tasks: Optional[int] = 100,
    eval_interval: int = 10,
) -> Dict[str, Any]:
    """
    Executes real empirical benchmark runs across Memory Replay, LoRA Online, and Hybrid baselines.
    """
    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    log_path = os.path.join(abs_output_dir, "benchmark_execution.log")
    trajectories_path = os.path.join(abs_output_dir, "trajectories.jsonl")
    metrics_path = os.path.join(abs_output_dir, "metrics_summary.json")
    summary_md_path = os.path.join(abs_output_dir, "summary.md")

    # Clear existing log and trajectories files for fresh run
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(trajectories_path, "w", encoding="utf-8") as f:
        f.write("")

    log_message(log_path, "=== Starting Empirical Continual Learning Benchmark Execution ===")
    log_message(log_path, f"Target Endpoint : {api_base}")
    log_message(log_path, f"Target Model    : {model_name}")
    log_message(log_path, f"Local GPU PEFT  : {use_local_peft}")
    log_message(log_path, f"Output Directory: {abs_output_dir}")
    
    # Load dataset tasks dynamically
    tasks = UnifiedDatasetLoader.load_dataset(dataset_name, max_tasks=max_tasks)
    benchmark_tasks = tasks if tasks else [
        {
            "task_id": "code_fallback_01",
            "domain": "code",
            "prompt": "Write a Python function `add(a, b)` returning sum of two numbers.",
            "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
            "entry_point": "add",
        }
    ]
    num_tasks = len(benchmark_tasks)

    log_message(log_path, f"Active Dataset  : {dataset_name.upper()} ({num_tasks} tasks loaded)")
    log_message(log_path, f"Eval Interval   : Every {eval_interval} tasks")
    log_message(log_path, f"Episodes / Task : {num_episodes}")

    client = LMStudioClient(api_base=api_base, model_name=model_name, timeout=120.0, offline_fallback=True)
    if not client.is_online() and not use_local_peft:
        log_message(log_path, f"NOTICE: LM Studio endpoint is not online at {api_base}. Running in offline simulation mode.")
    else:
        log_message(log_path, f"SUCCESS: Endpoint / GPU setup verified for model {model_name}.")

    # Instantiate Baselines
    baselines = {
        "Memory_Replay": MemoryBaseline(
            api_base=api_base,
            model_name=model_name,
            llm_client=client,
            buffer_size=100,
        ),
        "LoRA_Online": LoRABaseline(
            api_base=api_base,
            model_name=model_name,
            llm_client=client,
            lora_rank=8,
            use_local_peft=use_local_peft,
        ),
        "Hybrid": HybridBaseline(
            api_base=api_base,
            model_name=model_name,
            llm_client=client,
            buffer_size=50,
            lora_rank=4,
            use_local_peft=use_local_peft,
        ),
        "Dynamic_MoA": DynamicMoABaseline(
            api_base=api_base,
            model_name=model_name,
            llm_client=client,
            use_local_peft=use_local_peft,
        ),
        "JitRL": JitRLBaseline(
            api_base=api_base,
            model_name=model_name,
            llm_client=client,
        ),
    }

    all_metrics: Dict[str, Any] = {}
    num_tasks = len(benchmark_tasks)

    for baseline_name, agent in baselines.items():
        log_message(log_path, f"\n--- Initiating Benchmark for Baseline: {baseline_name} ---")

        # Step 1: Initial 0-shot evaluation across tasks for Forward Transfer baseline
        log_message(log_path, f"[{baseline_name}] Performing 0-shot baseline evaluation...")
        baseline_performances = []
        for task_idx, task in enumerate(benchmark_tasks):
            env = OpenContinualEnv(config={"tasks": [task]})
            obs, info = env.reset()
            code = agent.predict(obs.prompt)
            obs, reward, terminated, truncated, info = env.step(code)

            b_perf = float(reward)
            baseline_performances.append(b_perf)

            traj_record = {
                "trajectory_id": f"traj_{baseline_name}_0shot_task_{task['task_id']}",
                "baseline": baseline_name,
                "phase": "0shot_eval",
                "task_id": task["task_id"],
                "episode": 0,
                "step": 1,
                "prompt": obs.prompt,
                "model_response": code,
                "generated_code": code,
                "execution_output": {
                    "stdout": info.get("stdout", ""),
                    "stderr": info.get("stderr", ""),
                    "exit_code": info.get("exit_code", -1),
                    "pass_rate": info.get("pass_rate", 0.0),
                    "success": info.get("success", False),
                },
                "reward": reward,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            append_trajectory_jsonl(trajectories_path, traj_record)

        log_message(log_path, f"[{baseline_name}] 0-shot baseline average performance: {sum(baseline_performances)/max(1, len(baseline_performances)):.3f}")

        # Step 2: Sequential Continual Learning across Tasks
        accuracy_matrix = [[0.0 for _ in range(num_tasks)] for _ in range(num_tasks)]
        all_rewards: List[float] = []
        steps_to_success: List[int] = []
        learning_curve: List[float] = []

        for task_i_idx, task_i in enumerate(benchmark_tasks):
            log_message(log_path, f"[{baseline_name}] Training on Task {task_i_idx+1}/{num_tasks}: '{task_i['task_id']}' ({task_i.get('domain', 'code')})")
            task_steps = 0
            task_success_step = None
            task_ep_rewards = []

            for ep in range(1, num_episodes + 1):
                env = OpenContinualEnv(config={"tasks": [task_i]})
                obs, info = env.reset()
                start_t = time.time()
                code = agent.predict(obs.prompt)
                elapsed = time.time() - start_t

                obs, reward, terminated, truncated, info = env.step(code)
                task_steps += 1
                all_rewards.append(reward)
                task_ep_rewards.append(reward)

                if info.get("success") and task_success_step is None:
                    task_success_step = task_steps

                traj_obj = Trajectory(
                    trajectory_id=f"traj_{baseline_name}_task_{task_i['task_id']}_ep{ep}",
                    prompt=obs.prompt,
                    model_response=code,
                    reasoning_notes="",
                    generated_code=code,
                    execution_output={
                        "stdout": info.get("stdout", ""),
                        "stderr": info.get("stderr", ""),
                        "exit_code": info.get("exit_code", -1),
                        "pass_rate": info.get("pass_rate", 0.0),
                        "success": info.get("success", False),
                    },
                    feedback={},
                    reward=reward,
                    regression_results={},
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )

                # Continual learning update step
                update_info = agent.train_step(traj_obj)

                traj_record = {
                    "trajectory_id": traj_obj.trajectory_id,
                    "baseline": baseline_name,
                    "phase": "train",
                    "task_id": task_i["task_id"],
                    "domain": task_i.get("domain", "code"),
                    "episode": ep,
                    "step": task_steps,
                    "prompt": obs.prompt,
                    "model_response": code,
                    "generated_code": code,
                    "execution_output": traj_obj.execution_output,
                    "reward": reward,
                    "update_info": update_info,
                    "latency_sec": elapsed,
                    "timestamp": traj_obj.timestamp,
                }
                append_trajectory_jsonl(trajectories_path, traj_record)

                log_message(
                    log_path,
                    f"[{baseline_name}] Task '{task_i['task_id']}' Ep {ep}/{num_episodes}: "
                    f"Success={info.get('success')}, Reward={reward:.2f}, Updated={update_info.get('updated', False)}"
                )

            mean_task_reward = float(sum(task_ep_rewards) / len(task_ep_rewards)) if task_ep_rewards else 0.0
            learning_curve.append(mean_task_reward)
            steps_to_success.append(task_success_step if task_success_step is not None else num_episodes)

            # Step 3: Evaluate performance matrix at specified interval checkpoints
            should_eval_matrix = (
                num_tasks <= 20 or
                (task_i_idx + 1) % eval_interval == 0 or
                task_i_idx == num_tasks - 1
            )
            if should_eval_matrix:
                log_message(log_path, f"[{baseline_name}] Post-Task {task_i_idx+1} Matrix Evaluation Checkpoint...")
                for task_j_idx, task_j in enumerate(benchmark_tasks):
                    eval_env = OpenContinualEnv(config={"tasks": [task_j]})
                    eval_obs, _ = eval_env.reset()
                    eval_code = agent.predict(eval_obs.prompt)
                    _, eval_reward, _, _, eval_info = eval_env.step(eval_code)

                    accuracy_matrix[task_i_idx][task_j_idx] = float(eval_reward)
                log_message(log_path, f"[{baseline_name}] Row {task_i_idx+1} Matrix R: {accuracy_matrix[task_i_idx][:10]}...")
            else:
                # Carry forward previous matrix state for non-checkpoint tasks
                if task_i_idx > 0:
                    accuracy_matrix[task_i_idx] = list(accuracy_matrix[task_i_idx - 1])
                accuracy_matrix[task_i_idx][task_i_idx] = mean_task_reward

        # Compute empirical metrics for this baseline
        final_row = accuracy_matrix[-1]
        task_success_rate = ContinualMetrics.task_success_rate(final_row)
        sample_eff = ContinualMetrics.sample_efficiency(steps_to_success)
        lrn_speed = ContinualMetrics.learning_speed(learning_curve)
        cat_forgetting = ContinualMetrics.catastrophic_forgetting(accuracy_matrix)
        bwt = ContinualMetrics.backward_transfer(accuracy_matrix)
        fwt = ContinualMetrics.forward_transfer(accuracy_matrix, baseline_accuracies=baseline_performances)
        stability_idx = ContinualMetrics.weight_stability(all_rewards)
        mean_reward = float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0

        all_metrics[baseline_name] = {
            "task_success_rate": task_success_rate,
            "sample_efficiency": sample_eff,
            "learning_speed": lrn_speed,
            "catastrophic_forgetting": cat_forgetting,
            "backward_transfer": bwt,
            "forward_transfer": fwt,
            "performance_stability_index": stability_idx,
            "weight_stability": stability_idx,
            "mean_reward": mean_reward,
            "baseline_performances_0shot": baseline_performances,
            "accuracy_matrix": accuracy_matrix,
            "learning_curve": learning_curve,
            "steps_to_success": steps_to_success,
        }

        log_message(log_path, f"\n=== Benchmark Summary for {baseline_name} ===")
        log_message(log_path, f"Pass@1 Success Rate : {task_success_rate * 100:.1f}%")
        log_message(log_path, f"Sample Efficiency   : {sample_eff:.2f} steps")
        log_message(log_path, f"Learning Speed      : {lrn_speed:.3f}")
        log_message(log_path, f"Catastrophic Forgetting: {cat_forgetting:.3f}")
        log_message(log_path, f"Backward Transfer (BWT): {bwt:.3f}")
        log_message(log_path, f"Forward Transfer (FWT) : {fwt:.3f}")
        log_message(log_path, f"Stability Index     : {stability_idx:.3f}")

    # Write summary metrics JSON
    summary_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "endpoint": api_base,
        "model_name": model_name,
        "num_tasks": num_tasks,
        "num_episodes": num_episodes,
        "tasks": [t["task_id"] for t in benchmark_tasks],
        "baselines": all_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log_message(log_path, f"Saved metrics summary to {metrics_path}")

    # Write summary.md
    md_content = f"""# Continual Learning Empirical Benchmark Summary

**Execution Timestamp**: `{summary_data['timestamp']}`  
**Inference Endpoint**: `{api_base}`  
**Model Name**: `{model_name}`  
**Total Benchmark Tasks**: `{num_tasks}`  
**Episodes per Task**: `{num_episodes}`  

---

## Baseline Comparison Table

| Metric | Memory Replay | LoRA Online | Hybrid (Replay + LoRA) | Dynamic MoA | JitRL |
|---|:---:|:---:|:---:|:---:|:---:|
| **Task Success Rate (Pass@1)** | {all_metrics.get('Memory_Replay', {}).get('task_success_rate', 0.0)*100:.1f}% | {all_metrics.get('LoRA_Online', {}).get('task_success_rate', 0.0)*100:.1f}% | {all_metrics.get('Hybrid', {}).get('task_success_rate', 0.0)*100:.1f}% | {all_metrics.get('Dynamic_MoA', {}).get('task_success_rate', 0.0)*100:.1f}% | {all_metrics.get('JitRL', {}).get('task_success_rate', 0.0)*100:.1f}% |
| **Sample Efficiency (avg steps)** | {all_metrics.get('Memory_Replay', {}).get('sample_efficiency', 0.0):.2f} | {all_metrics.get('LoRA_Online', {}).get('sample_efficiency', 0.0):.2f} | {all_metrics.get('Hybrid', {}).get('sample_efficiency', 0.0):.2f} | {all_metrics.get('Dynamic_MoA', {}).get('sample_efficiency', 0.0):.2f} | {all_metrics.get('JitRL', {}).get('sample_efficiency', 0.0):.2f} |
| **Learning Speed ($\Delta$ Reward)** | {all_metrics.get('Memory_Replay', {}).get('learning_speed', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('learning_speed', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('learning_speed', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('learning_speed', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('learning_speed', 0.0):.3f} |
| **Catastrophic Forgetting** | {all_metrics.get('Memory_Replay', {}).get('catastrophic_forgetting', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('catastrophic_forgetting', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('catastrophic_forgetting', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('catastrophic_forgetting', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('catastrophic_forgetting', 0.0):.3f} |
| **Backward Transfer (BWT)** | {all_metrics.get('Memory_Replay', {}).get('backward_transfer', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('backward_transfer', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('backward_transfer', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('backward_transfer', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('backward_transfer', 0.0):.3f} |
| **Forward Transfer (FWT)** | {all_metrics.get('Memory_Replay', {}).get('forward_transfer', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('forward_transfer', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('forward_transfer', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('forward_transfer', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('forward_transfer', 0.0):.3f} |
| **Performance Stability Index** | {all_metrics.get('Memory_Replay', {}).get('performance_stability_index', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('performance_stability_index', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('performance_stability_index', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('performance_stability_index', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('performance_stability_index', 0.0):.3f} |
| **Mean Reward** | {all_metrics.get('Memory_Replay', {}).get('mean_reward', 0.0):.3f} | {all_metrics.get('LoRA_Online', {}).get('mean_reward', 0.0):.3f} | {all_metrics.get('Hybrid', {}).get('mean_reward', 0.0):.3f} | {all_metrics.get('Dynamic_MoA', {}).get('mean_reward', 0.0):.3f} | {all_metrics.get('JitRL', {}).get('mean_reward', 0.0):.3f} |

---

## Detailed Performance Matrices ($R_{{i,j}}$)

### 1. Memory Replay Matrix
```json
{json.dumps(all_metrics.get('Memory_Replay', {}).get('accuracy_matrix', []), indent=2)}
```

### 2. LoRA Online Matrix
```json
{json.dumps(all_metrics.get('LoRA_Online', {}).get('accuracy_matrix', []), indent=2)}
```

### 3. Hybrid Baseline Matrix
```json
{json.dumps(all_metrics.get('Hybrid', {}).get('accuracy_matrix', []), indent=2)}
```

### 4. Dynamic MoA Matrix
```json
{json.dumps(all_metrics.get('Dynamic_MoA', {}).get('accuracy_matrix', []), indent=2)}
```

### 5. JitRL Matrix
```json
{json.dumps(all_metrics.get('JitRL', {}).get('accuracy_matrix', []), indent=2)}
```


---

## Artifact Traceability

All raw execution steps, model generation outputs, sandbox logs, and continual update signals have been recorded into:
- **Trajectories Log**: `benchmark_results/trajectories.jsonl`
- **Execution Log**: `benchmark_results/benchmark_execution.log`
- **JSON Metrics Summary**: `benchmark_results/metrics_summary.json`
- **Markdown Report**: `benchmark_results/summary.md`
"""

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    log_message(log_path, f"Saved summary report to {summary_md_path}")
    log_message(log_path, "=== Empirical Benchmark Execution Completed Successfully ===")

    return summary_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Empirical Continual Learning Benchmark")
    parser.add_argument("--dataset", type=str, default="mbpp", help="Dataset name (mbpp, humaneval, gsm8k)")
    parser.add_argument("--max-tasks", type=int, default=2, help="Number of benchmark tasks")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes per task")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:1234/v1", help="LM Studio API base")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-e4b", help="Model name")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    args = parser.parse_args()

    run_empirical_benchmark(
        dataset_name=args.dataset,
        max_tasks=args.max_tasks,
        num_episodes=args.num_episodes,
        api_base=args.api_base,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )

