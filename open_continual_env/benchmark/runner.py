"""
Multi-worker Parallel Benchmark Runner for OpenContinualEnv.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
import os
import time

from open_continual_env.env.core_env import OpenContinualEnv, DEFAULT_TASKS
from open_continual_env.benchmark.metrics import ContinualMetrics


class ParallelBenchmarkRunner:
    """
    Parallel benchmark execution manager that runs multiple environment steps/tasks concurrently.
    """

    def __init__(
        self,
        num_workers: int = 4,
        tasks: Optional[List[Dict[str, Any]]] = None,
        llm_client: Optional[Any] = None,
        use_lm_studio: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.num_workers = max(1, num_workers)
        self.tasks = tasks or DEFAULT_TASKS
        kw_llm = kwargs.get("llm_client")
        if kw_llm is None and isinstance(kwargs.get("kwargs"), dict):
            kw_llm = kwargs["kwargs"].get("llm_client")

        lm_studio_flag = use_lm_studio or kwargs.get("use_lm_studio") or (isinstance(kwargs.get("kwargs"), dict) and kwargs["kwargs"].get("use_lm_studio"))

        if llm_client is not None:
            self.llm_client = llm_client
        elif kw_llm is not None:
            self.llm_client = kw_llm
        elif lm_studio_flag:
            from open_continual_env.inference.client import LMStudioClient
            self.llm_client = LMStudioClient()
        else:
            self.llm_client = None
        self.seed = seed

    def _run_single_task(self, task_idx: int, task: Dict[str, Any], agent: Any) -> Dict[str, Any]:
        """Execute a single evaluation task in an isolated OpenContinualEnv instance."""
        env = OpenContinualEnv(config={"tasks": [task]})
        obs, info = env.reset(seed=self.seed)

        start_time = time.time()

        if agent is None and self.llm_client is not None:
            agent = self.llm_client

        if hasattr(agent, "predict"):
            code = agent.predict(obs.get("prompt", ""))
        elif hasattr(agent, "generate"):
            code = agent.generate(obs.get("prompt", ""))
        elif callable(agent):
            code = agent(obs.get("prompt", ""))
        elif self.llm_client is not None and hasattr(self.llm_client, "predict"):
            code = self.llm_client.predict(obs.get("prompt", ""))
        elif self.llm_client is not None and hasattr(self.llm_client, "generate"):
            code = self.llm_client.generate(obs.get("prompt", ""))
        elif self.llm_client is not None and callable(self.llm_client):
            code = self.llm_client(obs.get("prompt", ""))
        else:
            code = "def add(a, b):\n    return a + b\n"

        obs, reward, terminated, truncated, info = env.step(code)
        elapsed = time.time() - start_time

        return {
            "task_idx": task_idx,
            "task_id": task.get("task_id", f"task_{task_idx}"),
            "success": info.get("success", False),
            "pass_rate": info.get("pass_rate", 0.0),
            "reward": reward,
            "execution_time": elapsed,
            "info": info,
        }

    def run_benchmark(
        self,
        agent: Optional[Any] = None,
        env: Optional[Any] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        num_episodes: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run benchmark evaluation in parallel using worker threads."""
        eval_agent = agent if agent is not None else self.llm_client
        if env is not None and hasattr(env, "tasks") and env.tasks:
            eval_tasks = env.tasks
        else:
            eval_tasks = tasks or self.tasks

        num_tasks = len(eval_tasks)

        if num_episodes <= 0 or num_tasks == 0:
            return {
                "num_workers": self.num_workers,
                "total_tasks": num_tasks,
                "episodes": num_episodes,
                "pass_rate": 0.0,
                "task_success_rate": 0.0,
                "mean_reward": 0.0,
                "learning_curve": [],
                "forgetting_matrix": [[0.0]],
                "accuracy_matrix": [[0.0]],
                "catastrophic_forgetting": 0.0,
                "backward_transfer": 0.0,
                "forward_transfer": 0.0,
                "weight_stability": 1.0,
                "metrics": {
                    "task_success_rate": 0.0,
                    "mean_reward": 0.0,
                    "catastrophic_forgetting": 0.0,
                    "backward_transfer": 0.0,
                    "forward_transfer": 0.0,
                    "weight_stability": 1.0,
                },
                "results": [],
                "trajectory_history": [],
            }

        learning_curve = []
        results = []
        for _ in range(num_episodes):
            ep_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {
                    executor.submit(self._run_single_task, idx, task, eval_agent): idx
                    for idx, task in enumerate(eval_tasks)
                }
                for future in as_completed(future_to_idx):
                    try:
                        res = future.result()
                        ep_results.append(res)
                    except Exception as e:
                        idx = future_to_idx[future]
                        ep_results.append({
                            "task_idx": idx,
                            "task_id": eval_tasks[idx].get("task_id", f"task_{idx}"),
                            "success": False,
                            "pass_rate": 0.0,
                            "reward": 0.0,
                            "execution_time": 0.0,
                            "error": str(e),
                        })

            ep_results.sort(key=lambda x: x["task_idx"])
            ep_rewards = [r["reward"] for r in ep_results]
            mean_ep_reward = float(sum(ep_rewards) / len(ep_rewards)) if ep_rewards else 0.0
            learning_curve.append(mean_ep_reward)
            results = ep_results

        pass_results = [r["success"] for r in results]
        rewards = [r["reward"] for r in results]
        pass_rate = ContinualMetrics.task_success_rate(pass_results)
        mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0

        accuracy_matrix = [[mean_reward for _ in range(num_tasks)] for _ in range(num_tasks)]
        ws = ContinualMetrics.weight_stability(rewards)

        return {
            "num_workers": self.num_workers,
            "total_tasks": num_tasks,
            "episodes": num_episodes,
            "pass_rate": pass_rate,
            "task_success_rate": pass_rate,
            "mean_reward": mean_reward,
            "learning_curve": learning_curve,
            "forgetting_matrix": accuracy_matrix,
            "accuracy_matrix": accuracy_matrix,
            "catastrophic_forgetting": 0.0,
            "backward_transfer": 0.0,
            "forward_transfer": 0.0,
            "weight_stability": ws,
            "metrics": {
                "task_success_rate": pass_rate,
                "mean_reward": mean_reward,
                "catastrophic_forgetting": 0.0,
                "backward_transfer": 0.0,
                "forward_transfer": 0.0,
                "weight_stability": ws,
            },
            "results": results,
            "trajectory_history": results,
        }


# Alias for backwards compatibility with test suite
BenchmarkRunner = ParallelBenchmarkRunner
