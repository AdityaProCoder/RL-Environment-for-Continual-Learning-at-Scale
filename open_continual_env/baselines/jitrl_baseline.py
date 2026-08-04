"""
JitRL: Training-Free Continual Reinforcement Learning via Retrieval and Logit Modulation.
Reference: Li et al., ICML 2026 (arXiv:2601.18510).
"""

import os
import json
import urllib.request
from typing import Any, Dict, List, Optional, Union

from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.env.core_env import LearningDecision, OpenContinualAction
from open_continual_env.memory.faiss_memory import FAISSMemory
from open_continual_env.utils.context_pruner import ContextPruner


class JitRLBaseline(BaseContinualAgent):
    """
    Training-Free Continual RL agent (JitRL).
    Maintains a non-parametric memory of past trajectories, retrieves relevant experiences
    at inference time, and modulates prompt / logit advantages on-the-fly without gradient updates.
    """

    def __init__(
        self,
        agent_name: str = "JitRL_Baseline",
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        memory: Optional[FAISSMemory] = None,
        top_k: int = 3,
        temperature: float = 0.1,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, llm_client=llm_client)
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "google/gemma-4-e4b")
        self.memory = memory or FAISSMemory()
        self.top_k = top_k
        self.temperature = temperature

    def calculate_advantage_bias(self, retrieved_trajectories: List[Any]) -> str:
        """
        Estimates advantage signal from retrieved high-reward past trajectories.
        Modulates system prompt with advantage-weighted guidance.
        """
        if not retrieved_trajectories:
            return ""

        positives = [t for t in retrieved_trajectories if getattr(t, "reward", 0.0) >= 0.8]
        negatives = [t for t in retrieved_trajectories if getattr(t, "reward", 0.0) < 0.3]

        guidance = []
        if positives:
            guidance.append("=== JIT-RL ADVANTAGE MODULATION (SUCCESS PATTERNS) ===")
            for idx, p in enumerate(positives[:2], 1):
                guidance.append(f"Success Pattern #{idx}:\nTask: {p.prompt}\nCode:\n```python\n{p.generated_code}\n```")

        if negatives:
            guidance.append("=== JIT-RL ADVANTAGE MODULATION (PITFALL PREVENTION) ===")
            for idx, n in enumerate(negatives[:1], 1):
                guidance.append(f"Avoid Failure Mode #{idx}: Do NOT write broken/failing patterns like in task: {n.prompt}")

        return "\n".join(guidance)

    def predict_action(self, prompt: str, task_context: str = "") -> OpenContinualAction:
        retrieved = self.memory.query(prompt, top_k=self.top_k)
        advantage_guidance = self.calculate_advantage_bias(retrieved)
        pruned_rag = ContextPruner.prune_retrieved_experiences(retrieved, max_tokens=512)

        full_prompt = (
            f"You are an expert Python programmer.\n"
            f"{advantage_guidance}\n\n"
            f"{pruned_rag}\n\n"
            f"Task: {prompt}\n"
            f"Write ONLY executable Python code enclosed in a standard markdown block: ```python\n[code]\n```."
        )

        code_solution = self._generate_with_llm(full_prompt, system_prompt=task_context or "You are a Python coding assistant.")
        if not code_solution:
            try:
                url = f"{self.api_base.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": task_context or "You are a Python coding assistant."},
                        {"role": "user", "content": full_prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 1536,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    code_solution = self._extract_code(content)
            except Exception:
                code_solution = "def code_solution(): pass\n"

        code = self._extract_code(code_solution or "") or "def code_solution(): pass\n"
        return OpenContinualAction(
            code=code,
            learning_decision=LearningDecision.STORE_MEMORY,  # JitRL stores experiences in non-parametric memory
            metadata={"jitrl": True, "retrieved_count": len(retrieved)},
        )

    def predict(self, prompt: str, task_context: str = "") -> str:
        action = self.predict_action(prompt, task_context=task_context)
        return action.code

    def train_step(self, trajectory: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        self.step_count += 1
        traj_obj = self._to_trajectory(trajectory)
        reward = float(getattr(traj_obj, "reward", 0.0))

        # JitRL zero-gradient update: ALWAYS store in non-parametric memory.
        # Every trajectory (even zero-reward failures) is valuable for future
        # retrieval — failed attempts serve as "pitfall prevention" guidance.
        self.memory.add(traj_obj)
        return {
            "step": self.step_count,
            "status": "jitrl_memory_stored",
            "zero_gradient": True,
            "reward": reward,
            "memory_size": len(self.memory) if hasattr(self.memory, "__len__") else None,
        }

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.memory.save(os.path.join(path, "jitrl_memory"))

    def load_checkpoint(self, path: str) -> None:
        mem_dir = os.path.join(path, "jitrl_memory")
        if os.path.exists(mem_dir):
            self.memory.load(mem_dir)
