"""
Memory / Experience Replay Continual Learning Baseline.
"""

from typing import Any, Dict, List, Optional
import json
import os
import urllib.request

from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.trajectory.store import ExperienceStore


def _compute_similarity(text1: str, text2: str) -> float:
    t1 = str(text1 or "").lower().split()
    t2 = str(text2 or "").lower().split()
    if not t1 or not t2:
        return 0.0
    s1, s2 = set(t1), set(t2)
    return len(s1.intersection(s2)) / float(len(s1.union(s2)))


class MemoryBaseline(BaseContinualAgent):
    def __init__(
        self,
        agent_name: str = "MemoryReplayBaseline",
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        experience_store: Optional[ExperienceStore] = None,
        buffer_size: int = 100,
        top_k: int = 3,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, llm_client=llm_client)
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "google/gemma-4-e4b")
        self.experience_store = experience_store or ExperienceStore()
        self.buffer_size = buffer_size
        self.top_k = top_k

    @property
    def buffer(self) -> List[Trajectory]:
        return self.experience_store.trajectories

    def sample_replay(self, batch_size: int = 5) -> List[Dict[str, Any]]:
        if self.buffer_size == 0 or len(self.experience_store) == 0:
            return []
        trajs = self.experience_store.get_replay_buffer(batch_size=batch_size, sample_strategy="recent")
        return [t.to_dict() if isinstance(t, Trajectory) else dict(t) for t in trajs]

    def generate(self, prompt: str, task_context: str = "") -> str:
        replay_samples = self.experience_store.get_replay_buffer(batch_size=self.top_k, sample_strategy="recent")

        memory_context = ""
        if replay_samples:
            memory_context = "\n--- RELEVANT PAST EXPERIENCES ---\n"
            for idx, traj in enumerate(replay_samples, 1):
                memory_context += f"Example {idx}:\nPrompt: {traj.prompt}\nSolution:\n```python\n{traj.generated_code}\n```\n"
            memory_context += "--- END PAST EXPERIENCES ---\n\n"

        full_prompt = (
            f"You are an expert Python programmer.\n"
            f"{memory_context}"
            f"Task: {prompt}\n"
            f"Write ONLY executable Python code enclosed in a standard markdown block: ```python\\n[code]\\n```. Do NOT write any explanations."
        )

        llm_resp = self._generate_with_llm(full_prompt, system_prompt=task_context)
        if llm_resp:
            extracted = self._extract_code(llm_resp)
            if extracted:
                return extracted

        try:
            url = f"{self.api_base.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a Python coding assistant. Output python code only inside ```python block."},
                    {"role": "user", "content": full_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1536,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                code = self._extract_code(content)
                if code:
                    return code
        except Exception:
            pass

        return "def code_solution(): pass\n"

    def predict(self, prompt: str, task_context: str = "") -> str:
        res = self.generate(prompt, task_context=task_context)
        return res or "def code_solution(): pass\n"

    def train_step(self, trajectory: Any) -> Dict[str, Any]:
        self.step_count += 1
        traj_obj = self._to_trajectory(trajectory)
        reward = getattr(traj_obj, "reward", 0.0)
        if reward is None:
            reward = 0.0
        else:
            reward = float(reward)

        added = False
        if self.buffer_size > 0 and reward > 0.0:
            self.experience_store.add(traj_obj)
            added = True
            if len(self.experience_store) > self.buffer_size:
                self.experience_store.trajectories = self.experience_store.trajectories[-self.buffer_size:]

        loss = float(max(0.0, 1.0 - reward))
        return {
            "step": self.step_count,
            "buffer_size": len(self.experience_store),
            "added_to_buffer": added,
            "loss": loss,
            "updated": added,
            "trajectory_id": traj_obj.trajectory_id,
            "metrics": {
                "loss": loss,
                "buffer_size": len(self.experience_store),
                "reward": reward,
            },
        }

    def update(self, trajectory: Any) -> Dict[str, Any]:
        return self.train_step(trajectory)

    def predict_action(self, prompt: str, task_context: str = "") -> Any:
        code = self.predict(prompt, task_context=task_context)
        from open_continual_env.env.core_env import OpenContinualAction, LearningDecision
        return OpenContinualAction(code=code, learning_decision=LearningDecision.STORE_MEMORY)

    def save_checkpoint(self, path: str) -> None:
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.experience_store.save_json(path)

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            self.experience_store.load_json(path, clear_existing=True)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid checkpoint file in {path}: {e}") from e


MemoryReplayBaseline = MemoryBaseline
MemoryReplayAgent = MemoryBaseline

