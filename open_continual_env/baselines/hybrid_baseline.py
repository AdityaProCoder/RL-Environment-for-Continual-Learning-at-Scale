"""
Hybrid Replay + LoRA Adapter Continual Learning Baseline.
"""

from typing import Any, Dict, List, Optional
import json
import os

from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.baselines.memory_baseline import MemoryBaseline
from open_continual_env.baselines.lora_baseline import LoRABaseline
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.trajectory.store import ExperienceStore


class HybridBaseline(BaseContinualAgent):
    def __init__(
        self,
        agent_name: str = "HybridReplayLoRABaseline",
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        experience_store: Optional[ExperienceStore] = None,
        buffer_size: int = 50,
        lora_rank: int = 4,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, llm_client=llm_client)
        self.experience_store = experience_store or ExperienceStore()
        self.buffer_size = buffer_size
        self.lora_rank = lora_rank
        self.memory_agent = MemoryBaseline(
            api_base=api_base,
            model_name=model_name,
            experience_store=self.experience_store,
            buffer_size=buffer_size,
            llm_client=llm_client,
        )
        self.lora_agent = LoRABaseline(
            api_base=api_base,
            model_name=model_name,
            lora_rank=lora_rank,
            llm_client=llm_client,
        )

    def sample_replay(self, batch_size: int = 5) -> List[Dict[str, Any]]:
        return self.memory_agent.sample_replay(batch_size=batch_size)

    def generate(self, prompt: str, task_context: str = "") -> str:
        lora_context = f"[LoRA Active Adapter Version: v{self.lora_agent.adapter_version}, Rank: {self.lora_rank}]"
        combined_context = f"{task_context}\n{lora_context}".strip()
        res = self.memory_agent.generate(prompt, task_context=combined_context)
        return res or "def code_solution(): pass\n"

    def predict(self, prompt: str, task_context: str = "") -> str:
        return self.generate(prompt, task_context=task_context)

    def train_step(self, trajectory: Any) -> Dict[str, Any]:
        self.step_count += 1
        traj_obj = self._to_trajectory(trajectory)
        mem_info = self.memory_agent.train_step(trajectory)
        lora_info = self.lora_agent.train_step(trajectory)

        replayed = self.memory_agent.sample_replay(batch_size=self.memory_agent.top_k)
        reward = getattr(traj_obj, "reward", 0.0)
        if reward is None:
            reward = 0.0
        else:
            reward = float(reward)

        loss = float(max(0.0, 1.0 - reward))
        updated = mem_info.get("updated", False) or lora_info.get("updated", False)

        return {
            "step": self.step_count,
            "buffer_size": len(self.memory_agent.buffer),
            "replayed_count": len(replayed),
            "memory_size": len(self.memory_agent.buffer),
            "lora_info": lora_info,
            "memory_info": mem_info,
            "loss": loss,
            "updated": updated,
            "trajectory_id": traj_obj.trajectory_id,
            "metrics": {
                "loss": loss,
                "buffer_size": len(self.memory_agent.buffer),
                "replayed_count": len(replayed),
            },
        }

    def update(self, trajectory: Any) -> Dict[str, Any]:
        return self.train_step(trajectory)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.memory_agent.save_checkpoint(path + ".memory.json")
        self.lora_agent.save_checkpoint(path + ".lora.json")

    def load_checkpoint(self, path: str) -> None:
        mem_path = path if not path.endswith(".memory.json") else path
        if not path.endswith(".memory.json") and not path.endswith(".lora.json"):
            mem_path = path + ".memory.json"
            lora_path = path + ".lora.json"
        else:
            mem_path = path
            lora_path = path

        if not os.path.exists(mem_path) and not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            if os.path.exists(mem_path):
                self.memory_agent.load_checkpoint(mem_path)
            elif os.path.exists(path):
                self.memory_agent.load_checkpoint(path)

            if os.path.exists(lora_path):
                self.lora_agent.load_checkpoint(lora_path)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise ValueError(f"Corrupted or invalid checkpoint file in {path}: {e}") from e


HybridReplayLoRABaseline = HybridBaseline
HybridContinualAgent = HybridBaseline
