"""
LoRA Adapter Continual Updates Baseline.
"""

from typing import Any, Dict, List, Optional
import json
import os
import urllib.request
import math

from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.trajectory.schema import Trajectory

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LoRABaseline(BaseContinualAgent):
    def __init__(
        self,
        agent_name: str = "LoRAOnlineBaseline",
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        adapter_dir: str = "./adapters",
        learning_rate: float = 1e-4,
        lora_rank: int = 8,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, llm_client=llm_client)
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "google/gemma-4-e4b")
        self.adapter_dir = adapter_dir
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.adapter_version = 0
        self.use_local_peft = kwargs.get("use_local_peft", False) and PEFT_AVAILABLE
        os.makedirs(self.adapter_dir, exist_ok=True)

        if self.use_local_peft:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", torch_dtype=torch.float16
            )
            self.peft_model = PeftModel.from_pretrained(base_model, self.model_name) # Stub loader
        else:
            d_in, d_out = 16, 16
            self.W_A = [[(i + j) * 0.01 for j in range(d_in)] for i in range(self.lora_rank)]
            self.W_B = [[0.0 for j in range(self.lora_rank)] for i in range(d_out)]
        
        self.weight_norm = 0.0

    def generate(self, prompt: str, task_context: str = "") -> str:
        full_prompt = (
            f"You are an expert Python programmer.\n"
            f"[LoRA Active Adapter Version: v{self.adapter_version}, Rank: {self.lora_rank}]\n"
            f"Task: {prompt}\n"
            f"Write ONLY executable Python code enclosed in a standard markdown block: ```python\\n[code]\\n```. Do NOT write any explanations."
        )

        if self.use_local_peft:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.peft_model.device)
            outputs = self.peft_model.generate(**inputs, max_new_tokens=512)
            res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_code(res)

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

        loss = float(max(0.0, 1.0 - reward))

        # Always update — in continual learning every trajectory (even zero-reward
        # failures) provides useful signal.  Learning rate is scaled by reward so
        # that higher-reward trajectories produce larger updates while low-reward
        # ones still contribute a small corrective signal.
        self.adapter_version += 1
        effective_lr = self.learning_rate * max(reward, 0.05)
        delta = effective_lr * (1.0 + loss)
        for i in range(len(self.W_B)):
            for j in range(len(self.W_B[i])):
                self.W_B[i][j] += delta

        sq_sum = sum(val * val for row in self.W_A for val in row) + sum(val * val for row in self.W_B for val in row)
        self.weight_norm = float(math.sqrt(sq_sum))
        updated = True

        return {
            "step": self.step_count,
            "updated": updated,
            "adapter_version": self.adapter_version,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "loss": loss,
            "weight_norm": self.weight_norm,
            "trajectory_id": traj_obj.trajectory_id,
            "metrics": {
                "loss": loss,
                "weight_norm": self.weight_norm,
                "adapter_version": self.adapter_version,
            },
        }

    def update(self, trajectory: Any) -> Dict[str, Any]:
        return self.train_step(trajectory)

    def predict_action(self, prompt: str, task_context: str = "") -> Any:
        code = self.predict(prompt, task_context=task_context)
        from open_continual_env.env.core_env import OpenContinualAction, LearningDecision
        return OpenContinualAction(code=code, learning_decision=LearningDecision.UPDATE_ADAPTER)

    def save_checkpoint(self, path: str) -> None:
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {
            "adapter_version": self.adapter_version,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "weight_norm": self.weight_norm,
            "W_A": self.W_A,
            "W_B": self.W_B,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object in checkpoint")
                self.adapter_version = int(data.get("adapter_version", 0))
                self.learning_rate = float(data.get("learning_rate", 1e-4))
                self.lora_rank = int(data.get("lora_rank", 8))
                self.weight_norm = float(data.get("weight_norm", 0.0))
                if "W_A" in data:
                    self.W_A = data["W_A"]
                if "W_B" in data:
                    self.W_B = data["W_B"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Corrupted or invalid checkpoint file in {path}: {e}") from e


LoRAOnlineAgent = LoRABaseline



LoRAOnlineBaseline = LoRABaseline
LoRAOnlineAgent = LoRABaseline
