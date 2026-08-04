"""
Dynamic Mixture of Adapters (MoA) Baseline Agent with 2D RL Action Space support.
"""

import os
import json
import urllib.request
from typing import Any, Dict, Optional, List, Union

from open_continual_env.baselines.base import BaseContinualAgent
from open_continual_env.env.core_env import LearningDecision, OpenContinualAction
from open_continual_env.memory.faiss_memory import FAISSMemory
from open_continual_env.routing.embedding_router import EmbeddingRouter
from open_continual_env.routing.adapter_registry import AdapterRegistry
from open_continual_env.training.online_trainer import OnlineTrainer
from open_continual_env.utils.novelty_gate import NoveltyGate
from open_continual_env.utils.context_pruner import ContextPruner

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class DynamicMoABaseline(BaseContinualAgent):
    """
    Agent that routes requests to dynamically loaded LoRA adapters based on task semantics
    and makes policy-driven 2D learning decisions (Task Action + Learning Action).
    """

    def __init__(
        self,
        agent_name: str = "DynamicMoABaseline",
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        adapter_dir: str = "./adapters",
        use_local_peft: bool = False,
        memory: Optional[FAISSMemory] = None,
        router: Optional[EmbeddingRouter] = None,
        registry: Optional[AdapterRegistry] = None,
        trainer: Optional[OnlineTrainer] = None,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(agent_name=agent_name, llm_client=llm_client)
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "google/gemma-4-e4b")
        self.adapter_dir = adapter_dir
        self.use_local_peft = use_local_peft and PEFT_AVAILABLE

        self.memory = memory or FAISSMemory()
        self.router = router or EmbeddingRouter()
        self.registry = registry or AdapterRegistry(adapter_dir=adapter_dir)
        self.trainer = trainer or OnlineTrainer(adapter_dir=adapter_dir, adapter_registry=self.registry)
        self.novelty_gate = NoveltyGate(memory=self.memory)

        self.model = None
        self.tokenizer = None
        self.active_adapters: List[str] = []

        if self.use_local_peft:
            self._init_local_model()

    def _init_local_model(self) -> None:
        """Loads base model for local PEFT routing."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
            self._load_available_adapters()
        except Exception:
            self.use_local_peft = False

    def _load_available_adapters(self) -> None:
        """Discovers trained adapters in the adapter directory."""
        if not os.path.exists(self.adapter_dir) or self.model is None:
            return

        for cluster_id in os.listdir(self.adapter_dir):
            adapter_path = os.path.join(self.adapter_dir, cluster_id)
            if os.path.isdir(adapter_path):
                try:
                    self.model.load_adapter(adapter_path, adapter_name=cluster_id)
                    self.active_adapters.append(cluster_id)
                except Exception:
                    pass

    def decide_learning(
        self,
        prompt: str,
        reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> LearningDecision:
        """
        Policy network over learning decisions.

        Every trajectory contributes to continual learning — the default is
        STORE_MEMORY.  UPDATE_ADAPTER is reserved for high-reward novel
        trajectories; REQUEST_REVIEW is rare (high novelty + medium reward
        where the signal is ambiguous).
        """
        novelty = self.novelty_gate.compute_novelty(prompt)

        # High reward + high novelty → actively update the adapter
        if reward >= 0.5 and novelty >= 0.4:
            return LearningDecision.UPDATE_ADAPTER

        # Ambiguous signal: high novelty but moderate reward — flag for review
        if novelty >= 0.6 and reward >= 0.3:
            return LearningDecision.REQUEST_REVIEW

        # Default: store the trajectory for future retrieval
        return LearningDecision.STORE_MEMORY


    def predict_action(self, prompt: str, task_context: str = "") -> OpenContinualAction:
        """Generates code and decides learning action (2D action)."""
        cluster_id = self.router.get_cluster_id(prompt)
        past_experiences = self.memory.query(prompt, top_k=3)
        rag_context = ContextPruner.prune_retrieved_experiences(past_experiences, max_tokens=768)

        code_solution = self._generate_solution(prompt, cluster_id, rag_context, task_context)
        return OpenContinualAction(
            code=code_solution,
            learning_decision=LearningDecision.ANSWER_ONLY,  # Default action during inference
            metadata={"cluster_id": cluster_id, "rag_used": bool(rag_context)},
        )

    def _generate_solution(self, prompt: str, cluster_id: str, rag_context: str, task_context: str) -> str:
        # Check LLM client first
        if self.llm_client is not None:
            combined_prompt = f"{rag_context}\n\nTask: {prompt}" if rag_context else prompt
            res = self._generate_with_llm(combined_prompt, system_prompt=task_context or "You are an expert Python programmer.")
            if res:
                code = self._extract_code(res)
                if code:
                    return code

        if self.use_local_peft and self.model is not None and cluster_id in self.active_adapters:
            try:
                self.model.set_adapter(cluster_id)
                full_p = f"{rag_context}\n\nTask: {prompt}" if rag_context else prompt
                inputs = self.tokenizer(full_p, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**inputs, max_new_tokens=512)
                res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._extract_code(res)
            except Exception:
                pass

        # Fallback to direct HTTP API
        full_prompt = (
            f"You are an expert Python programmer.\n"
            f"[Active MoA Cluster: {cluster_id}]\n"
            f"{rag_context}\n\n"
            f"Task: {prompt}\n"
            f"Write ONLY executable Python code enclosed in a standard markdown block: ```python\n[code]\n```. Do NOT write any explanations."
        )

        try:
            url = f"{self.api_base.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": task_context or "You are a Python coding assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                "temperature": 0.1,
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
                return self._extract_code(content) or "def code_solution(): pass\n"
        except Exception:
            return "def code_solution(): pass\n"

    def predict(self, prompt: str, task_context: str = "") -> str:
        action = self.predict_action(prompt, task_context=task_context)
        return action.code

    def train_step(self, trajectory: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        self.step_count += 1
        traj_obj = self._to_trajectory(trajectory)
        prompt = getattr(traj_obj, "prompt", "")
        reward = getattr(traj_obj, "reward", 0.0)

        decision = self.decide_learning(prompt, reward)

        update_info = {"decision": decision.value, "step": self.step_count}
        if decision == LearningDecision.UPDATE_ADAPTER:
            # Also store in memory so the trajectory is available for retrieval
            self.memory.add(traj_obj)
            cluster_id = self.router.get_cluster_id(prompt)
            task_res = self.trainer.queue(cluster_id, traj_obj)
            update_info["status"] = "queued_training"
            update_info["cluster_id"] = cluster_id
            update_info["training"] = task_res
        elif decision == LearningDecision.REQUEST_REVIEW:
            # Store in memory pending human/system review
            self.memory.add(traj_obj)
            update_info["status"] = "stored_for_review"
        else:
            # STORE_MEMORY (default) — always store, never skip
            self.memory.add(traj_obj)
            update_info["status"] = "stored_in_memory"

        return update_info

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.memory.save(os.path.join(path, "memory"))

    def load_checkpoint(self, path: str) -> None:
        mem_dir = os.path.join(path, "memory")
        if os.path.exists(mem_dir):
            self.memory.load(mem_dir)
