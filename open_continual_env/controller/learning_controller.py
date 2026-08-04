"""
Learning Controller for OpenContinualEnv.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union
import os

from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.trajectory.store import ExperienceStore


class LearningAction(str, Enum):
    IGNORE = "IGNORE"
    STORE_MEMORY = "STORE_MEMORY"
    UPDATE_LORA = "UPDATE_LORA"
    UPDATE_BASE = "UPDATE_BASE"
    NEEDS_TRAINING = "NEEDS_TRAINING"  # New state for MoA background training

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{self.value}'"


ControllerAction = LearningAction


class LearningController:
    """
    Sequential decision-making controller that evaluates interaction traces
    and determines whether and how experiences should be learned.
    """

    def __init__(
        self,
        min_reward_memory: float = 0.5,
        min_reward_lora: float = 0.8,
        min_reward_base: float = 0.95,
        safety_threshold: float = 0.0,
        experience_store: Optional[ExperienceStore] = None,
        **kwargs: Any,
    ):
        self.min_reward_memory = min_reward_memory
        self.min_reward_lora = min_reward_lora
        self.min_reward_base = min_reward_base
        self.safety_threshold = safety_threshold
        self.experience_store = experience_store or ExperienceStore()
        self.last_info: Dict[str, Any] = {}

        self._decision_counts: Dict[LearningAction, int] = {
            LearningAction.IGNORE: 0,
            LearningAction.STORE_MEMORY: 0,
            LearningAction.UPDATE_LORA: 0,
            LearningAction.UPDATE_BASE: 0,
            LearningAction.NEEDS_TRAINING: 0,
        }
        
        self.cluster_buffer: Dict[str, List[Any]] = {}
        self.cluster_threshold = 5  # Number of trajectories needed to trigger training

    def _get_cluster_id(self, prompt: str) -> str:
        """
        In a full implementation, this uses the local embedding model (text-embedding-nomic-embed-text-v1.5)
        to find the closest semantic cluster. For now, we use a simple keyword/hash fallback.
        """
        prompt_lower = prompt.lower()
        if "sort" in prompt_lower or "array" in prompt_lower: return "cluster_algorithms"
        if "math" in prompt_lower or "calculate" in prompt_lower: return "cluster_math"
        if "string" in prompt_lower or "text" in prompt_lower: return "cluster_text"
        return "cluster_general"

    def decide(
        self,
        trajectory: Any,
        context_metadata: Optional[Dict[str, Any]] = None,
        model_state: Optional[Dict[str, Any]] = None,
    ) -> LearningAction:
        """
        Evaluate trajectory and return a LearningAction enum decision.
        """
        if trajectory is None:
            self._decision_counts[LearningAction.IGNORE] += 1
            return LearningAction.IGNORE

        if isinstance(trajectory, str) or isinstance(trajectory, (int, float, list)):
            self._decision_counts[LearningAction.IGNORE] += 1
            return LearningAction.IGNORE

        if isinstance(trajectory, dict):
            try:
                trajectory = Trajectory.from_dict(trajectory)
            except Exception:
                self._decision_counts[LearningAction.IGNORE] += 1
                return LearningAction.IGNORE
        elif not isinstance(trajectory, Trajectory):
            self._decision_counts[LearningAction.IGNORE] += 1
            return LearningAction.IGNORE

        reward = float(getattr(trajectory, "reward", 0.0) or 0.0)
        exec_out = getattr(trajectory, "execution_output", {}) or {}
        pass_rate = float(exec_out.get("pass_rate", 0.0) or 0.0)
        success = bool(exec_out.get("success", False))

        feedback = getattr(trajectory, "feedback", {}) or {}
        safety_score = float(feedback.get("safety_score", 1.0) or 1.0)

        ms = model_state or {}
        if safety_score < self.safety_threshold or reward < self.min_reward_memory:
            action = LearningAction.IGNORE
            reason = f"Low reward ({reward:.2f}) or safety failure."
        elif reward >= self.min_reward_lora or success:
            prompt = getattr(trajectory, "prompt", "") if hasattr(trajectory, "prompt") else str(trajectory)
            cluster_id = self._get_cluster_id(prompt)

            if cluster_id not in self.cluster_buffer:
                self.cluster_buffer[cluster_id] = []
            self.cluster_buffer[cluster_id].append(trajectory)

            if len(self.cluster_buffer[cluster_id]) >= self.cluster_threshold:
                action = LearningAction.NEEDS_TRAINING
                reason = f"Cluster {cluster_id} reached threshold {self.cluster_threshold}. Ready for async PEFT training."
                self.cluster_buffer[cluster_id] = []
            elif ms.get("critical_milestone") or reward >= 0.99:
                action = LearningAction.UPDATE_BASE
                reason = f"Critical milestone / high reward ({reward:.2f}), base model update."
            else:
                action = LearningAction.UPDATE_LORA
                reason = f"High reward ({reward:.2f}), LoRA update."
        else:
            action = LearningAction.STORE_MEMORY
            reason = f"Stored in memory for RAG."



        self._decision_counts[action] += 1

        info = {
            "decision": action.value,
            "reason": reason,
            "reward": reward,
            "pass_rate": pass_rate,
            "decision_counts": {k.value: v for k, v in self._decision_counts.items()},
        }
        self.last_info = info

        if action in (LearningAction.STORE_MEMORY, LearningAction.UPDATE_LORA, LearningAction.UPDATE_BASE, LearningAction.NEEDS_TRAINING):
            self.experience_store.add(trajectory)

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Return decision distribution statistics."""
        total = sum(self._decision_counts.values())
        return {
            "total_decisions": total,
            "counts": {k.value: v for k, v in self._decision_counts.items()},
            "ratios": {
                k.value: (v / total if total > 0 else 0.0)
                for k, v in self._decision_counts.items()
            },
        }

    def get_action_distribution(self) -> Dict[str, Any]:
        """Alias for get_stats to match test suite."""
        return self.get_stats()
