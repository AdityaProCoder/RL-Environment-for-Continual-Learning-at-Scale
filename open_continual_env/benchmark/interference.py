"""
Gradient Interference Matrix (GIM) - Novel Mathematical Contribution D1.
Computes task-to-task gradient alignment: cos(∇L_i, ∇L_j).
Positive = synergistic transfer; Negative = catastrophic interference.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class GradientInterferenceMatrix:
    """
    Tracks and analyzes task-pair gradient alignment matrices during continual learning.
    """

    def __init__(self, num_tasks: int = 10):
        self.num_tasks = num_tasks
        self.gim_matrix = np.eye(num_tasks, dtype=np.float32)

    def compute_gradient_cosine(self, grad_i: np.ndarray, grad_j: np.ndarray) -> float:
        """Computes cosine similarity between two gradient vectors."""
        norm_i = np.linalg.norm(grad_i)
        norm_j = np.linalg.norm(grad_j)
        if norm_i == 0 or norm_j == 0:
            return 0.0
        return float(np.dot(grad_i, grad_j) / (norm_i * norm_j))

    def update_pair(self, task_i: int, task_j: int, cos_val: float) -> None:
        if 0 <= task_i < self.num_tasks and 0 <= task_j < self.num_tasks:
            self.gim_matrix[task_i, task_j] = cos_val
            self.gim_matrix[task_j, task_i] = cos_val

    def get_interference_score(self, task_i: int) -> float:
        """Returns mean interference (negative alignment) for task_i against all past tasks."""
        row = self.gim_matrix[task_i, :task_i]
        if len(row) == 0:
            return 0.0
        negative_alignments = [min(0.0, float(x)) for x in row]
        return float(np.abs(np.mean(negative_alignments)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": self.gim_matrix.tolist(),
            "mean_synergy": float(np.mean(self.gim_matrix[self.gim_matrix > 0])),
            "mean_interference": float(np.mean(self.gim_matrix[self.gim_matrix < 0])) if np.any(self.gim_matrix < 0) else 0.0,
        }
