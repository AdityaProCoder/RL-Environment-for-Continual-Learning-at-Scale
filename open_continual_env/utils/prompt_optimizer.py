"""
PromptOptimizer: Fast-Slow Learning (FST - May 2026).
Optimizes task-specific textual prompt prefixes ("fast weights") while preserving slow model weights.
"""

from typing import List, Dict, Any, Optional


class PromptOptimizer:
    """
    Manages fast weight prompt prefix optimization based on execution feedback.
    """

    def __init__(self, max_prefixes: int = 50):
        self.max_prefixes = max_prefixes
        self.optimized_prefixes: Dict[str, str] = {}

    def optimize_prefix(self, task_id: str, prompt: str, execution_feedback: Dict[str, Any]) -> str:
        """
        Generates/updates optimized prompt prefix ("fast weight") for a task based on sandbox feedback.
        """
        success = execution_feedback.get("success", False)
        stdout = execution_feedback.get("stdout", "")
        stderr = execution_feedback.get("stderr", "")

        prefix = f"[FAST-WEIGHT OPTIMIZATION: Task '{task_id}']"
        if not success and stderr:
            prefix += f" Note: Avoid previous runtime error: {stderr[:100].strip()}"
        elif success:
            prefix += " Note: Maintain previous successful function structure."

        self.optimized_prefixes[task_id] = prefix
        return prefix

    def get_prefix(self, task_id: str) -> Optional[str]:
        return self.optimized_prefixes.get(task_id)
