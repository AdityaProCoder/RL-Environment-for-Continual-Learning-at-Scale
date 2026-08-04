"""
ContextPruner for fitting retrieved experiences into model context window budgets.
"""

from typing import List, Any, Union
from open_continual_env.trajectory.schema import Trajectory


class ContextPruner:
    """
    Ranks and prunes retrieved trajectories and prompts based on token length estimates.
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimation (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    @classmethod
    def prune_retrieved_experiences(
        cls,
        experiences: List[Union[Trajectory, dict, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """
        Formats and truncates a list of retrieved past trajectories to fit within max_tokens.
        """
        if not experiences:
            return ""

        formatted_blocks = []
        accumulated_tokens = 0

        for idx, exp in enumerate(experiences):
            if isinstance(exp, Trajectory):
                prompt = exp.prompt
                code = exp.generated_code or exp.model_response
                reward = exp.reward
            elif isinstance(exp, dict):
                prompt = exp.get("prompt", "")
                code = exp.get("generated_code", exp.get("model_response", ""))
                reward = exp.get("reward", 0.0)
            else:
                prompt = str(exp)
                code = ""
                reward = 0.0

            block = (
                f"--- Reference Example #{idx + 1} (Reward: {reward:.2f}) ---\n"
                f"Task: {prompt}\n"
                f"Solution:\n```python\n{code}\n```\n"
            )
            tokens = cls.estimate_tokens(block)

            if accumulated_tokens + tokens > max_tokens:
                # Stop adding more blocks if token budget exceeded
                break

            formatted_blocks.append(block)
            accumulated_tokens += tokens

        return "\n".join(formatted_blocks)

    @classmethod
    def prune_prompt(cls, prompt: str, max_tokens: int = 2048) -> str:
        """Truncates prompt text if it exceeds token budget while preserving structure."""
        tokens = cls.estimate_tokens(prompt)
        if tokens <= max_tokens:
            return prompt
        char_limit = max_tokens * 4
        return prompt[:char_limit] + "\n...[truncated due to length]"
