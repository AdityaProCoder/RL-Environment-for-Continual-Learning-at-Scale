"""Base class for continual learning baseline agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from open_continual_env.trajectory.schema import Trajectory


class BaseContinualAgent(ABC):
    """Abstract base class for continual learning agents."""

    def __init__(
        self,
        agent_name: str = "BaseContinualAgent",
        llm_client: Optional[Any] = None,
    ) -> None:
        self.agent_name = agent_name
        self.step_count: int = 0
        self.llm_client = llm_client

    def _generate_with_llm(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response using llm_client if available."""
        if self.llm_client is not None:
            if hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt, system_prompt=system_prompt)
            elif hasattr(self.llm_client, "predict"):
                return self.llm_client.predict(prompt)
            elif callable(self.llm_client):
                return self.llm_client(prompt)
        return None

    def _to_trajectory(self, trajectory: Union[Trajectory, dict, Any]) -> Trajectory:
        """Standardize trajectory input into a Trajectory dataclass object."""
        if isinstance(trajectory, Trajectory):
            return trajectory
        elif isinstance(trajectory, dict):
            return Trajectory.from_dict(trajectory)
        else:
            return Trajectory(
                trajectory_id="unknown",
                prompt=str(trajectory) if trajectory is not None else "",
                model_response="",
            )

    @abstractmethod
    def train_step(self, trajectory: Union[Trajectory, dict]) -> Dict[str, Any]:
        """Perform a single continual learning update step given a trajectory."""
        pass

    @abstractmethod
    def predict(self, prompt: str) -> str:
        """Generate response code / text for a given prompt."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint/weights to file path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint/weights from file path."""
        pass

    @staticmethod
    def _extract_code(response_text: str) -> str:
        """Extract python code block from raw LLM text output."""
        if not response_text:
            return ""

        blocks = []
        parts = response_text.split("```python")
        for part in parts[1:]:
            block = part.split("```")[0].strip()
            if block:
                blocks.append(block)

        if not blocks:
            parts = response_text.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i].strip()
                if block:
                    blocks.append(block)

        valid_blocks = []
        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            is_placeholder = False
            if len(lines) == 1 and lines[0] in ("...", "pass", "return"):
                is_placeholder = True
            elif len(lines) <= 2 and all(l in ("...", "pass", "return", "def solution():") or (l.startswith("def ") and l.endswith("pass")) for l in lines):
                is_placeholder = True

            if not is_placeholder:
                valid_blocks.append(block)

        if valid_blocks:
            return valid_blocks[-1]
        elif blocks:
            return blocks[-1]

        return response_text.strip()

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics dict for tracking agent state and performance."""
        return {
            "agent_name": self.agent_name,
            "step_count": self.step_count,
        }

