"""
Protocol and Abstract Base Class for Semantic Memory in OpenContinualEnv.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from open_continual_env.trajectory.schema import Trajectory


class SemanticMemory(ABC):
    """Abstract Base Class for semantic trajectory memory."""

    @abstractmethod
    def add(
        self,
        trajectory: Union[Trajectory, Dict[str, Any]],
        embedding: Optional[List[float]] = None
    ) -> str:
        """Add a trajectory to semantic memory. Returns trajectory_id."""
        pass

    @abstractmethod
    def query(
        self,
        prompt: str,
        top_k: int = 5
    ) -> List[Trajectory]:
        """Query memory by text prompt and return top_k most similar trajectories."""
        pass

    @abstractmethod
    def query_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5
    ) -> List[Trajectory]:
        """Query memory by embedding vector and return top_k most similar trajectories."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of trajectories stored."""
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        """Save memory index and stored trajectories to disk."""
        pass

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load memory index and stored trajectories from disk."""
        pass
