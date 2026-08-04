"""
Abstract Base Class for Task Routing and Cluster Assignment in OpenContinualEnv.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Router(ABC):
    """Abstract Base Class for task/cluster routing."""

    @abstractmethod
    def get_cluster_id(self, prompt: str) -> str:
        """Determines cluster_id for a given prompt."""
        pass

    @abstractmethod
    def get_adapter(self, cluster_id: str) -> Optional[str]:
        """Gets adapter file path/name registered for a cluster_id."""
        pass

    @abstractmethod
    def register_adapter(self, cluster_id: str, adapter_path: str) -> None:
        """Registers a trained adapter for a cluster_id."""
        pass

    @abstractmethod
    def list_clusters(self) -> Dict[str, int]:
        """Returns map of cluster_id -> assigned trajectory count."""
        pass
