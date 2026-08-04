"""
AdapterRegistry tracks trained LoRA adapters on disk and manages LRU loading/eviction.
"""

import os
from typing import Dict, List, Optional, Any
from collections import OrderedDict


class AdapterRegistry:
    """
    Manages adapter paths, versions, and active VRAM loading using LRU eviction.
    """

    def __init__(self, adapter_dir: str = "./adapters", max_active_adapters: int = 4):
        self.adapter_dir = adapter_dir
        self.max_active_adapters = max_active_adapters
        self.adapters: Dict[str, str] = {}
        self.versions: Dict[str, int] = {}
        self.active_lru: OrderedDict[str, Any] = OrderedDict()
        self._scan_adapters()

    def _scan_adapters(self) -> None:
        if not os.path.exists(self.adapter_dir):
            return
        for item in os.listdir(self.adapter_dir):
            path = os.path.join(self.adapter_dir, item)
            if os.path.isdir(path):
                self.adapters[item] = path
                self.versions[item] = self.versions.get(item, 1)

    def register(self, cluster_id: str, adapter_path: str) -> int:
        version = self.versions.get(cluster_id, 0) + 1
        self.adapters[cluster_id] = adapter_path
        self.versions[cluster_id] = version
        return version

    def get_adapter_path(self, cluster_id: str) -> Optional[str]:
        return self.adapters.get(cluster_id)

    def touch_active(self, cluster_id: str, adapter_obj: Any = None) -> Optional[str]:
        """Mark adapter as actively used in LRU cache. Evict if over budget."""
        if cluster_id not in self.adapters:
            return None

        if cluster_id in self.active_lru:
            self.active_lru.move_to_end(cluster_id)
        else:
            self.active_lru[cluster_id] = adapter_obj or self.adapters[cluster_id]

        evicted = None
        if len(self.active_lru) > self.max_active_adapters:
            evicted, _ = self.active_lru.popitem(last=False)
        return evicted

    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for c_id, path in self.adapters.items():
            result[c_id] = {
                "path": path,
                "version": self.versions.get(c_id, 1),
                "is_active": c_id in self.active_lru,
            }
        return result
