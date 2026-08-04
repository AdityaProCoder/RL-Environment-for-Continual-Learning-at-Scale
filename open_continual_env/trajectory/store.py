"""Structured experience store for trajectory logging, persistence, and querying."""

import json
import os
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from open_continual_env.trajectory.schema import Trajectory


class ExperienceStore:
    """Thread-safe store for indexing, querying, and persisting Trajectory objects."""

    def __init__(self, trajectories: Optional[List[Trajectory]] = None) -> None:
        self._lock = threading.RLock()
        self._trajectories: List[Trajectory] = []
        if trajectories:
            for t in trajectories:
                self.add(t)

    @property
    def trajectories(self) -> List[Trajectory]:
        """Return all stored trajectories."""
        with self._lock:
            return list(self._trajectories)

    @trajectories.setter
    def trajectories(self, value: List[Trajectory]) -> None:
        """Set stored trajectories."""
        with self._lock:
            self._trajectories = list(value)

    def add(self, trajectory: Union[Trajectory, Dict[str, Any]]) -> None:
        """Add a trajectory to the store."""
        if isinstance(trajectory, dict):
            trajectory = Trajectory.from_dict(trajectory)
        elif not isinstance(trajectory, Trajectory):
            raise TypeError(f"Expected Trajectory or dict, got {type(trajectory).__name__}")

        with self._lock:
            self._trajectories.append(trajectory)

    def get_all(self) -> List[Trajectory]:
        """Return a copy of all stored trajectories."""
        with self._lock:
            return list(self._trajectories)

    def query(self, filter_fn: Callable[[Trajectory], bool]) -> List[Trajectory]:
        """Filter stored trajectories using a boolean predicate function."""
        with self._lock:
            return [t for t in self._trajectories if filter_fn(t)]

    def filter_by_reward(self, min_reward: float) -> List[Trajectory]:
        """Filter trajectories with reward >= min_reward."""
        with self._lock:
            return [t for t in self._trajectories if t.reward >= min_reward]

    def filter_by_feedback(self, feedback_key: str, min_score: float) -> List[Trajectory]:
        """Filter trajectories where feedback[feedback_key] >= min_score."""
        with self._lock:
            results = []
            for t in self._trajectories:
                val = t.feedback.get(feedback_key)
                if isinstance(val, (int, float)) and val >= min_score:
                    results.append(t)
            return results

    def get_replay_buffer(
        self,
        batch_size: int,
        sample_strategy: str = "uniform",
        freshness_decay: float = 0.05
    ) -> List[Trajectory]:
        """
        Sample a batch of trajectories for experience replay.
        Supports FA-PER (2026) Freshness-Aware Prioritized Replay with exponential age decay.
        """
        with self._lock:
            if not self._trajectories or batch_size <= 0:
                return []

            total = len(self._trajectories)
            k = min(batch_size, total)

            if sample_strategy in ("recent", "latest"):
                return list(self._trajectories[-k:])

            if sample_strategy in ("reward_weighted", "prioritized", "freshness_aware"):
                import math
                priorities = []
                for idx, t in enumerate(self._trajectories):
                    base_p = max(0.01, float(t.reward))
                    age = total - 1 - idx
                    # FA-PER age decay: priority = base_p * exp(-decay * age)
                    freshness_p = base_p * math.exp(-freshness_decay * age)
                    priorities.append(freshness_p)

                sum_p = sum(priorities)
                if sum_p <= 0:
                    return random.sample(self._trajectories, k)
                weights = [p / sum_p for p in priorities]
                selected_indices = random.choices(range(total), weights=weights, k=k)
                return [self._trajectories[i] for i in selected_indices]

            return random.sample(self._trajectories, k)


    def save_json(self, path: str) -> None:
        """Serialize all trajectories to a single JSON array file."""
        with self._lock:
            data = [t.to_dict() for t in self._trajectories]
            dir_name = os.path.dirname(os.path.abspath(path))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def save_jsonl(self, path: str) -> None:
        """Serialize all trajectories to a JSONL file (one JSON per line)."""
        with self._lock:
            dir_name = os.path.dirname(os.path.abspath(path))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for t in self._trajectories:
                    f.write(t.to_json() + "\n")

    def load_json(self, path: str, clear_existing: bool = False) -> List[Trajectory]:
        """Load trajectories from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            with self._lock:
                if clear_existing:
                    self._trajectories.clear()
                return list(self._trajectories)

        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Corrupted JSON content in {path}") from e

        if not isinstance(data, list):
            raise ValueError(f"Corrupted JSON content in {path}")

        trajs = []
        for item in data:
            if isinstance(item, dict):
                try:
                    trajs.append(Trajectory.from_dict(item))
                except Exception as e:
                    raise ValueError(f"Corrupted JSON content in {path}") from e
            else:
                raise ValueError(f"Corrupted JSON content in {path}")

        with self._lock:
            if clear_existing:
                self._trajectories = trajs
            else:
                self._trajectories.extend(trajs)
            return list(self._trajectories)

    def load_jsonl(self, path: str, clear_existing: bool = False) -> List[Trajectory]:
        """Load trajectories from a JSONL file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        trajs = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except Exception as e:
                    raise ValueError(f"Corrupted JSONL content in {path}") from e

                if not isinstance(data, dict):
                    raise ValueError(f"Corrupted JSONL content in {path}")

                try:
                    trajs.append(Trajectory.from_dict(data))
                except Exception as e:
                    raise ValueError(f"Corrupted JSONL content in {path}") from e

        with self._lock:
            if clear_existing:
                self._trajectories = trajs
            else:
                self._trajectories.extend(trajs)
            return list(self._trajectories)

    def clear(self) -> None:
        """Clear all stored trajectories."""
        with self._lock:
            self._trajectories.clear()

    def __len__(self) -> int:
        """Return the number of stored trajectories."""
        with self._lock:
            return len(self._trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Indexing access for stored trajectories."""
        with self._lock:
            return self._trajectories[idx]
