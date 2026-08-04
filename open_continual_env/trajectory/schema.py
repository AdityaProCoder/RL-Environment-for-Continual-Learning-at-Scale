"""Schema for interaction trajectories in OpenContinualEnv."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union


@dataclass
class Trajectory:
    """Dataclass representing a complete interaction trajectory."""

    trajectory_id: str = ""
    prompt: str = ""
    model_response: str = ""
    reasoning_notes: str = ""
    generated_code: str = ""
    execution_output: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    regression_results: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Normalize defaults and enforce types."""
        if self.trajectory_id is None:
            self.trajectory_id = ""
        if self.prompt is None:
            self.prompt = ""
        if self.model_response is None:
            self.model_response = ""
        if self.reasoning_notes is None:
            self.reasoning_notes = ""
        if self.generated_code is None:
            self.generated_code = ""
        if self.execution_output is None:
            self.execution_output = {}
        if self.feedback is None:
            self.feedback = {}
        if self.reward is None:
            self.reward = 0.0
        else:
            self.reward = float(self.reward)
        if self.regression_results is None:
            self.regression_results = {}
        if self.timestamp is None:
            self.timestamp = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trajectory to a standard Python dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "prompt": self.prompt,
            "model_response": self.model_response,
            "reasoning_notes": self.reasoning_notes,
            "generated_code": self.generated_code,
            "execution_output": self.execution_output,
            "feedback": self.feedback,
            "reward": self.reward,
            "regression_results": self.regression_results,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize trajectory to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        """Construct Trajectory from a dictionary."""
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict to convert to Trajectory, got {type(d).__name__}")
        return cls(
            trajectory_id=str(d.get("trajectory_id", "")),
            prompt=str(d.get("prompt", "")),
            model_response=str(d.get("model_response", "") or ""),
            reasoning_notes=str(d.get("reasoning_notes", "") or ""),
            generated_code=str(d.get("generated_code", "") or ""),
            execution_output=dict(d.get("execution_output") or {}) if isinstance(d.get("execution_output"), dict) else {},
            feedback=dict(d.get("feedback") or {}) if isinstance(d.get("feedback"), dict) else {},
            reward=float(d.get("reward", 0.0) or 0.0),
            regression_results=dict(d.get("regression_results") or {}) if isinstance(d.get("regression_results"), dict) else {},
            timestamp=str(d.get("timestamp", "") or ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "Trajectory":
        """Construct Trajectory from a JSON string."""
        if not isinstance(s, (str, bytes, bytearray)):
            raise ValueError(f"Expected string/bytes to parse JSON, got {type(s).__name__}")
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
        return cls.from_dict(data)

    def __getitem__(self, item: str) -> Any:
        """Dict-like indexing access for fields."""
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(f"Key {item!r} not found in Trajectory")

    def get(self, item: str, default: Any = None) -> Any:
        """Dict-like safe getter for fields."""
        d = self.to_dict()
        return d.get(item, default)
