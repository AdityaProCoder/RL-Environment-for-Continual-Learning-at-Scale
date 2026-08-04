"""gcl: Grounded Continual Learning for deployed code LLMs (real-learning core)."""
from .config import ExperimentConfig
from .sandbox import PythonSandbox, ExecutionResult
from .verify import Verifier, RewardWeights, code_quality
from .curriculum import (Task, Family, StreamAssembler, api_rename, spec_paraphrase,
                         spec_perturb, canary_report)
from .env import GroundedContinualEnv, Action, LearnOp, Observation
from .learners.learners import (LEARNERS, FrozenLearner, AlwaysLoRALearner, ReplayLearner,
                                EWCLearner, ControllerLearner, VSRLearner)

__version__ = "0.2.0"
__all__ = [
    "ExperimentConfig", "PythonSandbox", "ExecutionResult", "Verifier", "RewardWeights",
    "code_quality", "Task", "Family", "StreamAssembler", "api_rename", "spec_paraphrase",
    "canary_report", "GroundedContinualEnv", "Action", "LearnOp", "Observation",
    "LEARNERS",
]
