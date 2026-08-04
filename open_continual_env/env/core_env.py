"""
OpenContinualEnv built on Hugging Face openenv framework architecture
"""

from typing import Any, Dict, Optional, Tuple, Union, List
import sys
import os
try:
    import numpy as np
except ImportError:
    np = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    class DummySpace:
        def __init__(self, *args, **kwargs): pass
    class DummySpaces:
        Dict = DummySpace
        Text = DummySpace
    spaces = DummySpaces()

from enum import Enum
from pydantic import Field

from open_continual_env.env.sandbox import PythonSandbox, ExecutionResult
from open_continual_env.env.rewards import RewardEngine
from open_continual_env.memory.faiss_memory import FAISSMemory, LegacyExperienceStoreWrapper
from open_continual_env.routing.embedding_router import EmbeddingRouter
from open_continual_env.routing.adapter_registry import AdapterRegistry
from open_continual_env.training.online_trainer import OnlineTrainer
from open_continual_env.training.dpo_learner import DPOLearner
from open_continual_env.utils.novelty_gate import NoveltyGate

# Ensure openenv is accessible from site-packages if installed in user site-packages
_USER_SITE = r"C:\Users\Aditya\AppData\Local\Programs\Python\Python311\Lib\site-packages"
if os.path.exists(_USER_SITE) and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

try:
    from openenv.core import Environment, Action, Observation, State
    HAS_HF_OPENENV = True
except ImportError:
    try:
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State
        HAS_HF_OPENENV = True
    except ImportError:
        HAS_HF_OPENENV = False
        from pydantic import BaseModel

        class Action(BaseModel):
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class Observation(BaseModel):
            done: bool = Field(default=False)
            reward: Optional[Union[int, float]] = Field(default=None)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class State(BaseModel):
            episode_id: Optional[str] = Field(default=None)
            step_count: int = Field(default=0)

        class Environment:
            pass


class LearningDecision(str, Enum):

    """Actions the agent can choose to update its internal continual learning state."""
    ANSWER_ONLY    = "answer_only"     # Solve task, don't update state
    STORE_MEMORY   = "store_memory"    # Store trajectory in semantic memory
    UPDATE_ADAPTER = "update_adapter"  # Trigger LoRA update for active cluster
    SKIP_UPDATE    = "skip_update"     # Explicitly skip update (anti-forgetting signal)
    REQUEST_REVIEW = "request_review"  # Request re-evaluation / feedback


DEFAULT_TASKS = [
    {
        "task_id": "default_add_01",
        "prompt": "Write a Python function named `add(a, b)` that returns the sum of two numbers.",
        "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
        "entry_point": "add",
    },
    {
        "task_id": "default_multiply_02",
        "prompt": "Write a Python function named `multiply(a, b)` that returns the product of two numbers.",
        "test_code": "assert multiply(2, 3) == 6\nassert multiply(4, 5) == 20",
        "entry_point": "multiply",
    },
]


class OpenContinualAction(Action):
    """Hugging Face OpenEnv 2D Action schema for OpenContinualEnv."""
    code: str = Field(default="", description="Python code string to execute in sandbox")
    learning_decision: LearningDecision = Field(
        default=LearningDecision.ANSWER_ONLY,
        description="Learning decision policy action"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Action metadata, cluster hint, confidence, etc.")

    @classmethod
    def from_action(cls, action: Union[str, Dict[str, Any], Tuple[Any, ...], "OpenContinualAction", None]) -> "OpenContinualAction":
        if action is None:
            return cls(code="", learning_decision=LearningDecision.ANSWER_ONLY)
        if isinstance(action, cls):
            return action
        if isinstance(action, str):
            return cls(code=action, learning_decision=LearningDecision.ANSWER_ONLY)
        if isinstance(action, tuple) and len(action) >= 2:
            code_val = str(action[0])
            dec_val = action[1]
            if isinstance(dec_val, LearningDecision):
                decision_enum = dec_val
            elif isinstance(dec_val, int):
                mapping = [
                    LearningDecision.ANSWER_ONLY,
                    LearningDecision.STORE_MEMORY,
                    LearningDecision.UPDATE_ADAPTER,
                    LearningDecision.SKIP_UPDATE,
                    LearningDecision.REQUEST_REVIEW,
                ]
                decision_enum = mapping[dec_val % len(mapping)]
            else:
                try:
                    decision_enum = LearningDecision(str(dec_val))
                except ValueError:
                    decision_enum = LearningDecision.ANSWER_ONLY
            return cls(code=code_val, learning_decision=decision_enum)
        if isinstance(action, dict):
            dec_val = action.get("learning_decision", LearningDecision.ANSWER_ONLY)
            if isinstance(dec_val, LearningDecision):
                decision_enum = dec_val
            elif isinstance(dec_val, int):
                mapping = [
                    LearningDecision.ANSWER_ONLY,
                    LearningDecision.STORE_MEMORY,
                    LearningDecision.UPDATE_ADAPTER,
                    LearningDecision.SKIP_UPDATE,
                    LearningDecision.REQUEST_REVIEW,
                ]
                decision_enum = mapping[dec_val % len(mapping)]
            else:
                try:
                    decision_enum = LearningDecision(str(dec_val))
                except ValueError:
                    decision_enum = LearningDecision.ANSWER_ONLY
            return cls(
                code=action.get("code", action.get("action", "")),
                learning_decision=decision_enum,
                metadata=action.get("metadata", {}),
            )
        return cls(code=str(action), learning_decision=LearningDecision.ANSWER_ONLY)


class OpenContinualObservation(Observation):
    """Hugging Face OpenEnv Observation schema for OpenContinualEnv."""
    prompt: str = Field(default="", description="Task prompt text")
    task_id: str = Field(default="", description="Task identifier")
    context: str = Field(default="", description="Task context information")
    execution_result: Optional[Any] = Field(default=None, description="Sandbox execution result")
    memory_state: Dict[str, Any] = Field(default_factory=dict, description="Current agent memory state snapshot")
    learning_hint: str = Field(default="answer_only", description="Suggested learning decision based on env state")
    forgetting_estimate: float = Field(default=0.0, description="Running estimate of forgetting metric")
    info: Dict[str, Any] = Field(default_factory=dict, description="Step metadata and execution info")

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        if key in self.info:
            return self.info[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.metadata or key in self.info

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        if key in self.metadata:
            return self.metadata[key]
        if key in self.info:
            return self.info[key]
        return default

    def keys(self):
        return list(self.model_fields.keys())

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class OpenContinualState(State):
    """Hugging Face OpenEnv State schema for OpenContinualEnv."""
    current_task_idx: int = Field(default=0, description="Index of current active task")
    current_task: Dict[str, Any] = Field(default_factory=dict, description="Current task dict")


class OpenContinualEnv(Environment):
    """
    Environment for continual LLM code evaluation built on Hugging Face openenv framework architecture.
    Features 2D action space (Task Action + Learning Action) executed directly inside step().
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 10)
        self.tasks: List[Dict[str, Any]] = self.config.get("tasks", DEFAULT_TASKS)

        self.sandbox = self.config.get("sandbox") or PythonSandbox()
        self.reward_engine = self.config.get("reward_engine") or RewardEngine()
        self.experience_store = self.config.get("experience_store", None)

        # Semantic components
        if self.config.get("semantic_memory"):
            self.semantic_memory = self.config["semantic_memory"]
        elif self.experience_store is not None:
            self.semantic_memory = LegacyExperienceStoreWrapper(self.experience_store)
        else:
            self.semantic_memory = FAISSMemory()

        self.embedding_router = self.config.get("embedding_router") or EmbeddingRouter()
        self.adapter_registry = self.config.get("adapter_registry") or AdapterRegistry()
        self.online_trainer = self.config.get("online_trainer") or OnlineTrainer(adapter_registry=self.adapter_registry)
        self.novelty_gate = self.config.get("novelty_gate") or NoveltyGate(memory=self.semantic_memory)
        self.dpo_learner = self.config.get("dpo_learner") or DPOLearner()

        self.reward_weights = self.config.get("reward_weights", {
            "alpha": 1.0,     # Task reward weight
            "beta": 0.1,      # Backward transfer weight
            "gamma": 0.05,    # Forward transfer weight
            "delta": 0.2,     # Forgetting penalty weight
            "epsilon": 0.01,  # Compute cost penalty weight
        })

        self.current_step = 0
        self.current_task_idx = 0
        self.current_task = self.tasks[0] if self.tasks else {}
        self._history: List[Dict[str, Any]] = []

        if gym is not None:
            self.observation_space = spaces.Dict({
                "prompt": spaces.Text(max_length=4096),
                "task_id": spaces.Text(max_length=128),
                "context": spaces.Text(max_length=4096),
            })
            self.action_space = spaces.Dict({
                "code": spaces.Text(max_length=8192),
                "learning_decision": spaces.Discrete(5),
            })
        else:
            self.observation_space = None
            self.action_space = None

    @property
    def state(self) -> OpenContinualState:
        """Returns the current environment state object."""
        return OpenContinualState(
            episode_id=str(self.current_task_idx),
            step_count=self.current_step,
            current_task_idx=self.current_task_idx,
            current_task=self.current_task,
        )

    def observe(self) -> Dict[str, Any]:
        """Returns snapshot of current memory, cluster distribution, and learning estimates."""
        mem_size = len(self.semantic_memory) if self.semantic_memory is not None else 0
        clusters = self.embedding_router.list_clusters() if self.embedding_router is not None else {}
        adapters = self.adapter_registry.list_adapters() if self.adapter_registry is not None else {}
        recent_rewards = [h["reward"] for h in self._history[-10:]] if self._history else []
        
        # Heuristic learning recommendation based on recent performance
        if not recent_rewards:
            suggested = LearningDecision.ANSWER_ONLY
        elif recent_rewards[-1] >= 0.8:
            suggested = LearningDecision.STORE_MEMORY
        elif recent_rewards[-1] < 0.3:
            suggested = LearningDecision.SKIP_UPDATE
        else:
            suggested = LearningDecision.ANSWER_ONLY

        return {
            "memory_size": mem_size,
            "cluster_distribution": clusters,
            "adapter_versions": adapters,
            "recent_rewards": recent_rewards,
            "forgetting_estimate": getattr(self, "_forgetting_estimate", 0.0),
            "suggested_action": suggested.value,
        }

    def reward(
        self,
        task_reward: float,
        trajectory: Any = None,
        learning_decision: Union[LearningDecision, str] = LearningDecision.ANSWER_ONLY,
        bwt_delta: float = 0.0,
        fwt_delta: float = 0.0,
        forgetting_delta: float = 0.0,
    ) -> float:
        """Computes composite reward: task execution reward + CL transfer metrics - penalties."""
        w = self.reward_weights
        dec_enum = LearningDecision(learning_decision) if isinstance(learning_decision, str) else learning_decision
        
        cost_map = {
            LearningDecision.ANSWER_ONLY: 0.0,
            LearningDecision.STORE_MEMORY: 0.01,
            LearningDecision.UPDATE_ADAPTER: 0.1,
            LearningDecision.SKIP_UPDATE: 0.0,
            LearningDecision.REQUEST_REVIEW: 0.05,
        }
        compute_cost = cost_map.get(dec_enum, 0.0)

        composite = (
            w.get("alpha", 1.0) * float(task_reward)
            + w.get("beta", 0.1) * float(bwt_delta)
            + w.get("gamma", 0.05) * float(fwt_delta)
            - w.get("delta", 0.2) * float(forgetting_delta)
            - w.get("epsilon", 0.01) * float(compute_cost)
        )
        return float(composite)

    def update(
        self,
        trajectory: Any,
        learning_decision: Union[LearningDecision, str] = LearningDecision.ANSWER_ONLY
    ) -> Dict[str, Any]:
        """Executes the selected learning decision inside the step flow."""
        dec_enum = LearningDecision(learning_decision) if isinstance(learning_decision, str) else learning_decision
        update_info = {"decision": dec_enum.value, "status": "executed"}

        if dec_enum == LearningDecision.STORE_MEMORY:
            if self.semantic_memory is not None:
                traj_id = self.semantic_memory.add(trajectory)
                update_info["trajectory_id"] = traj_id
                update_info["status"] = "stored"

        elif dec_enum == LearningDecision.UPDATE_ADAPTER:
            cluster_id = "cluster_general"
            if self.embedding_router is not None and hasattr(trajectory, "prompt"):
                cluster_id = self.embedding_router.get_cluster_id(trajectory.prompt)
            
            # Check novelty gate before training
            is_novel = self.novelty_gate.should_update(
                prompt=getattr(trajectory, "prompt", ""),
                reward=getattr(trajectory, "reward", 0.0)
            ) if self.novelty_gate is not None else True

            if is_novel:
                train_info = self.online_trainer.queue(cluster_id, trajectory) if self.online_trainer is not None else {}
                update_info["cluster_id"] = cluster_id
                update_info["training"] = train_info
                update_info["gated"] = False
            else:
                update_info["status"] = "gated_novelty_low"
                update_info["gated"] = True

        elif dec_enum == LearningDecision.SKIP_UPDATE:
            update_info["anti_forgetting_signal"] = True
            update_info["status"] = "skipped"

        elif dec_enum == LearningDecision.REQUEST_REVIEW:
            update_info["review_requested"] = True
            update_info["status"] = "review_queued"

        elif dec_enum == LearningDecision.ANSWER_ONLY:
            update_info["status"] = "noop"

        return update_info

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[OpenContinualObservation, Dict[str, Any]]:
        """Resets environment state and returns (OpenContinualObservation, info) tuple."""
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"Invalid seed type: {type(seed).__name__}. Seed must be an integer or None.")

        options = options or {}
        self.current_step = 0

        if self.tasks:
            if seed is not None:
                self.current_task_idx = int(seed) % len(self.tasks)
            elif "task_idx" in options:
                self.current_task_idx = int(options["task_idx"]) % len(self.tasks)
            else:
                self.current_task_idx = 0
            self.current_task = self.tasks[self.current_task_idx]
        else:
            self.current_task = {"task_id": "empty", "prompt": "", "test_code": ""}

        memory_snapshot = self.observe()

        info = {
            "step": self.current_step,
            "task_id": self.current_task.get("task_id", ""),
            "max_steps": self.max_steps,
            "memory_state": memory_snapshot,
        }

        obs = OpenContinualObservation(
            prompt=self.current_task.get("prompt", ""),
            task_id=self.current_task.get("task_id", ""),
            context=self.current_task.get("context", ""),
            memory_state=memory_snapshot,
            learning_hint=memory_snapshot.get("suggested_action", "answer_only"),
            info=info,
            done=False,
            reward=None,
        )

        return obs, info

    def step(
        self,
        action: Union[OpenContinualAction, Tuple[Any, ...], Dict[str, Any], str, None],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[OpenContinualObservation, float, bool, bool, Dict[str, Any]]:
        """Executes action in sandbox, runs learning action in update(), and computes composite reward."""
        self.current_step += 1

        action_obj = OpenContinualAction.from_action(action)
        code_str = action_obj.code
        learning_dec = action_obj.learning_decision
        domain = self.current_task.get("domain", "code")

        if domain == "math":
            ref_ans = self.current_task.get("reference_answer", "")
            task_reward, math_info = self.reward_engine.compute_math_reward(code_str, ref_ans)
            terminated = bool(math_info.get("match", False))
            truncated = bool(self.current_step >= self.max_steps)
            exec_info = {
                "step": self.current_step,
                "success": math_info.get("match", False),
                "pass_rate": 1.0 if math_info.get("match") else 0.0,
                "math_match": math_info.get("match", False),
                "extracted": math_info.get("extracted", ""),
                "reference": ref_ans,
                "execution_time": 0.0,
            }
            exec_res = None
        else:
            test_code = self.current_task.get("test_code", "")
            timeout = float(timeout_s if timeout_s is not None else self.current_task.get("timeout", 5.0))
            exec_res: ExecutionResult = self.sandbox.execute(code_str, test_code=test_code, timeout=timeout)

            task_reward = float(self.reward_engine.compute_reward(exec_res, code=code_str))
            terminated = bool(exec_res.pass_rate == 1.0)
            truncated = bool(self.current_step >= self.max_steps)

            exec_info = {
                "step": self.current_step,
                "success": exec_res.success,
                "exit_code": exec_res.exit_code,
                "stdout": exec_res.stdout,
                "stderr": exec_res.stderr,
                "tests_passed": exec_res.tests_passed,
                "tests_total": exec_res.tests_total,
                "pass_rate": exec_res.pass_rate,
                "execution_time": exec_res.execution_time,
                "execution_result": exec_res,
            }

        # Build trajectory dataclass
        from open_continual_env.trajectory.schema import Trajectory
        traj = Trajectory(
            trajectory_id=f"step_{self.current_step}_{self.current_task_idx}",
            prompt=self.current_task.get("prompt", ""),
            model_response=code_str,
            reasoning_notes="",
            generated_code=code_str,
            execution_output={
                "stdout": exec_res.stdout if exec_res else "",
                "stderr": exec_res.stderr if exec_res else "",
                "exit_code": exec_res.exit_code if exec_res else 0,
                "pass_rate": exec_res.pass_rate if exec_res else (1.0 if terminated else 0.0),
                "success": exec_res.success if exec_res else terminated,
            },
            feedback={},
            reward=task_reward,
            regression_results={},
            timestamp="",
        )

        # Execute learning decision via self.update()
        update_result = self.update(traj, learning_dec)

        # Compute composite reward
        composite_reward = self.reward(
            task_reward=task_reward,
            trajectory=traj,
            learning_decision=learning_dec,
        )

        memory_snapshot = self.observe()

        info = {
            **exec_info,
            "task_reward": task_reward,
            "composite_reward": composite_reward,
            "learning_decision_taken": learning_dec.value,
            "update_result": update_result,
            "memory_state": memory_snapshot,
            "novelty_score": self.novelty_gate.compute_novelty(traj.prompt) if self.novelty_gate else 1.0,
            "composite_reward_breakdown": {
                "task_reward": task_reward,
                "bwt_delta": 0.0,
                "fwt_delta": 0.0,
                "forgetting_penalty": 0.0,
                "compute_cost": 0.01 if learning_dec == LearningDecision.STORE_MEMORY else (0.1 if learning_dec == LearningDecision.UPDATE_ADAPTER else 0.0),
            },
        }

        obs = OpenContinualObservation(
            prompt=self.current_task.get("prompt", ""),
            task_id=self.current_task.get("task_id", ""),
            context=self.current_task.get("context", ""),
            execution_result=exec_res,
            memory_state=memory_snapshot,
            learning_hint=memory_snapshot.get("suggested_action", "answer_only"),
            info=info,
            done=terminated or truncated,
            reward=composite_reward,
        )

        step_record = {
            "step": self.current_step,
            "obs": obs,
            "action": action_obj,
            "reward": composite_reward,
            "task_reward": task_reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        self._history.append(step_record)

        return obs, composite_reward, terminated, truncated, info

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns step execution history."""
        return self._history

    def evaluate(
        self,
        model: Any = None,
        test_suite: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluates model across specified test_suite or internal tasks."""
        suite = test_suite if test_suite is not None else self.tasks
        if not suite:
            return {
                "pass_rate": 0.0,
                "mean_reward": 0.0,
                "total_tasks": 0,
                "successful_tasks": 0,
            }

        total_reward = 0.0
        successful_tasks = 0
        total_tasks = len(suite)

        for task in suite:
            prompt = task.get("prompt", "")
            domain = task.get("domain", "code")

            if model is not None and hasattr(model, "predict"):
                action = model.predict(prompt)
            elif model is not None and callable(model):
                action = model(prompt)
            else:
                action = "pass"

            action_obj = OpenContinualAction.from_action(action)

            if domain == "math":
                ref_ans = task.get("reference_answer", "")
                r, math_info = self.reward_engine.compute_math_reward(action_obj.code, ref_ans)
                if math_info.get("match", False):
                    successful_tasks += 1
            else:
                test_code = task.get("test_code", "")
                exec_res = self.sandbox.execute(action_obj.code, test_code=test_code)
                r = self.reward_engine.compute_reward(exec_res, code=action_obj.code)
                if exec_res.pass_rate == 1.0:
                    successful_tasks += 1

            total_reward += r

        return {
            "pass_rate": float(successful_tasks / total_tasks),
            "mean_reward": float(total_reward / total_tasks),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
        }



_BaseGymEnv = gym.Env if gym is not None else object


class OpenContinualGymWrapper(_BaseGymEnv):

    """
    Farama Gymnasium compatibility wrapper around HF OpenEnv OpenContinualEnv.
    Supports 2D actions: Dict({code: Text, learning_decision: Discrete(5)})
    """

    def __init__(self, env: Optional[OpenContinualEnv] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.env = env or OpenContinualEnv(config=config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = getattr(self.env, "metadata", {"render_modes": []})

    @property
    def state(self) -> OpenContinualState:
        return self.env.state

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_dict = obs.to_dict() if isinstance(obs, OpenContinualObservation) else dict(obs)
        return obs_dict, info

    def step(
        self,
        action: Any
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = obs.to_dict() if isinstance(obs, OpenContinualObservation) else dict(obs)
        return obs_dict, reward, terminated, truncated, info

    def evaluate(
        self,
        model: Any = None,
        test_suite: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return self.env.evaluate(model=model, test_suite=test_suite)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.env.get_history()

