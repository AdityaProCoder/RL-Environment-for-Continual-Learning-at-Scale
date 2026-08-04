"""Experiment configuration — the single source of truth for a run (I4)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class ExperimentConfig:
    # model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"   # smoke; 1.5B/4B for signal/scale
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    # lora
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    train_steps_per_update: int = 3
    max_seq_len: int = 512
    # stream
    families: List[str] = field(default_factory=list)
    tasks_per_family: int = 8
    episodes_per_task: int = 1
    max_attempts_per_task: int = 2
    # safety gate
    gate_epsilon: float = 0.02          # max allowed holdout regression to promote
    holdout_size: int = 8
    # clip/budget
    max_updates: int = 200
    # ---- Verified Skill Regeneration (VSR) ----
    use_vsr_gate: bool = False          # primary safety = vault-test veto (vs noisy holdout-eps)
    use_reference_injection: bool = False  # train toward external truth, not the model's own guess
    vsr_learners: List[str] = field(default_factory=lambda: ["vsr"])  # which learners get the vault
    refinject_learners: List[str] = field(default_factory=lambda: ["vsr"])  # which get gold/reference injection
    vault_commit_min: float = 0.9       # min reward to ADMIT a skill to the vault
    vault_retrieve_k: int = 3           # skills injected as in-context grounding per step
    vault_recall_min_sim: float = 0.0   # (reserved) retrieval similarity floor
    vault_gate_check: int = 3           # verified skills re-checked on every gated update
    # output
    out_dir: str = "runs/run"
    seed: int = 42
    # rl / flags (honest GRPO: only if trl present + explicitly enabled)
    enable_grpo: bool = False

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as f:
            return ExperimentConfig(**json.load(f))
