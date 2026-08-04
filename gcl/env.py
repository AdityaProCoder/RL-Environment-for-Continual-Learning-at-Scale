"""GroundedContinualEnv: lifelong MDP with the deployment SafetyGate in-loop.

step(action) executes:  verified reward -> learning op (store/update/consolidate)
-> for UPDATE it performs a gated update: snapshot -> gradient -> holdout gate
(base vs candidate via adapter disable) -> keep (register) or rollback. This is
the safe-continual-weight-update story realized concretely (I1 + I3).
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# engine imports torch at module scope; guard so env/mechanics are testable on a
# bare CPU box (the real TrainingEngine path re-exports the same helpers).
try:
    from .engine import extract_code, _build_prompt
except Exception:  # torch absent (e.g. headless test env)
    def extract_code(text: str) -> str:
        import re
        text = text or ""
        blocks = [m.strip() for m in re.findall(r"```(?:python)?\n(.+?)```", text, flags=re.DOTALL)]
        blocks = [b for b in blocks if b.strip()]
        def _score(b: str) -> int:
            has_def = "def " in b
            stub = re.sub(r"#.*", "", b).strip()
            is_stub = bool(re.match(r"^def\s+\w+\s*\(.*\):\s*(pass|\.\.\.)\s*$", stub))
            return (2 if (has_def and not is_stub) else (1 if has_def else 0))
        if blocks:
            best = max(blocks, key=_score)
            if _score(best) > 0:
                return best.strip()
        matches = list(re.finditer(r"(?m)^(\s*def\s+\w+\s*\(.*)$", text))
        if matches:
            return text[matches[-1].start():].strip()
        return text.strip()

    def _build_prompt(task) -> str:
        if getattr(task, "domain", "code") == "math":
            return (f"Solve and give ONLY the final numeric answer.\n{task.prompt}\nAnswer: ")
        ep = getattr(task, "entry_point", "") or ""
        name_hint = f" The function MUST be named `{ep}`." if ep else ""
        return ("You are an expert Python programmer. Write ONLY the complete function "
                f"inside a ```python block. No analysis, no <think>, no empty 'pass' stubs."
                f"{name_hint}\n{task.prompt}\n```python\n")

try:  # VSR is optional so the frozen controls keep running on old configs
    from .vault import SkillVault
except Exception:  # pragma: no cover
    SkillVault = None


def _ground_prompt(base_prompt: str, retrieved) -> str:
    """Prepend up to `k` verified sibling solutions as in-context examples."""
    if not retrieved:
        return base_prompt
    shots = []
    for t in retrieved:
        code = (t.generated_code or "").strip()
        if code:
            shots.append(f"# Example solution (verified):\n```python\n{code}\n```")
    if not shots:
        return base_prompt
    return ("\n".join(shots) + "\n\n# Now solve the following.\n" + base_prompt)


class LearnOp(str, enum.Enum):
    IGNORE = "ignore"
    STORE = "store"
    UPDATE_LORA = "update_lora"
    CONSOLIDATE = "consolidate"
    REQUEST_REVIEW = "request_review"


@dataclass
class Action:
    answer: str = ""
    learn_op: LearnOp = LearnOp.IGNORE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    task_id: str
    family: str
    prompt: str
    domain: str
    step: int
    mem_stats: Dict[str, Any] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    traj_id: str
    task_id: str
    family: str
    prompt: str
    answer: str
    extracted: str
    reward: float
    pass_rate: float
    success: bool
    learn_op: str
    update_info: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    def to_dict(self):
        return asdict(self)


# Safe terminal observation used when the stream is finished.
from .curriculum import Task as _Task
_TERMINAL_TASK = _Task(task_id="TERMINAL", family="DONE", prompt="", domain="code",
                       test_code="", reference_answer="", entry_point="")


class GroundedContinualEnv:
    def __init__(self, config, engine, verifier, stream, eval_hook=None,
                 gate_epsilon: Optional[float] = None, holdout: Optional[List] = None,
                 vault = None, vsr_gate: Optional[bool] = None):
        self.cfg = config
        self.engine = engine
        self.verifier = verifier
        self.stream = stream
        self.eval_hook = eval_hook
        self.epsilon = config.gate_epsilon if gate_epsilon is None else gate_epsilon
        self.holdout = holdout or []
        self.enabled_ops_budget = config.max_updates
        self.vault = vault            # SkillVault or None (VSR)
        # VSR mechanics are per-learner: reference injection is on iff a vault is
        # attached for THIS learner; the vault-test gate is separately toggleable.
        self._use_reference_injection = bool(vault is not None)
        self._use_vsr_gate = (bool(getattr(config, "use_vsr_gate", False)) if vsr_gate is None
                              else bool(vsr_gate)) and vault is not None
        self.last_retrieved: List = []  # set per-step for recall measurement
        self.reset()

    def build_prompt(self) -> str:
        """VSR-conditioned prompt: prior verified skills as in-context grounding when
        available (forward transfer), else the raw task prompt. Controls see the
        unchanged prompt (`act_prompt` handles non-VSR)."""
        task = self._task()
        base = _build_prompt(task)
        if self._use_reference_injection and self.vault is not None:
            retrieved = self.vault.retrieve(task.prompt, k=self.cfg.vault_retrieve_k)
            self.last_retrieved = retrieved
            return _ground_prompt(base, retrieved)
        return base

    def reset(self):
        self.t = 0
        self.family_idx = 0
        self.task_idx = 0
        self.done = False
        self.trajectories: List[Trajectory] = []
        self.rewards: List[float] = []
        self.update_count = 0
        self.rollback_count = 0
        return self._obs()

    def _family(self):
        return self.stream[self.family_idx]

    def _task(self):
        # Guard terminal boundary: after finishing the last family, done=True is set
        # before _obs() builds the terminal observation; return a sentinel instead of
        # indexing past the end of the task list.
        if self.done or self.family_idx >= len(self.stream):
            return _TERMINAL_TASK
        fam = self._family()
        if self.task_idx >= len(fam.tasks):
            return _TERMINAL_TASK
        return fam.tasks[self.task_idx]

    def _obs(self) -> Observation:
        t = self._task()
        return Observation(task_id=t.task_id, family=t.family, domain=t.domain,
                           prompt=t.prompt, step=self.t,
            mem_stats={"replay": len(getattr(self.engine, "_replay", [])),
                       "vault": len(self.vault) if self.vault is not None else 0,
                       "adapter_version": self.engine.registry.active_version,
                       "rollbacks": self.rollback_count},
                           perf={"recent_mean": (sum(self.rewards[-8:]) / min(8, len(self.rewards))) if self.rewards else 0.0,
                                 "updates": self.update_count})

    def _task_passes(self, code: str, tests: str, ref: str = "") -> bool:
        try:
            _, info, _ = self.verifier.reward(domain="code", code=code,
                                              test_code=tests, reference_answer=ref)
            return float(info.get("pass_rate", 0.0)) >= 1.0 and bool(info.get("success", False))
        except Exception:
            return False

    def _gated_update(self, pairs, op, task=None, candidate_code: str = "",
                      retrieved: Optional[List] = None) -> Dict[str, Any]:
        """Snapshot -> grad update -> safety gate -> keep/rollback (I3).

        Primary gate (VSR): provable vault-test veto — the post-update candidate
        must still pass the CURRENT task's tests AND not break a compact set of
        previously verified skills. The old holdout-epsilon margin is kept only as
        a cheap secondary floor for back-compat configs.
        """
        eng = self.engine
        snap = eng._snapshot()
        use_vsr = bool(getattr(self.cfg, "use_vsr_gate", False)) and self.vault is not None and task is not None and self._use_vsr_gate

        gate: Dict[str, Any] = {"epsilon": self.epsilon, "method": ("vsr" if use_vsr else "holdout_eps")}

        if not use_vsr:
            base_score = eng.holdout_score(self.holdout, self.verifier, adapter_on=False) if self.holdout else 1.0
            cand_before = eng.holdout_score(self.holdout, self.verifier, adapter_on=True) if self.holdout else base_score
            gate.update({"base_score": base_score, "cand_before": cand_before})

        m = eng.apply_update(pairs)

        if use_vsr:
            veto = self.vault.violates(task, candidate_code, self.verifier,
                                       retrieved=retrieved,
                                       check_skills=getattr(self.cfg, "vault_gate_check", 3),
                                       domain=getattr(task, "domain", "code"))
            if self.holdout:  # cheap secondary floor
                base_h = eng.holdout_score(self.holdout, self.verifier, adapter_on=False)
                cand_h = eng.holdout_score(self.holdout, self.verifier, adapter_on=True)
                gate.update({"base_h": base_h, "cand_h": cand_h,
                             "holdout_floor_ok": cand_h >= base_h - self.epsilon})
            gate.update({"veto": veto["veto"], "veto_reason": veto["reason"],
                         "checked": veto["checked"], "broke": veto["broke"]})
            accepted = not veto["veto"]
        else:
            cand_after = eng.holdout_score(self.holdout, self.verifier, adapter_on=True) if self.holdout else 1.0
            accepted = (cand_after >= gate["base_score"] - self.epsilon)
            gate.update({"cand_after": cand_after})
        gate["accepted"] = accepted

        if accepted:
            meta = eng.register_adapter(op, {**m, "gate": gate})
            self.update_count += 1
            return {"executed": True, "accepted": True, "loss": m["loss_end"],
                    "grad_norm": m["grad_norm"], "adapter_version": meta.version,
                    "hash": meta.content_hash, "gate": gate}
        eng._restore(snap)
        self.rollback_count += 1
        return {"executed": True, "accepted": False,
                "reason": ("vault_veto" if use_vsr else "holdout_regression"), "gate": gate}

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        task = self._task()
        extracted = extract_code(action.answer) if task.domain == "code" else action.answer
        reward, info, res = self.verifier.reward(domain=task.domain, code=extracted,
                                                 test_code=task.test_code,
                                                 reference_answer=task.reference_answer)
        op = action.learn_op
        update_info: Dict[str, Any] = {"op": op.value, "executed": False}
        can_update = self.update_count < self.enabled_ops_budget

        use_vsr = self._use_reference_injection and self.vault is not None
        use_vsr_gate = self._use_vsr_gate and self.vault is not None

        # Gold reference may also be supplied at act-time via metadata for the
        # reference-injection / ablation paths; NEVER enters the model prompt.
        gold_from_meta = (action.metadata or {}).get("reference_answer", "")
        gold_available = task.reference_answer or gold_from_meta

        # ---- VSR: retrieve verified skills for grounding (measured for FWT) ----
        retrieved: List = []
        if self.vault is not None:
            try:
                retrieved = self.vault.retrieve(task.prompt, k=self.cfg.vault_retrieve_k)
            except Exception:
                retrieved = []
        self.last_retrieved = retrieved
        recall_hit = False

        # ---- Training target: gold > verified-skill > (correct) self ------------
        passed = float(info.get("pass_rate", 0.0)) >= 1.0 and bool(info.get("success", False))
        if use_vsr:
            cand = self.vault.choose_target(
                task, extracted, reward, self.verifier,
                gold=gold_available, retrieved=retrieved, domain=task.domain)
            target_code = cand.code
            target_source = cand.source
            target_verified = cand.verified
            recall_hit = cand.source == "verified_skill"
            trainable = target_verified  # only train on something proven to pass
        else:
            target_code = extracted
            target_source = "self"
            target_verified = passed
            trainable = len((extracted or "").strip()) > 0

        good_pair = {"prompt": _build_prompt(task), "target": target_code}
        update_info["target_source"] = target_source
        update_info["target_verified"] = target_verified

        if op == LearnOp.STORE:
            self.engine._replay.append(good_pair)
            update_info["executed"] = True
        elif op == LearnOp.UPDATE_LORA and can_update and trainable:
            update_info = {"op": op.value,
                           **self._gated_update([good_pair], "update_lora",
                                                task=task, candidate_code=target_code,
                                                retrieved=retrieved),
                           "target_source": target_source,
                           "target_verified": target_verified}
        elif op == LearnOp.UPDATE_LORA and can_update and not trainable:
            update_info["executed"] = False
            update_info["reason"] = "no_verified_target"
        elif op == LearnOp.CONSOLIDATE and can_update:
            m = self.engine.consolidate_ewc([good_pair])
            update_info = {"op": op.value, "executed": True, "ewc_params": m.get("ewc_params", 0)}
            self.update_count += 1
        elif op == LearnOp.REQUEST_REVIEW:
            update_info = {"op": op.value, "executed": False, "review": "queued"}

        # ---- Corrective update: model wrong but we HAVE verified truth ----
        if use_vsr and can_update and (not passed) and target_verified and \
                target_source in ("gold", "verified_skill") and reward < self.cfg.vault_commit_min:
            # Probe: would this exact verified answer survive the current adapter?
            would_hold = self._task_passes(target_code, task.test_code, task.reference_answer)
            if not would_hold:
                corr = self._gated_update([good_pair], "corrective",
                                          task=task, candidate_code=target_code,
                                          retrieved=retrieved)
                update_info["corrective"] = corr
                update_info["corrective"]["accepted"] = corr.get("accepted", False)

        # ---- Commit a genuinely-new verified skill to the vault ---------------
        if use_vsr and passed:
            self.vault.commit(task, extracted, reward, domain=task.domain,
                              min_reward=self.cfg.vault_commit_min,
                              pass_rate=float(info.get("pass_rate", 0.0)))

        traj = Trajectory(traj_id=f"t{self.t}_{task.task_id}", task_id=task.task_id,
                          family=task.family, prompt=task.prompt, answer=action.answer,
                          extracted=extracted, reward=reward, pass_rate=info.get("pass_rate", 0.0),
                          success=bool(info.get("success", False)), learn_op=op.value,
                          update_info=update_info)
        self.trajectories.append(traj)
        self.rewards.append(reward)
        self.t += 1
        # roll updates committed via a corrective pass (they run through _gated_update,
        # which already increments update_count, so they are not double-counted here)

        self.task_idx += 1
        if self.task_idx >= len(self._family().tasks):
            finished = self.family_idx
            self.family_idx += 1
            self.task_idx = 0
            self.done = self.family_idx >= len(self.stream)
            if self.eval_hook is not None and not self.done:
                try:
                    self.eval_hook(self.engine, finished)
                except Exception:
                    pass
        self.done = self.family_idx >= len(self.stream)

        o = self._obs() if not self.done else Observation(task_id="DONE", family="", prompt="",
                                                          domain="code", step=self.t)
        step_info = {"reward": reward, "verifier": info, "learn_op": op.value,
                     "update_info": update_info, "task_id": traj.task_id, "family": traj.family,
                     "vsr": {"recall_hit": recall_hit,
                             "n_retrieved": len(retrieved),
                             "target_source": update_info.get("target_source", "self"),
                             "vault_size": len(self.vault) if self.vault is not None else 0}}
        return o, reward, self.done, step_info
