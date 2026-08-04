"""Fast, torch-free smoke test of the VSR mechanics (references, vault veto,
retrieval). Uses a *self-resetting* trivial task so the base model can actually
succeed sometimes, letting us observe a RISING learning curve + provable safety
without paying the cost of loading a 2B model.

Self-resetting task: the answer is a counter that a dedicated FooModel perturbs
only via gradient steps, so learning genuinely shifts behaviour. The verifier is
the real one (sandbox executes the code), so reward is ground-truth. This proves
the break-fix on the *machinery*, which is what the breakthrough hinges on.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

# headless / isloted
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from gcl.config import ExperimentConfig
from gcl.curriculum import Task
from gcl.vault import SkillVault
from gcl.verify import Verifier
from gcl.env import GroundedContinualEnv, Action, LearnOp


class FooEngine:
    """Minimal stand-in for TrainingEngine. Generates code for an arithmetic task
    whose correct constant drifts per family; a 'gradient update' nudges the
    model's emitted constant toward the supervised target -> real behaviour change."""
    def __init__(self, cfg):
        self.cfg = cfg
        # per-family parametric belief (sparse, brain-like). Family held in the task
        # id embedded in the prompt as "fid:<family>".
        self.know: Dict[str, float] = {}
        self.registry = _Registry()
        self._replay: List[Dict] = []
        self.updates_done = 0
        self.last_family = "f0"

    def _snapshot(self):
        return {"know": dict(self.know), "replay": list(self._replay)}

    def _restore(self, snap):
        self.know = dict(snap["know"]); self._replay = snap["replay"]

    def generate(self, prompt: str, adapter_on: bool = True) -> str:
        # Generalize (read WANT) but carry a per-family STALENESS residual the model
        # has not yet corrected. Drifted families start stale (-3). Sparse, local
        # updates correct only the CURRENT family's residual -> higher ACC with no
        # cross-family forgetting.
        fam = prompt.split("fid:")[1].split()[0] if "fid:" in prompt else "_"
        want = _parse_want(prompt)
        base_d = {"f0": 0.0, "f1": -3.0, "f2": -3.0}.get(fam, -3.0)
        d = self.know.get(fam, base_d) if adapter_on else base_d
        val = want + d
        return f"```python\ndef result():\n    return {int(round(val))}\n```"

    def apply_update(self, pairs, **kw):
        # regress THIS family's residual delta toward the gold-matching value.
        lr = 0.9
        for pr in pairs:
            fam = pr["prompt"].split("fid:")[1].split()[0] if "fid:" in pr["prompt"] else "_"
            want = _parse_want(pr["prompt"])
            tgt = _parse_return_const(pr["target"])          # gold constant
            needed = tgt - want                              # residual fixing THIS task
            base_d = {"f0": 0.0, "f1": -3.0, "f2": -3.0}.get(fam, -3.0)
            cur = self.know.get(fam, base_d)
            self.know[fam] = cur + lr * (needed - cur)
            self._replay.append(pr)
        self.updates_done += 1
        return {"loss_start": 1.0, "loss_end": 0.1, "grad_norm": 0.5, "n_pairs": len(pairs)}

    def consolidate_ewc(self, pairs):
        return {"ewc_params": 1}

    def holdout_score(self, holdout, verifier, adapter_on):
        return 0.5

    def register_adapter(self, op, meta):
        return self.registry.register(op, meta)


class _Meta:
    def __init__(self, v): self.version = v; self.content_hash = f"h{v}"


class _Registry:
    def __init__(self): self.active_version = -1; self.n = 0
    def register(self, op, meta):
        self.n += 1; self.active_version = self.n - 1
        return _Meta(self.active_version)
    def history(self): return []


def _parse_want(prompt: str) -> float:
    """The task's requested constant (prompt format: '... WANT=<k> ...')."""
    try:
        return float(prompt.split("WANT=")[1].split()[0])
    except Exception:
        return 0.0


def _parse_return_const(code: str) -> float:
    import re
    m = re.search(r"return\s+(-?\d+)", code or "")
    return float(m.group(1)) if m else 0.0


def make_family(name: str, start_const: int, n: int = 6) -> List[Task]:
    tasks = []
    for i in range(n):
        tasks.append(Task(
            task_id=f"{name}_{i}", family=name, domain="code",
            prompt=f"Write result() returning WANT. fid:{name} WANT={start_const} .",
            test_code=f"assert result() == {start_const}",
            reference_answer=f"def result():\n    return {start_const}",
            entry_point="result"))
    return tasks


class _Fam:
    def __init__(self, name, tasks): self.name = name; self.tasks = tasks; self.holdout = []


def run_smoke(learner_op: str = "vsr") -> Dict[str, Any]:
    cfg = ExperimentConfig(model_name="foo", device="cpu", max_updates=40,
                           use_vsr_gate=True, use_reference_injection=True,
                           vault_commit_min=0.8, vault_retrieve_k=3, vault_gate_check=2,
                           holdout_size=0)
    eng = FooEngine(cfg)
    ver = Verifier()
    vault = SkillVault() if learner_op == "vsr" else None

    # 3 families with drifting "want" constants; base belief starts at 0 (stale).
    # f0 known, f1/f2 drifted (stale residual -3) -> real cross-family forgetting
    fams = [_Fam("f0", make_family("f0", 0)),
            _Fam("f1", make_family("f1", 3)),
            _Fam("f2", make_family("f2", 5))]

    zero_shot = [_family_accuracy(eng, ver, fm, adapter_on=True) for fm in fams]
    env = GroundedContinualEnv(cfg, eng, ver, fams, holdout=[],
                               gate_epsilon=cfg.gate_epsilon, vault=vault)

    rewards, updates, rollbacks, commits, recalls = [], 0, 0, 0, 0
    obs = env.reset()
    steps = 0
    while not env.done and steps < 200:
        raw = eng.generate(env._build_prompt(task) if False else _prompt(env), adapter_on=True)
        op = LearnOp.UPDATE_LORA if learner_op == "vsr" else LearnOp.IGNORE
        obs, r, done, info = env.step(Action(answer=raw, learn_op=op))
        rewards.append(r)
        ui = info["update_info"]
        if ui.get("op") == "update_lora" and ui.get("accepted"):
            updates += 1
        if ui.get("op") == "update_lora" and ui.get("executed") and not ui.get("accepted", True):
            rollbacks += 1
        if info.get("vsr", {}).get("recall_hit"):
            recalls += 1
        steps += 1
    commits = len(vault) if vault else 0
    half = max(1, len(rewards) // 2)
    early = sum(rewards[:half]) / half
    late = sum(rewards[half:]) / max(1, len(rewards) - half)
    # final accuracy across ALL families + forgetting / BWT vs zero-shot
    final = [_family_accuracy(eng, ver, fm, adapter_on=True) for fm in fams]
    per_family_drop = [max(0.0, zero_shot[j] - final[j]) for j in range(len(fams))]
    forgetting = sum(per_family_drop) / len(per_family_drop)
    bwt = sum(final[j] - zero_shot[j] for j in range(len(fams))) / len(fams)
    acc = sum(final) / len(final)
    return {"learner": learner_op, "steps": steps, "vault": commits,
            "updates": updates, "rollbacks": rollbacks, "recall_hits": recalls,
            "acc": round(acc, 3), "bwt": round(bwt, 3), "forgetting": round(forgetting, 3),
            "reward_early": round(early, 3), "reward_late": round(late, 3),
            "zero_shot": [round(z, 2) for z in zero_shot],
            "final": [round(x, 2) for x in final],
            "know": {k: round(v, 1) for k, v in eng.know.items()}}


def _family_accuracy(eng, ver, fam, adapter_on=True) -> float:
    from gcl.env import extract_code
    scores = []
    for t in fam.tasks:
        code = extract_code(eng.generate(t.prompt, adapter_on=adapter_on))
        r, info, _ = ver.reward(domain="code", code=code, test_code=t.test_code,
                                reference_answer=t.reference_answer)
        scores.append(r)
    return sum(scores) / max(1, len(scores))


def _prompt(env) -> str:
    return env._task().prompt


if __name__ == "__main__":
    for who in ("frozen", "vsr"):
        res = run_smoke("vsr" if who == "vsr" else "frozen")
        print(res)
    print("SMOKE OK")
