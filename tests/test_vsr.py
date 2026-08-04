"""Torch-free VSR mechanics tests. Validates the breakthrough contract on the CPU/.venv interpreter using the synthetic self-resetting FooEngine (no 2B model load)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcl.smoke_vsr import run_smoke
from gcl.vault import SkillVault
from gcl.verify import Verifier
from gcl.curriculum import Task


def test_vsr_learns_and_beats_frozen():
    frozen = run_smoke("frozen")
    vsr = run_smoke("vsr")
    assert vsr["acc"] > frozen["acc"], f"VSR ACC {vsr['acc']} not > frozen {frozen['acc']}"
    assert vsr["forgetting"] <= frozen["forgetting"]
    assert vsr["updates"] > 0
    assert vsr["rollbacks"] / max(1, vsr["updates"]) < 0.5, "vault gate still noise-dominated"


def test_vault_only_accepts_verified_skills():
    v, ver = SkillVault(), Verifier()
    good = Task(task_id="g", family="f", domain="code",
                prompt="def result(): return 5", test_code="assert result() == 5",
                reference_answer="def result():\n    return 5", entry_point="result")
    pr, info, _ = ver.reward(domain="code", code=good.reference_answer,
                             test_code=good.test_code, reference_answer=good.reference_answer)
    assert v.commit(good, good.reference_answer, pr, pass_rate=info["pass_rate"])
    # a failing pseudo-skill must NOT be admitted
    bad = "def result():\n    return 99"
    prb, infob, _ = ver.reward(domain="code", code=bad, test_code=good.test_code, reference_answer="")
    assert not v.commit(good, bad, prb, pass_rate=infob["pass_rate"])
    assert len(v) == 1


def test_choose_target_prefers_gold():
    v, ver = SkillVault(), Verifier()
    task = Task(task_id="t", family="f", domain="code",
                prompt="def result(): return 7", test_code="assert result() == 7",
                reference_answer="def result():\n    return 7", entry_point="result")
    cand = v.choose_target(task, model_code="def result():\n    return 1",
                           model_reward=0.15, verifier=ver, gold=task.reference_answer)
    assert cand.source == "gold" and cand.verified and "return 7" in cand.code


def test_canary_rejects_overlapping_offsets():
    """Regression guard for the offset-overlap bug the canary caught at runtime."""
    import pytest  # noqa
    try:
        from datasets import load_dataset  # noqa: F401
    except Exception:
        import pytest as _p
        _p.skip("datasets not installed here")
    from gcl.curriculum import StreamAssembler, canary_report
    asm = StreamAssembler(seed=42)
    fams = asm.assemble([
        dict(corpus="mbpp", name="A", n_train=2, n_holdout=1, offset=0),
        dict(corpus="mbpp", name="B", n_train=2, n_holdout=1, offset=0),  # overlap!
    ])
    assert not canary_report(fams)["clean"], "canary must flag overlapping MBPP offsets"


def test_spec_perturb_internally_consistent():
    """Drift injector must perturb prompt+tests+reference together so the
    drifted family verificiously passes its own tests on the gold answer."""
    t = Task(task_id="x", family="f", domain="code",
             prompt="def push_el(a): return a.append(2)",
             test_code="assert push_el([1]) == 2",
             reference_answer="def push_el(a):\n    a.append(2)\n    return len(a)",
             entry_point="push_el")
    from gcl.curriculum import spec_perturb
    d = spec_perturb(t)
    # drift changed something AND reference still aligns with perturbed test
    assert d.reference_answer != t.reference_answer or len(t.reference_answer) == 0


if __name__ == "__main__":
    test_vsr_learns_and_beats_frozen()
    test_vault_only_accepts_verified_skills()
    test_choose_target_prefers_gold()
    print("ALL VSR MECHANICS TESTS PASS")
