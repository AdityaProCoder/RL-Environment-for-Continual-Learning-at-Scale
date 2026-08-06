"""Self-taught (STaR-style) rollout: find a *verified* solution without gold.

This is what makes the no-gold learner real continual learning rather than
pass@1-rooftopper memorization:

  1. sample K diverse candidates (temperature > 0)   — pass@K coverage
  2. if none pass: in-context SELF-REPAIR — feed back (code, traceback) and
     resample K2 greedy-ish candidates                   — learn from env feedback
  3. verify each candidate in the sandbox; choose the *shortest* fully-loading
     passing solution                                     — anti-degenerate tie-break
  4. the verified solution becomes the training target; it's committed to the
     vault so it can bootstrap future, related failures.

Everything is reward-grounded; no reference answer is ever read. The vault is
not a cache of gold — it's a cache of *self-solved, sandbox-verified* programs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# engine is heavy (torch); fall back to env's copy when torch isn't available.,
try:
    from .engine import extract_code, _build_prompt
except Exception:  # pragma: no cover
    from .env import extract_code, _build_prompt


def _err_signature(res) -> str:
    if res is None:
        return ""
    parts = []
    if getattr(res, "error_type", None):
        parts.append(str(res.error_type))
    if getattr(res, "stderr", None):
        parts.append(str(res.stderr)[:400])
    if getattr(res, "error_message", None):
        parts.append(str(res.error_message)[:400])
    return "\n".join(parts)


def _repair_prompt(task: Any, code: str, err: str) -> str:
    return (
        f"The following solution to the task FAILED its tests. Find and fix the bug.\n\n"
        f"TASK:\n{_build_prompt(task)}\n\n"
        f"FAILING CODE:\n```python\n{code}\n```\n\n"
        f"EXECUTION / TEST FAILURE:\n```\n{err}\n```\n\n"
        f"Rewrite ONLY the corrected function inside a ```python block. No analysis.\n"
    )


def self_taught_solve(engine, verifier, task: Any,
                      k_samples: int = 4, temp: float = 0.7,
                      repair_rounds: int = 1, repair_k: int = 2,
                      commit_min: float = 0.9) -> Dict[str, Any]:
    """Try to produce a fully-passing program for `task` from the live model.

    Returns a dict with: found, code, samples_used, repair_used, reward, pass_rate,
    info. The training pipeline only consumes entries with `found == True`.
    """
    prompt = _build_prompt(task)
    candidates = engine.sample_candidates(prompt, n=k_samples, temperature=temp,
                                          adapter_on=True)

    best: Optional[Dict[str, Any]] = None
    samples_used = len(candidates)

    for txt in candidates:
        code = extract_code(txt)
        if not code or not code.strip():
            continue
        r, info, res = verifier.reward(domain=task.domain, code=code,
                                       test_code=task.test_code, reference_answer="")
        pr = float(info.get("pass_rate", 0.0))
        if pr >= 1.0 and bool(info.get("success", False)):
            # prefer shortest passing solution (less degenerate on held-out)
            if best is None or len(code) < len(best["code"]):
                best = {"code": code, "reward": float(r), "pass_rate": 1.0, "info": info}

    if best is not None:
        return {"found": True, "code": best["code"], "samples_used": samples_used,
                "repair_used": False, "reward": best["reward"], "pass_rate": 1.0,
                "info": best["info"]}

    # ---------- self-repair loop ----------
    repair_used = False
    last_code = ""
    last_err = ""
    for txt in candidates:
        code = extract_code(txt)
        if code and code.strip():
            last_code = code
            _, _, res = verifier.reward(domain=task.domain, code=code,
                                        test_code=task.test_code, reference_answer="")
            last_err = _err_signature(res)

    for _ in range(max(0, repair_rounds)):
        rp = _repair_prompt(task, last_code, last_err)
        rcands = engine.sample_candidates(rp, n=repair_k, temperature=max(0.2, temp - 0.4),
                                          adapter_on=True,
                                          max_new_tokens=getattr(engine.cfg, "max_new_tokens", 256))
        samples_used += len(rcands)
        for txt in rcands:
            code = extract_code(txt)
            if not code or not code.strip():
                continue
            r, info, res = verifier.reward(domain=task.domain, code=code,
                                           test_code=task.test_code, reference_answer="")
            pr = float(info.get("pass_rate", 0.0))
            if pr >= 1.0 and bool(info.get("success", False)):
                repair_used = True
                return {"found": True, "code": code, "samples_used": samples_used,
                        "repair_used": True, "reward": float(r), "pass_rate": 1.0,
                        "info": info}
            last_code = code or last_code
            last_err = _err_signature(res) or last_err

    return {"found": False, "code": "", "samples_used": samples_used,
            "repair_used": repair_used, "reward": 0.0, "pass_rate": 0.0, "info": {}}
