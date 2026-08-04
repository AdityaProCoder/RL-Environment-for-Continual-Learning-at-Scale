"""Verified reward: score code by actually running its tests (I2).

Anti-hack: penalizes degenerate 'return literal' solutions that game printed
tests without control/logic, so reward hacking does not masquerade as learning.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .sandbox import ExecutionResult, PythonSandbox


def code_quality(code: str) -> float:
    """1.0 good, 0.5 suspicious (return-without-logic), 0.0 empty."""
    if not code or not isinstance(code, str) or not code.strip():
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    has_logic = has_return = has_def = False
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.Call, ast.BinOp, ast.BoolOp,
                          ast.Compare, ast.ListComp, ast.DictComp, ast.Try)):
            has_logic = True
        if isinstance(n, ast.Return):
            has_return = True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_def = True
    if has_return and not has_logic:
        return 0.5
    if not has_def and not has_logic:
        return 0.5
    return 1.0


@dataclass
class RewardWeights:
    exec: float = 0.3
    test: float = 0.5
    quality: float = 0.1
    safety: float = 0.1


@dataclass
class Verifier:
    sandbox: PythonSandbox = field(default_factory=PythonSandbox)
    weights: RewardWeights = field(default_factory=RewardWeights)

    def _eval_math(self, output: str, reference: str) -> Tuple[float, Dict[str, Any]]:
        import re
        if not output:
            return 0.0, {"match": False, "reason": "empty"}
        ref = str(reference).strip().lower().replace(",", "")
        ex = ""
        if "####" in output:
            ex = output.split("####")[-1].strip().splitlines()[0].strip().lower()
        elif "\\boxed{" in output:
            try: ex = output.split("\\boxed{")[1].split("}")[0].strip().lower()
            except IndexError: ex = ""
        if not ex:
            nums = re.findall(r"-?\d+(?:\.\d+)?", output)
            ex = nums[-1] if nums else ""
        def norm(s):
            s = s.replace(",", "").strip()
            try: return float(s)
            except: return None
        m = norm(ex) is not None and norm(ref) is not None and abs(norm(ex)-norm(ref)) < 1e-4
        return (1.0 if m else 0.0), {"match": m, "extracted": ex, "reference": ref}

    def reward(self, *, domain: str, code: str = "", test_code: str = "",
               reference_answer: str = "", timeout: float = 6.0) -> Tuple[float, Dict[str, Any], ExecutionResult | None]:
        if domain == "math":
            r, info = self._eval_math(code, reference_answer)
            return r, {"reward": r, **info}, None
        res = self.sandbox.execute(code, test_code=test_code, timeout=timeout)
        w = self.weights
        s_exec = 1.0 if res.success or (res.exit_code == 0 and res.error_type is None) else 0.0
        s_test = max(0.0, min(1.0, res.pass_rate))
        s_qual = code_quality(code)
        s_safe = 0.0 if res.safety_violation else 1.0
        total = w.exec*s_exec + w.test*s_test + w.quality*s_qual + w.safety*s_safe
        if res.safety_violation:
            total = 0.05
        total = float(max(0.0, min(1.0, total)))
        info = {"reward": total, "exec": s_exec, "test": s_test, "quality": s_qual,
                "safety": s_safe, "pass_rate": res.pass_rate, "success": res.success,
                "error_type": res.error_type, "tests_passed": res.tests_passed,
                "tests_total": res.tests_total}
        return total, info, res
