"""
Modular Reward Pipeline Module for OpenContinualEnv (Feature F4)
"""

import ast
import re
import math
from typing import Tuple, Dict, Any, Optional, Union


class RewardEngine:
    """Modular, configurable reward engine for evaluating code generation trajectories."""

    def __init__(
        self,
        execution_weight: float = 0.4,
        unit_test_weight: float = 0.4,
        efficiency_weight: float = 0.1,
        safety_weight: float = 0.1,
        w_exec: Optional[float] = None,
        w_test: Optional[float] = None,
        w_eff: Optional[float] = None,
        w_safety: Optional[float] = None,
    ):
        exec_w = w_exec if w_exec is not None else execution_weight
        test_w = w_test if w_test is not None else unit_test_weight
        eff_w = w_eff if w_eff is not None else efficiency_weight
        safe_w = w_safety if w_safety is not None else safety_weight

        if any(w < 0.0 for w in (exec_w, test_w, eff_w, safe_w)):
            raise ValueError("Reward weights must be non-negative")

        self.execution_weight = float(exec_w)
        self.unit_test_weight = float(test_w)
        self.efficiency_weight = float(eff_w)
        self.safety_weight = float(safe_w)

        self.w_exec = self.execution_weight
        self.w_test = self.unit_test_weight
        self.w_eff = self.efficiency_weight
        self.w_safety = self.safety_weight

    def _evaluate_safety(self, code: str) -> float:
        """Evaluate code safety using AST and regex inspection."""
        if not code or not isinstance(code, str):
            return 1.0

        unsafe_patterns = [
            r"__import__",
            r"os\.(system|popen|spawn|exec|remove|unlink|rmdir)",
            r"subprocess",
            r"shutil",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"rm\s+-rf",
        ]
        for pattern in unsafe_patterns:
            if re.search(pattern, code):
                return 0.0

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os", "subprocess", "shutil", "sys", "socket"):
                            return 0.0
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ("os", "subprocess", "shutil", "sys", "socket"):
                        return 0.0
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "__import__"):
                        return 0.0
        except SyntaxError:
            pass

        return 1.0

    def _evaluate_code_quality(self, code: str) -> float:
        """
        Evaluate code quality to prevent reward hacking.
        Penalizes code that lacks structural logic (e.g., just returning a hardcoded literal).
        Returns a score between 0.0 (hacky) and 1.0 (good).
        """
        if not code or not isinstance(code, str):
            return 1.0
            
        try:
            tree = ast.parse(code)
            has_logic = False
            has_return = False
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Call, ast.BinOp, ast.Compare, ast.ListComp, ast.DictComp)):
                    has_logic = True
                if isinstance(node, ast.Return):
                    has_return = True
                    
            # If the code just returns something without any logic, it might be a hardcoded hack to pass a specific test.
            if has_return and not has_logic:
                return 0.5  # Penalize score
        except SyntaxError:
            pass
            
        return 1.0

    def _evaluate_efficiency(self, execution_time: float, code: str) -> float:
        """Evaluate efficiency score based on execution time and code length."""
        if execution_time is None or math.isnan(execution_time) or math.isinf(execution_time):
            time_score = 0.0
        else:
            time_score = math.exp(-max(0.0, float(execution_time)))

        if code and isinstance(code, str):
            line_count = len(code.splitlines())
            length_score = max(0.0, 1.0 - line_count / 100.0)
        else:
            length_score = 1.0

        return 0.5 * time_score + 0.5 * length_score

    def calculate_reward(
        self,
        execution_result: Any = None,
        test_result: Any = None,
        efficiency_metrics: Any = None,
        safety_status: Any = None,
        code: str = ""
    ) -> Tuple[float, dict]:
        """Calculates structured reward breakdown dictionary and total scalar reward float."""
        if (
            self.execution_weight == 0.0
            and self.unit_test_weight == 0.0
            and self.efficiency_weight == 0.0
            and self.safety_weight == 0.0
        ):
            return 0.0, {
                "execution_score": 0.0,
                "unit_test_score": 0.0,
                "efficiency_score": 0.0,
                "safety_score": 0.0,
                "weighted_execution": 0.0,
                "weighted_unit_test": 0.0,
                "weighted_efficiency": 0.0,
                "weighted_safety": 0.0,
                "safety_penalty": 0.0,
                "total_reward": 0.0,
            }

        success = False
        pass_rate = 0.0
        exec_time = 0.0

        if execution_result is not None:
            if isinstance(execution_result, dict):
                success = bool(execution_result.get("success", False))
                pass_rate = float(execution_result.get("pass_rate", 1.0 if success else 0.0))
                exec_time = float(execution_result.get("execution_time", 0.0))
            else:
                success = bool(getattr(execution_result, "success", False))
                pass_rate = float(getattr(execution_result, "pass_rate", 1.0 if success else 0.0))
                exec_time = float(getattr(execution_result, "execution_time", 0.0))

        if test_result is not None:
            if isinstance(test_result, (int, float)):
                pass_rate = float(test_result)
            elif isinstance(test_result, dict):
                pass_rate = float(test_result.get("pass_rate", 0.0))

        if efficiency_metrics is not None:
            if isinstance(efficiency_metrics, (int, float)):
                exec_time = float(efficiency_metrics)
            elif isinstance(efficiency_metrics, dict):
                exec_time = float(efficiency_metrics.get("execution_time", 0.0))

        exec_score = 1.0 if success else 0.0

        if math.isnan(pass_rate) or math.isinf(pass_rate):
            test_score = 0.0
        else:
            test_score = max(0.0, min(1.0, float(pass_rate)))

        eff_score = self._evaluate_efficiency(exec_time, code)

        if safety_status is not None and isinstance(safety_status, (int, float)):
            safety_score = max(0.0, min(1.0, float(safety_status)))
        else:
            safety_score = self._evaluate_safety(code)
            
        quality_score = self._evaluate_code_quality(code)

        w_exec_val = self.execution_weight * exec_score
        w_test_val = self.unit_test_weight * test_score
        w_eff_val = self.efficiency_weight * eff_score
        w_safety_val = self.safety_weight * safety_score

        safety_penalty = self.safety_weight if safety_score < 0.5 else 0.0
        quality_penalty = 0.2 if quality_score < 0.8 else 0.0

        raw_total = w_exec_val + w_test_val + w_eff_val + w_safety_val - safety_penalty - quality_penalty

        if math.isnan(raw_total) or math.isinf(raw_total):
            total_reward = 0.0
        else:
            total_reward = max(0.0, min(1.0, float(raw_total)))

        breakdown = {
            "execution_score": float(exec_score),
            "unit_test_score": float(test_score),
            "efficiency_score": float(eff_score),
            "safety_score": float(safety_score),
            "quality_score": float(quality_score),
            "weighted_execution": float(w_exec_val),
            "weighted_unit_test": float(w_test_val),
            "weighted_efficiency": float(w_eff_val),
            "weighted_safety": float(w_safety_val),
            "safety_penalty": float(safety_penalty),
            "quality_penalty": float(quality_penalty),
            "total_reward": float(total_reward),
        }

        return float(total_reward), breakdown

    def compute_reward(
        self,
        execution_result: Any = None,
        code: str = ""
    ) -> float:
        """Returns scalar reward float."""
        reward, _ = self.calculate_reward(execution_result=execution_result, code=code)
        return reward

    def compute_math_reward(self, model_output: str, reference_answer: str) -> Tuple[float, dict]:
        """Evaluates math reasoning output against reference answer."""
        if not model_output or not reference_answer:
            return 0.0, {"match": False, "extracted": "", "reference": reference_answer}

        ref_clean = str(reference_answer).strip().lower()
        extracted = ""

        # Strategy 1: Look for #### pattern
        if "####" in model_output:
            extracted = model_output.split("####")[-1].strip().split("\n")[0].strip().lower()
        # Strategy 2: Look for \boxed{...} pattern
        elif "\\boxed{" in model_output:
            try:
                extracted = model_output.split("\\boxed{")[1].split("}")[0].strip().lower()
            except IndexError:
                extracted = ""
        # Strategy 3: Look for "answer is" pattern
        elif "answer is" in model_output.lower():
            match = re.search(r'answer is\s*[:\s]*([-\d\.\,\/]+)', model_output, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().lower()

        # Strategy 4: Fallback to last number or fraction in text
        if not extracted:
            numbers = re.findall(r'-?\d+(?:[\.,]\d+)*(?:/\d+)?', model_output)
            if numbers:
                extracted = numbers[-1].strip().lower()

        # Helper to convert strings (including commas and fractions like "3/4" or "1,000") to float
        def _parse_val(val_str: str) -> Optional[float]:
            if not val_str:
                return None
            s = val_str.replace(",", "").strip()
            if "/" in s:
                parts = s.split("/")
                if len(parts) == 2:
                    try:
                        n, d = float(parts[0]), float(parts[1])
                        if d != 0:
                            return n / d
                    except ValueError:
                        pass
            try:
                return float(s)
            except ValueError:
                return None

        val_extracted = _parse_val(extracted)
        val_ref = _parse_val(ref_clean)

        is_match = False
        if val_extracted is not None and val_ref is not None:
            if abs(val_extracted - val_ref) < 1e-4:
                is_match = True

        if not is_match:
            is_match = (extracted.replace(",", "").strip() == ref_clean.replace(",", "").strip())

        reward = 1.0 if is_match else 0.0
        return reward, {
            "match": is_match,
            "extracted": extracted,
            "reference": reference_answer,
            "total_reward": reward
        }

