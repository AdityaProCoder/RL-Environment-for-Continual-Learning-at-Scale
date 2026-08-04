"""
Python Execution Sandbox & Error Capture Module for OpenContinualEnv (Feature F2)
"""

import sys
import os
import re
import ast
import json
import time
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    tests_passed: int
    tests_total: int
    pass_rate: float
    execution_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    safety_violation: bool = False


RUNNER_SCRIPT = """
import sys
import json
import time
import traceback
import ast

def run():
    start_time = time.perf_counter()
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    code = data.get("code", "")
    test_code = data.get("test_code", "")
    
    stderr_captured = ""
    exit_code = 0
    success = False
    tests_passed = 0
    tests_total = 0
    pass_rate = 0.0
    error_type = None
    error_message = None
    
    namespace = {}
    
    # 1. Compile & Execute Code
    try:
        compiled_code = compile(code, "<sandbox_code>", "exec")
        exec(compiled_code, namespace)
        code_success = True
    except SyntaxError as e:
        code_success = False
        exit_code = 1
        error_type = "SyntaxError"
        error_message = str(e)
        stderr_captured = traceback.format_exc()
    except Exception as e:
        code_success = False
        exit_code = 1
        error_type = type(e).__name__
        error_message = str(e)
        stderr_captured = traceback.format_exc()
        
    if not code_success:
        tests_passed = 0
        tests_total = 1
        pass_rate = 0.0
        success = False
    else:
        if not test_code or not test_code.strip():
            tests_passed = 1
            tests_total = 1
            pass_rate = 1.0
            success = True
        else:
            try:
                compiled_tests = compile(test_code, "<test_code>", "exec")
            except SyntaxError as e:
                exit_code = 1
                error_type = "SyntaxError"
                error_message = str(e)
                stderr_captured = traceback.format_exc()
                tests_passed = 0
                tests_total = 1
                pass_rate = 0.0
                success = False
                compiled_tests = None

            if compiled_tests is not None:
                lines = [line.strip() for line in test_code.splitlines() if line.strip() and not line.strip().startswith("#")]
                assert_lines = [line for line in lines if line.startswith("assert")]
                
                if assert_lines:
                    tests_total = len(assert_lines)
                    for stmt in assert_lines:
                        try:
                            exec(stmt, namespace)
                            tests_passed += 1
                        except Exception as e:
                            stderr_captured += f"\\nAssertion failed ({stmt}): {e}"
                            if not error_type:
                                error_type = type(e).__name__
                                error_message = str(e)
                    pass_rate = tests_passed / tests_total if tests_total > 0 else 0.0
                    success = (tests_passed == tests_total)
                    if not success:
                        exit_code = 1
                else:
                    try:
                        exec(compiled_tests, namespace)
                        tests_passed = 1
                        tests_total = 1
                        pass_rate = 1.0
                        success = True
                    except Exception as e:
                        exit_code = 1
                        error_type = type(e).__name__
                        error_message = str(e)
                        stderr_captured = traceback.format_exc()
                        tests_passed = 0
                        tests_total = 1
                        pass_rate = 0.0
                        success = False

    exec_time = time.perf_counter() - start_time
    
    result = {
        "exit_code": exit_code,
        "success": success,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "pass_rate": pass_rate,
        "execution_time": exec_time,
        "error_type": error_type,
        "error_message": error_message,
        "stderr_extra": stderr_captured,
    }
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

if __name__ == "__main__":
    run()
"""


class PythonSandbox:
    """Isolated Python code execution sandbox using subprocesses."""

    def __init__(self):
        pass

    def _check_safety(self, code: str) -> Optional[str]:
        """Check code string for dangerous calls or imports."""
        if not code or not isinstance(code, str):
            return None

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
                return f"SecurityViolation: Unsafe pattern matched: {pattern}"

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("subprocess", "shutil", "socket"):
                            return f"SecurityViolation: Forbidden import '{alias.name}'"
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ("subprocess", "shutil", "socket"):
                        return f"SecurityViolation: Forbidden import from '{node.module}'"
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "__import__"):
                        return f"SecurityViolation: Forbidden function call '{node.func.id}'"
        except SyntaxError:
            pass

        return None

    def execute(
        self,
        code: str,
        test_code: str = "",
        timeout: float = 5.0
    ) -> ExecutionResult:
        """Executes code and optional test_code inside an isolated subprocess."""
        # Safety inspection
        safety_issue = self._check_safety(code) or self._check_safety(test_code)
        if safety_issue:
            return ExecutionResult(
                stdout="",
                stderr=safety_issue,
                exit_code=1,
                success=False,
                tests_passed=0,
                tests_total=1,
                pass_rate=0.0,
                execution_time=0.0,
                error_type="SecurityViolation",
                error_message=safety_issue,
                safety_violation=True,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_data = {"code": code or "", "test_code": test_code or ""}
            input_file = os.path.join(temp_dir, "input.json")
            runner_file = os.path.join(temp_dir, "runner.py")
            result_file = os.path.join(temp_dir, "result.json")

            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(input_data, f)

            with open(runner_file, "w", encoding="utf-8") as f:
                f.write(RUNNER_SCRIPT)

            start_time = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, runner_file],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed_time = time.perf_counter() - start_time
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stdout="",
                    stderr=f"TimeoutError: Execution timed out after {timeout} seconds",
                    exit_code=-1,
                    success=False,
                    tests_passed=0,
                    tests_total=1,
                    pass_rate=0.0,
                    execution_time=timeout,
                    error_type="TimeoutError",
                    error_message=f"Execution timed out after {timeout} seconds",
                    safety_violation=False,
                )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        res_data = json.load(f)

                    extra_stderr = res_data.get("stderr_extra") or ""
                    if extra_stderr and extra_stderr not in stderr:
                        stderr = (stderr + "\n" + extra_stderr).strip()

                    error_type = res_data.get("error_type")
                    error_message = res_data.get("error_message")

                    if not stderr and error_type:
                        stderr = f"{error_type}: {error_message}"

                    return ExecutionResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=res_data.get("exit_code", proc.returncode),
                        success=res_data.get("success", proc.returncode == 0),
                        tests_passed=res_data.get("tests_passed", 0),
                        tests_total=res_data.get("tests_total", 0),
                        pass_rate=res_data.get("pass_rate", 0.0),
                        execution_time=res_data.get("execution_time", elapsed_time),
                        error_type=error_type,
                        error_message=error_message,
                        safety_violation=False,
                    )
                except Exception as e:
                    stderr += f"\nFailed to parse execution result JSON: {e}"

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr or "Execution failed without result JSON",
                exit_code=proc.returncode if proc.returncode != 0 else 1,
                success=False,
                tests_passed=0,
                tests_total=1,
                pass_rate=0.0,
                execution_time=elapsed_time,
                error_type="ExecutionError",
                error_message="Execution failed without result JSON",
                safety_violation=False,
            )

    def run_unit_tests(
        self,
        code: str,
        unit_tests: List[str],
        timeout: float = 10.0
    ) -> ExecutionResult:
        """Executes code against a list of unit test strings."""
        test_code = "\n".join(unit_tests) if unit_tests else ""
        return self.execute(code, test_code=test_code, timeout=timeout)
