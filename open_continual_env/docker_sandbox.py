"""
Production-Grade Docker Container Sandbox Provider for OpenContinualEnv.

Runs code execution inside isolated Docker containers with CPU/Memory cgroup limits
and network restriction, falling back to secure process sandboxing if Docker daemon is offline.
"""

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

from open_continual_env.env.sandbox import ExecutionResult, PythonSandbox


class DockerSandbox:
    """
    Production Docker Container Sandbox Provider.
    """

    def __init__(
        self,
        image_name: str = "python:3.12-slim",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        network_mode: str = "none",
        timeout: float = 5.0,
    ):
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_mode = network_mode
        self.timeout = timeout
        self.fallback_sandbox = PythonSandbox()
        self.docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker CLI daemon is available."""
        if not shutil.which("docker"):
            return False
        try:
            res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
            return res.returncode == 0
        except Exception:
            return False

    def execute(self, code: str, test_code: Optional[str] = None) -> ExecutionResult:
        """Execute code in isolated Docker container or fallback subprocess sandbox."""
        if not self.docker_available:
            return self.fallback_sandbox.execute(code, test_code=test_code, timeout=self.timeout)

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "script.py")
            full_content = code
            if test_code:
                full_content += f"\n\n{test_code}\n"

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            cmd = [
                "docker", "run", "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                f"--network={self.network_mode}",
                "-v", f"{temp_dir}:/app",
                "-w", "/app",
                self.image_name,
                "python", "script.py"
            ]

            start_t = time.time()
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=self.timeout)
                exec_time = time.time() - start_t
                pass_rate = 1.0 if proc.returncode == 0 else 0.0
                return ExecutionResult(
                    success=(proc.returncode == 0),
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    tests_passed=1 if proc.returncode == 0 else 0,
                    tests_total=1,
                    pass_rate=pass_rate,
                    execution_time=exec_time,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"DockerSandbox Execution Timed Out (> {self.timeout}s)",
                    exit_code=-1,
                    tests_passed=0,
                    tests_total=1,
                    pass_rate=0.0,
                    execution_time=self.timeout,
                )
