"""
Unit tests for Production-Grade FastAPI Server & Async OpenContinualEnv.
"""

import asyncio
from fastapi.testclient import TestClient

from open_continual_env.server import app
from open_continual_env.env.async_env import AsyncOpenContinualEnv
from open_continual_env.docker_sandbox import DockerSandbox


def test_fastapi_server_health_and_endpoints():
    client = TestClient(app)
    
    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 2. Reset endpoint
    res = client.post("/reset", json={"seed": 42})
    assert res.status_code == 200
    data = res.json()
    assert "observation" in data
    assert "prompt" in data["observation"]

    # 3. Step endpoint
    code_snippet = "def add(a, b):\n    return a + b\n"
    res = client.post("/step", json={"code": code_snippet})
    assert res.status_code == 200
    step_data = res.json()
    assert "reward" in step_data
    assert "terminated" in step_data
    assert step_data["reward"] >= 0.0

    # 4. State endpoint
    res = client.get("/state")
    assert res.status_code == 200


def test_async_env_interface():
    async def run_test():
        async_env = AsyncOpenContinualEnv()
        obs, info = await async_env.reset_async()
        assert obs.prompt is not None

        obs, reward, term, trunc, info = await async_env.step_async("def test(): pass")
        assert reward >= 0.0

    asyncio.run(run_test())


def test_docker_sandbox_fallback():
    sandbox = DockerSandbox(timeout=2.0)
    res = sandbox.execute("print('hello docker')")
    assert res.execution_time >= 0.0
