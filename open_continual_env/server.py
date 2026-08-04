"""
Production-Grade FastAPI Environment Microservice Server for OpenContinualEnv.

Exposes OpenContinualEnv over HTTP REST endpoints following the Hugging Face OpenEnv Server specification
for deployment on Kubernetes, Ray Clusters, or Hugging Face Spaces.
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_continual_env.env.core_env import OpenContinualEnv, OpenContinualAction


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    code: str
    arguments: Optional[Dict[str, Any]] = None


app = FastAPI(
    title="OpenContinualEnv Server",
    description="Production OpenEnv Environment HTTP Service for Lifelong LLM Learning",
    version="1.0.0",
)

# Global active environment instance
env_instance = OpenContinualEnv()


@app.get("/health")
def health_check():
    """Kubernetes / Load Balancer Health Probe Endpoint."""
    return {"status": "healthy", "service": "open_continual_env_server", "version": "1.0.0"}


@app.post("/reset")
def reset_env(req: ResetRequest):
    """Reset environment state."""
    try:
        obs, info = env_instance.reset(seed=req.seed, options=req.options)
        return {
            "observation": {
                "prompt": obs.prompt,
                "task_id": obs.task_id,
                "execution_result": obs.execution_result,
            },
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step_env(req: StepRequest):
    """Step environment with an action."""
    try:
        action = OpenContinualAction(code=req.code)
        obs, reward, terminated, truncated, info = env_instance.step(action)
        return {
            "observation": {
                "prompt": obs.prompt,
                "task_id": obs.task_id,
                "execution_result": obs.execution_result,
            },
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """Retrieve current state snapshot."""
    return env_instance.state


def launch_server(host: str = "0.0.0.0", port: int = 8000):
    """Launch FastAPI uvicorn production server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    launch_server()
