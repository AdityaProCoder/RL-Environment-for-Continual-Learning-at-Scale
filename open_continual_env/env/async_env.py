"""
Production-Grade Asynchronous Environment Interface for OpenContinualEnv.

Provides AsyncOpenContinualEnv for high-throughput, non-blocking rollout collection
during RL training (TRL, GRPO, Ray RLlib, SkyRL).
"""

import asyncio
from typing import Any, Dict, Optional, Tuple, Union

from open_continual_env.env.core_env import (
    OpenContinualEnv,
    OpenContinualAction,
    OpenContinualObservation,
)


class AsyncOpenContinualEnv:
    """
    Asynchronous, non-blocking OpenEnv wrapper for production RL training clusters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._sync_env = OpenContinualEnv(config=config)

    async def reset_async(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[OpenContinualObservation, Dict[str, Any]]:
        """Asynchronously reset the environment."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_env.reset, seed, options)

    async def step_async(
        self, action: Union[OpenContinualAction, str, Dict[str, Any]]
    ) -> Tuple[OpenContinualObservation, float, bool, bool, Dict[str, Any]]:
        """Asynchronously step the environment with an action."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_env.step, action)

    async def state_async(self) -> Dict[str, Any]:
        """Asynchronously get current state snapshot."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync_env.state)
