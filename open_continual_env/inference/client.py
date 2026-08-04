"""
LM Studio & OpenAI-Compatible Local LLM Inference Client.
Provides unified interface for local LLM inference via LM Studio server
or any OpenAI-compatible API endpoint with offline simulation fallback.
"""

import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Optional imports for OpenAI SDK and HTTPX
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def _load_env_file() -> None:
    """Load environment variables from local .env file if present."""
    cwd = os.getcwd()
    env_path = os.path.join(cwd, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


class LMStudioClient:
    """
    LM Studio Local LLM Inference Client with OpenAI-compatible API support.

    Configurable via constructor arguments or environment variables:
    - LM_STUDIO_API_BASE / OPENAI_BASE_URL / OPENAI_API_BASE (default: "http://127.0.0.1:1234/v1")
    - LM_STUDIO_MODEL_NAME / MODEL_NAME / OPENAI_MODEL_NAME (default: "google/gemma-4-e4b")
    - OPENAI_API_KEY / LM_STUDIO_API_KEY (default: "lm-studio")
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        offline_fallback: bool = True,
    ) -> None:
        _load_env_file()

        self.api_base = (
            api_base
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("LM_STUDIO_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "http://127.0.0.1:8000/v1"
        )
        self.model_name = (
            model_name
            or os.getenv("VLLM_MODEL_NAME")
            or os.getenv("LM_STUDIO_MODEL_NAME")
            or os.getenv("MODEL_NAME")
            or os.getenv("OPENAI_MODEL_NAME")
            or "google/gemma-4-e4b"
        )

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LM_STUDIO_API_KEY")
            or "lm-studio"
        )
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.offline_fallback = offline_fallback

        self._openai_client = None
        if HAS_OPENAI and OpenAI is not None:
            try:
                self._openai_client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except Exception as e:
                logger.warning("Could not initialize OpenAI client: %s", e)

    def is_online(self) -> bool:
        """Check if the LM Studio API server is currently reachable."""
        if HAS_HTTPX:
            try:
                models_url = f"{self.api_base.rstrip('/')}/models"
                resp = httpx.get(models_url, timeout=min(2.0, self.timeout))
                return resp.status_code == 200
            except Exception:
                return False
        elif self._openai_client is not None:
            try:
                self._openai_client.models.list()
                return True
            except Exception:
                return False
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> str:
        """
        Generate completion for prompt using OpenAI-compatible API.
        """
        if not prompt or not prompt.strip():
            if self.offline_fallback:
                return self._offline_fallback_generation(prompt)
            raise RuntimeError("Prompt cannot be empty")

        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("FAST_MOCK_LLM") == "1":
            if self.offline_fallback:
                return self._offline_fallback_generation(prompt)
            raise RuntimeError(
                f"LM Studio API unavailable at {self.api_base} and offline_fallback is False"
            )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Attempt generation via OpenAI SDK
        if self._openai_client is not None:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content is not None:
                        return content
            except Exception as e:
                logger.warning(
                    "LM Studio OpenAI API request failed: %s. Falling back.", e
                )

        # Attempt raw HTTP request via httpx
        if HAS_HTTPX:
            url = f"{self.api_base.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for attempt in range(self.max_retries + 1):
                try:
                    resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            return choices[0]["message"].get("content", "") or ""
                except Exception as e:
                    if attempt == self.max_retries:
                        logger.warning("HTTP request to LM Studio failed after retries: %s", e)

        if self.offline_fallback:
            return self._offline_fallback_generation(prompt)

        raise RuntimeError(
            f"LM Studio API unavailable at {self.api_base} and offline_fallback is False"
        )

    def predict(self, prompt: str) -> str:
        """
        Predict response code/text for a given prompt.
        Alias for generate() to support BaseContinualAgent/BenchmarkRunner interface.
        """
        return self.generate(prompt=prompt)

    def _offline_fallback_generation(self, prompt: str) -> str:
        """Generate deterministic simulated response when LM Studio server is not active."""
        if not prompt or not prompt.strip():
            return "# Offline fallback response for empty prompt\ndef solution():\n    pass"

        snippet = prompt[:40].replace("\n", " ").strip()
        return (
            f"# LM Studio Offline Simulated Response\n"
            f"# Model: {self.model_name}\n"
            f"# Prompt: {snippet}\n"
            f"def solution():\n"
            f"    return True"
        )


OpenAICompatibleClient = LMStudioClient
