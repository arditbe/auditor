"""Model under test, reachable over HTTP.

This is the path for auditing a genuinely fine-tuned model that the user has
deployed somewhere -- Vertex AI, a Cloud Run inference server, or any
OpenAI-compatible gateway. Not wired into the UI yet; the API accepts it so the
switch from local Ollama to a real endpoint is a config change.
"""
from __future__ import annotations

import httpx

from ..config import settings
from .base import Completion, TargetModel, _Timer


class HttpEndpointTarget(TargetModel):
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        endpoint: str,
        model_name: str = "default",
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.spec = f"http:{model_name}@{self.endpoint}"

        hdrs = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            hdrs["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=settings.target_timeout_s, headers=hdrs
        )

    async def generate(self, prompt: str, system: str | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with _Timer() as t:
            try:
                resp = await self._client.post(
                    f"{self.endpoint}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                return Completion(
                    text="",
                    latency_ms=t.ms,
                    error=f"{type(exc).__name__}: {exc}",
                )

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return Completion(
            text=(choice.get("message") or {}).get("content", "").strip(),
            latency_ms=t.ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def close(self) -> None:
        await self._client.aclose()
