"""Model under test, reachable over HTTP.

This is the path for auditing a genuinely fine-tuned model that the user has
deployed somewhere -- Vertex AI, a Cloud Run inference server, or any
OpenAI-compatible gateway.
"""
from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from ..config import settings
from .base import Completion, TargetModel, _Timer


#: Someone pasting a URL may give the API root ("https://host/v1") or the
#: completions endpoint itself, in any of the shapes a real deployment uses:
#: "/chat/completions", "/completions", "/v1/chat/completions.php". Guessing
#: wrong produces ".../completions.php/chat/completions", which 404s with no
#: hint about why.
def resolve_request_url(endpoint: str) -> str:
    """The URL to POST to, whether `endpoint` is the API root or the route."""
    base = endpoint.rstrip("/")
    # Inspect the path only. A host called "completions.example.com" is a host,
    # not a completions endpoint.
    path = urlsplit(base).path
    if path:
        last = path.rsplit("/", 1)[-1].lower()
        # Matches "completions", "completions.php", "completions.cgi", ...
        if last.split(".", 1)[0] == "completions":
            return base
    return f"{base}/chat/completions"


class HttpEndpointTarget(TargetModel):
    """Calls an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        endpoint: str,
        model_name: str = "default",
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.request_url = resolve_request_url(self.endpoint)
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
                    self.request_url,
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
