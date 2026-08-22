"""Model under test, served by a local Ollama daemon."""
from __future__ import annotations

import httpx

from ..config import settings
from .base import Completion, TargetModel, _Timer


class OllamaTarget(TargetModel):
    def __init__(self, model_tag: str, host: str | None = None) -> None:
        self.model_tag = model_tag
        self.spec = f"ollama:{model_tag}"
        self._host = (host or settings.ollama_host).rstrip("/")
        self._client = httpx.AsyncClient(timeout=settings.target_timeout_s)

    async def generate(self, prompt: str, system: str | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with _Timer() as t:
            try:
                resp = await self._client.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": self.model_tag,
                        "messages": messages,
                        "stream": False,
                        # Low temperature: we are measuring the model, so we want
                        # its typical behaviour, not its most creative sample.
                        "options": {"temperature": 0.2},
                        # The judge and the model under test share one Ollama
                        # daemon and alternate every probe. Without this, Ollama
                        # evicts whichever ran last and every turn pays a full
                        # model load -- which is what a "timeout" here usually is.
                        "keep_alive": settings.ollama_keep_alive,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
                return Completion(
                    text="",
                    latency_ms=t.ms,
                    error=f"{type(exc).__name__}: {exc}",
                )

        return Completion(
            text=(data.get("message") or {}).get("content", "").strip(),
            latency_ms=t.ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def close(self) -> None:
        await self._client.aclose()


async def list_ollama_models(host: str | None = None) -> list[dict]:
    """Everything pulled locally -- this is the 'choose your model' dropdown."""
    base = (host or settings.ollama_host).rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base}/api/tags")
        resp.raise_for_status()
        models = resp.json().get("models", [])

    out = []
    for m in models:
        name = m.get("name", "")
        details = m.get("details") or {}
        size = m.get("size") or 0
        out.append(
            {
                "spec": f"ollama:{name}",
                "name": name,
                "size_bytes": size,
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family"),
                "is_local": is_local_tag(name, size),
            }
        )
    # Local first, then smallest first: the default selection should be the
    # model that answers fastest, not whichever sorts first alphabetically.
    return sorted(
        out, key=lambda m: (not m["is_local"], m["size_bytes"], m["name"])
    )


#: No real set of weights is smaller than this. A tag under it is a stub
#: manifest for a model that actually runs on Ollama's servers.
MIN_LOCAL_WEIGHTS_BYTES = 1_000_000


def is_local_tag(name: str, size_bytes: int) -> bool:
    """Whether an Ollama tag has real weights on this machine.

    Cloud tags cannot be audited offline, and they are not consistently named:
    both `deepseek-v3.2:cloud` and `gemma4:31b-cloud` appear in `ollama list`.
    Size is the reliable signal -- a cloud tag is a manifest of a few hundred
    bytes -- so both are checked.
    """
    tag = name.rsplit(":", 1)[-1] if ":" in name else ""
    if "cloud" in tag.lower():
        return False
    return size_bytes >= MIN_LOCAL_WEIGHTS_BYTES
