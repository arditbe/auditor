"""The model-under-test abstraction.

Auditor does not care what it is auditing, only that it can send a prompt and
get text back. Today that is a local Ollama tag; swapping in a real deployed
endpoint means adding one class here, not touching the agent.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class TargetModel(abc.ABC):
    """A model being validated."""

    #: Provider-qualified id, e.g. "ollama:qwen2:0.5b". Echoed into reports.
    spec: str

    @abc.abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> Completion:
        ...

    async def close(self) -> None:
        return None


class _Timer:
    """Wall-clock helper so every provider reports latency the same way.

    `ms` is readable at any point, including from inside an `except` block
    before the context manager has exited.
    """

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        return None

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)
