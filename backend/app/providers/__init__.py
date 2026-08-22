"""Target-model construction from a provider-qualified spec string."""
from __future__ import annotations

from .base import Completion, TargetModel
from .http_endpoint import HttpEndpointTarget
from .ollama import OllamaTarget, list_ollama_models
from .registry import (
    DEFAULT_VALIDATOR,
    VALIDATORS,
    ValidatorSpec,
    get_validator,
    list_validators,
)

__all__ = [
    "Completion",
    "TargetModel",
    "OllamaTarget",
    "HttpEndpointTarget",
    "list_ollama_models",
    "build_target",
    "VALIDATORS",
    "ValidatorSpec",
    "DEFAULT_VALIDATOR",
    "get_validator",
    "list_validators",
]


def build_target(spec: str, **kwargs) -> TargetModel:
    """Build the model under test from e.g. "ollama:qwen2:0.5b".

    The scheme is everything before the first colon; the rest is provider
    specific (Ollama tags themselves contain colons, so we split once only).

    """
    scheme, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(
            f"Malformed target spec {spec!r}; expected '<provider>:<model>'"
        )

    if scheme == "ollama":
        return OllamaTarget(rest, **kwargs)
    if scheme in ("http", "https"):
        # The scheme is part of the URL here, not a provider prefix, so the
        # whole spec is the endpoint.
        return HttpEndpointTarget(spec, **kwargs)
    if scheme == "endpoint":
        return HttpEndpointTarget(kwargs.pop("endpoint", None) or rest, **kwargs)

    raise ValueError(f"Unsupported target provider {scheme!r}")
