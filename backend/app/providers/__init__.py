"""Target-model construction from a provider-qualified spec string."""
from __future__ import annotations

from .base import Completion, TargetModel
from .detect import Detection, Readiness, SourceKind, inspect_source
from .gguf import GgufImportError, import_gguf, ollama_available
from .http_endpoint import HttpEndpointTarget
from .ollama import OllamaTarget, list_ollama_models
from .prepared import (
    PrepareError,
    PreparedModel,
    ensure_server,
    export_to_ollama,
    find_gguf_converter,
    fuse_adapters,
    list_prepared,
    mlx_available,
    register_folder,
    stop_all_servers,
    unregister,
)
from .uploads import (
    MAX_FILES,
    MAX_UPLOAD_BYTES,
    UploadError,
    is_useful,
    new_upload_dir,
    prune_uploads,
    safe_component,
    safe_relative_path,
)
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
    # model onboarding
    "Detection",
    "Readiness",
    "SourceKind",
    "inspect_source",
    "GgufImportError",
    "import_gguf",
    "ollama_available",
    "PrepareError",
    "PreparedModel",
    "ensure_server",
    "export_to_ollama",
    "find_gguf_converter",
    "fuse_adapters",
    "list_prepared",
    "mlx_available",
    "register_folder",
    "stop_all_servers",
    "unregister",
    # uploads
    "MAX_FILES",
    "MAX_UPLOAD_BYTES",
    "UploadError",
    "is_useful",
    "new_upload_dir",
    "prune_uploads",
    "safe_component",
    "safe_relative_path",
]


async def build_target(spec: str, **kwargs) -> TargetModel:
    """Build the model under test from e.g. "ollama:qwen2:0.5b".

    The scheme is everything before the first colon; the rest is provider
    specific (Ollama tags themselves contain colons, so we split once only).

    Async because a `prepared:` model may need its local server started, which
    can take a while for a 7B model loading into memory.
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
    if scheme == "prepared":
        # A model Auditor fused for the user; served locally on demand.
        base_url, model_id = await ensure_server(rest)
        target = HttpEndpointTarget(base_url, model_name=model_id, **kwargs)
        # Report the friendly name rather than the internal URL.
        target.spec = spec
        return target

    raise ValueError(f"Unsupported target provider {scheme!r}")
