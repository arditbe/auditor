"""Work out what a person actually handed us.

Most people who fine-tune a model do not know what Ollama is. They have a
folder their training script wrote, or a file they downloaded, or a URL. This
module takes that one string and says, in plain language, what it is and
whether Auditor can audit it yet.

It never guesses silently: if something is missing, the detection says exactly
what is missing and what to do about it.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    OLLAMA_TAG = "ollama_tag"
    GGUF_FILE = "gguf_file"
    MLX_ADAPTERS = "mlx_adapters"
    PEFT_ADAPTERS = "peft_adapters"
    MODEL_DIR = "model_dir"
    SERVER_URL = "server_url"
    UNKNOWN = "unknown"


class Readiness(str, Enum):
    #: Usable right now; `spec` is set.
    READY = "ready"
    #: Auditor can make it usable with one action.
    NEEDS_PREPARE = "needs_prepare"
    #: The person has to fix something first.
    BLOCKED = "blocked"


class Detection(BaseModel):
    kind: SourceKind = SourceKind.UNKNOWN
    readiness: Readiness = Readiness.BLOCKED
    #: One line naming what was found, in the person's terms.
    title: str = ""
    #: A sentence or two: what it is, and what happens next.
    detail: str = ""
    #: Target spec, once the thing is actually usable.
    spec: str | None = None
    #: Label for the button that makes it usable, when NEEDS_PREPARE.
    action_label: str | None = None
    #: Base model an adapter was trained on.
    base_model: str | None = None
    #: Resolved path to the adapter weights or model file we settled on.
    resolved_path: str | None = None
    #: Suggested display name, used when registering with Ollama.
    suggested_name: str = ""
    warnings: list[str] = Field(default_factory=list)


# MLX writes `adapters.safetensors` plus periodic `0000400_adapters.safetensors`
# checkpoints. PEFT writes `adapter_model.safetensors`.
_MLX_FINAL = "adapters.safetensors"
_MLX_CHECKPOINT = re.compile(r"^(\d+)_adapters\.safetensors$")
_PEFT_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

_URL_PREFIXES = ("http://", "https://")


def _slugify(name: str) -> str:
    """A name Ollama will accept as a model tag."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-._")
    return slug or "my-model"


def _newest_mlx_checkpoint(folder: Path) -> Path | None:
    """Highest-numbered `NNNNNNN_adapters.safetensors`, if any.

    A training run that was stopped before it finished writing the final
    adapter leaves only checkpoints. That is still a perfectly auditable
    model, so take the latest rather than declaring nothing was found.
    """
    best: tuple[int, Path] | None = None
    for entry in folder.iterdir():
        match = _MLX_CHECKPOINT.match(entry.name)
        if not match:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, entry)
    return best[1] if best else None


def _read_adapter_config(folder: Path) -> dict:
    try:
        return json.loads((folder / "adapter_config.json").read_text())
    except (OSError, ValueError):
        return {}


def _describe_adapters(folder: Path) -> Detection:
    """An adapter folder: LoRA weights that need a base model to run."""
    config = _read_adapter_config(folder)
    peft_weights = [w for w in _PEFT_WEIGHTS if (folder / w).exists()]

    # PEFT names the base in `base_model_name_or_path`; MLX names it `model`.
    base = (
        config.get("base_model_name_or_path")
        or config.get("model")
        or None
    )

    if peft_weights:
        return Detection(
            kind=SourceKind.PEFT_ADAPTERS,
            readiness=Readiness.BLOCKED,
            title=f"LoRA adapters for {base or 'an unknown base model'}",
            detail=(
                "These were trained with PEFT/Hugging Face. Auditor can only "
                "prepare MLX adapters automatically right now. Merge these "
                "into the base model yourself, then point Auditor at the "
                "merged folder or a GGUF export of it."
            ),
            base_model=base,
            resolved_path=str(folder / peft_weights[0]),
        )

    weights = folder / _MLX_FINAL
    used_checkpoint = False
    if not weights.exists():
        checkpoint = _newest_mlx_checkpoint(folder)
        if checkpoint is None:
            return Detection(
                kind=SourceKind.MLX_ADAPTERS,
                readiness=Readiness.BLOCKED,
                title="Adapter settings, but no trained weights",
                detail=(
                    f"{folder.name} has an adapter_config.json but no "
                    "adapters.safetensors and no checkpoints. This training "
                    "run did not save any weights, so there is nothing to "
                    "audit yet."
                ),
                base_model=base,
            )
        weights = checkpoint
        used_checkpoint = True

    if not base:
        return Detection(
            kind=SourceKind.MLX_ADAPTERS,
            readiness=Readiness.BLOCKED,
            title="Adapters found, but the base model is unknown",
            detail=(
                "adapter_config.json does not record which model these were "
                "trained on, so Auditor cannot rebuild the full model."
            ),
            resolved_path=str(weights),
        )

    warnings: list[str] = []
    if used_checkpoint:
        step = _MLX_CHECKPOINT.match(weights.name)
        warnings.append(
            f"No final adapters.safetensors — using checkpoint at "
            f"{int(step.group(1)) if step else '?'} steps."
        )

    return Detection(
        kind=SourceKind.MLX_ADAPTERS,
        readiness=Readiness.NEEDS_PREPARE,
        title=f"Your fine-tune of {base}",
        detail=(
            "Auditor will combine your adapters with the base model and start "
            "it locally, then it appears in the list like any other model."
        ),
        action_label="Prepare this model",
        base_model=base,
        resolved_path=str(folder),
        suggested_name=_slugify(folder.name),
        warnings=warnings,
    )


def _largest_gguf(folder: Path) -> Path | None:
    files = [f for f in folder.glob("*.gguf") if f.is_file()]
    if not files:
        return None
    # A split or multi-quant folder: the biggest file is the real model.
    return max(files, key=lambda f: f.stat().st_size)


def _describe_gguf(path: Path) -> Detection:
    size_gb = path.stat().st_size / 1e9
    return Detection(
        kind=SourceKind.GGUF_FILE,
        readiness=Readiness.NEEDS_PREPARE,
        title=f"A GGUF model file ({size_gb:.1f} GB)",
        detail=(
            "Auditor will register this with Ollama so it can be audited. "
            "This takes a few seconds and does not copy the file."
        ),
        action_label="Add this model",
        resolved_path=str(path),
        suggested_name=_slugify(path.stem),
    )


def _describe_model_dir(folder: Path) -> Detection:
    """A full model folder — safetensors weights plus a config."""
    config_path = folder / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        config = {}

    arch = (config.get("architectures") or [None])[0]
    quantized = "quantization" in config or "quantization_config" in config
    label = arch or folder.name

    warnings = []
    if quantized:
        warnings.append(
            "This model is already quantized, which limits what it can be "
            "converted to."
        )

    return Detection(
        kind=SourceKind.MODEL_DIR,
        readiness=Readiness.NEEDS_PREPARE,
        title=f"A complete model folder ({label})",
        detail=(
            "Auditor will start this model locally so it can be audited."
        ),
        action_label="Prepare this model",
        resolved_path=str(folder),
        suggested_name=_slugify(folder.name),
        warnings=warnings,
    )


def _describe_path(path: Path) -> Detection:
    if path.is_file():
        if path.suffix.lower() == ".gguf":
            return _describe_gguf(path)
        # People paste the config or a weights file rather than the folder.
        if path.name == "adapter_config.json" or path.name.endswith(
            ".safetensors"
        ):
            return _describe_path(path.parent)
        return Detection(
            title=f"{path.name} is not a model file",
            detail=(
                "Auditor understands .gguf files, and folders containing "
                "adapters or model weights. Try selecting the folder your "
                "training run produced."
            ),
        )

    if not path.is_dir():
        return Detection(
            title="Nothing at that path",
            detail=f"{path} does not exist. Check the path and try again.",
        )

    if (path / "adapter_config.json").exists():
        return _describe_adapters(path)

    gguf = _largest_gguf(path)
    if gguf:
        return _describe_gguf(gguf)

    if (path / "config.json").exists() and (
        any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    ):
        return _describe_model_dir(path)

    # A common miss: they picked the parent of the folder they meant.
    candidates = [
        child.name
        for child in path.iterdir()
        if child.is_dir() and (child / "adapter_config.json").exists()
    ]
    if candidates:
        listed = ", ".join(sorted(candidates)[:4])
        return Detection(
            title="No model here, but there is one just inside",
            detail=(
                f"{path.name} contains adapter folders: {listed}. "
                "Select one of those instead."
            ),
        )

    return Detection(
        title="No model found in that folder",
        detail=(
            f"{path.name} has no adapters, .gguf file, or model weights in it. "
            "Pick the folder your training run wrote its output to."
        ),
    )


def inspect_source(source: str, known_ollama_tags: set[str] | None = None) -> Detection:
    """Identify whatever the person typed or picked.

    `known_ollama_tags` lets an already-installed model be recognised without
    a round trip to the daemon.
    """
    text = (source or "").strip().strip("'\"")
    if not text:
        return Detection(
            title="Nothing entered",
            detail="Paste a folder path, a .gguf file, or a model server URL.",
        )

    if text.startswith(_URL_PREFIXES):
        base = text.rstrip("/")
        # People paste the chat route; the target wants the API root.
        for suffix in ("/chat/completions", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        return Detection(
            kind=SourceKind.SERVER_URL,
            readiness=Readiness.READY,
            title="A model server",
            detail=(
                f"Auditor will send probes to {base}. It must accept "
                "OpenAI-style /chat/completions requests."
            ),
            spec=base,
            resolved_path=base,
            suggested_name=_slugify(base.split("//")[-1]),
        )

    if known_ollama_tags and text in known_ollama_tags:
        return Detection(
            kind=SourceKind.OLLAMA_TAG,
            readiness=Readiness.READY,
            title=f"{text} is already installed",
            detail="This model is ready to audit.",
            spec=f"ollama:{text}",
            suggested_name=_slugify(text),
        )

    expanded = Path(text).expanduser()
    if expanded.is_absolute() or text.startswith(("~", ".", "/")):
        return _describe_path(expanded)

    # Not a path and not installed — most likely a mistyped tag.
    if "/" not in text:
        return Detection(
            title=f"No model named {text}",
            detail=(
                "That is not an installed model, a folder, or a URL. Paste "
                "the full path to your model folder."
            ),
        )

    return _describe_path(expanded)
