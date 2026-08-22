"""Receiving a model the user picked in their browser.

Typing a filesystem path only ever works when Auditor runs on the same machine
as the model. Uploading works everywhere, which is why it is the primary way
in: the browser's folder picker sends the files, and Auditor treats the result
exactly like a local folder.

LoRA adapters are what makes this practical -- a rank-8 adapter for a 7B model
is about 21 MB, because the base model is named in the config rather than
shipped.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from pathlib import Path

from .prepared import AUDITOR_HOME

log = logging.getLogger(__name__)

UPLOADS_DIR = AUDITOR_HOME / "uploads"

#: Generous for adapters, far too small for full model weights. Uploading a
#: 14 GB base model through a browser is not a workflow worth supporting.
MAX_UPLOAD_BYTES = 400 * 1024 * 1024
MAX_FILES = 40

#: Only files that are part of a model. Anything else in the folder the user
#: picked is ignored rather than uploaded.
ALLOWED_SUFFIXES = {".safetensors", ".json", ".bin", ".gguf", ".model", ".txt", ".jinja"}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(ValueError):
    """Rejected upload. The message is user-facing."""


def safe_component(name: str) -> str:
    """One path segment, stripped of anything that could escape the directory."""
    cleaned = _UNSAFE.sub("-", name.strip()).strip("-.")
    # "..", "." and empty names all collapse to something inert.
    return cleaned or "file"


def safe_relative_path(raw: str) -> Path:
    """Flatten a browser-supplied relative path into a safe one.

    `webkitRelativePath` is attacker-controlled in the general case, so every
    component is sanitised and any traversal is discarded rather than resolved.
    """
    parts = [
        safe_component(part)
        for part in Path(raw.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    if not parts:
        raise UploadError(f"Cannot store a file named {raw!r}.")
    # Drop the folder the user picked; keep only its interior structure so the
    # upload directory itself is the model root.
    return Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])


def is_useful(relative: Path) -> bool:
    """Whether a file is worth storing.

    Skips MLX's periodic checkpoints: `0000200_adapters.safetensors` is an
    intermediate the final `adapters.safetensors` supersedes, and uploading
    all of them triples the transfer for no benefit.
    """
    if relative.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    if re.match(r"^\d+_adapters\.safetensors$", relative.name):
        return False
    if relative.name.startswith("."):
        return False
    return True


def new_upload_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    folder = UPLOADS_DIR / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    return folder


def prune_uploads(keep_hours: float = 24.0) -> int:
    """Delete upload directories older than `keep_hours`. Returns how many."""
    if not UPLOADS_DIR.is_dir():
        return 0
    cutoff = time.time() - keep_hours * 3600
    removed = 0
    for folder in UPLOADS_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
