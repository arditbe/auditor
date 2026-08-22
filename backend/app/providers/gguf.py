"""Registering a GGUF file with Ollama.

A GGUF is a single self-contained file, so Ollama can adopt it in seconds
without copying anything. This is the shortest path from "I downloaded a
quantized model" to "it is in the dropdown".
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

#: Ollama tags allow lowercase alphanumerics plus a few separators.
_TAG_SAFE = re.compile(r"[^a-z0-9._-]+")


class GgufImportError(RuntimeError):
    """Import failed. The message is user-facing."""


def safe_tag(name: str) -> str:
    tag = _TAG_SAFE.sub("-", name.strip().lower()).strip("-._")
    return tag or "my-model"


async def import_gguf(
    gguf_path: str, name: str, *, quantize: str | None = None
) -> str:
    """Register a .gguf file with Ollama. Returns the resulting model tag.

    `quantize` (e.g. "q4_K_M") shrinks the model as Ollama adopts it. Use it
    for a freshly exported f16 GGUF; leave it unset for a file the user
    downloaded, which is almost always quantized already.
    """
    path = Path(gguf_path).expanduser()
    if not path.is_file():
        raise GgufImportError(f"{path} is not a file.")
    if path.suffix.lower() != ".gguf":
        raise GgufImportError(f"{path.name} is not a .gguf file.")

    tag = safe_tag(name)

    # Ollama reads the Modelfile from disk, so it has to be a real file.
    with tempfile.TemporaryDirectory() as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(f"FROM {path}\n")

        cmd = ["ollama", "create", tag, "-f", str(modelfile)]
        if quantize:
            cmd += ["-q", quantize]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()

    output = (stdout or b"").decode(errors="replace")
    if process.returncode != 0:
        raise GgufImportError(_explain(output, path))

    log.info("imported %s into ollama as %s", path.name, tag)
    return tag


def _explain(output: str, path: Path) -> str:
    lowered = output.lower()
    if "no such file" in lowered or "not found" in lowered:
        if "ollama" in lowered and "executable" in lowered:
            return (
                "Ollama is not installed, or not on PATH. Install it from "
                "ollama.com and try again."
            )
        return f"Ollama could not read {path.name}."
    if "unsupported" in lowered or "unknown model architecture" in lowered:
        return (
            f"Ollama does not support this GGUF's architecture. It may have "
            "been made with an incompatible converter version."
        )
    if "permission denied" in lowered:
        return f"Auditor is not allowed to read {path}. Check file permissions."

    tail = output.strip().splitlines()[-1] if output.strip() else ""
    return f"Ollama could not import the file. {tail}".strip()


async def ollama_available() -> bool:
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama", "--version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await process.wait()
        return process.returncode == 0
    except (OSError, FileNotFoundError):
        return False
