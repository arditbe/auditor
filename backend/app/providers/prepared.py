"""Models Auditor prepared on the user's behalf.

Someone who fine-tuned a model has a folder of LoRA adapters, not something
that can answer a question. This module closes that gap: it fuses the adapters
into their base model, then runs the result behind a local OpenAI-compatible
server so the rest of Auditor can treat it like any other target.

Everything lives under ~/.auditor/models. The registry is a small JSON file so
prepared models survive a restart and show up in the dropdown next time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

AUDITOR_HOME = Path(
    os.environ.get("AUDITOR_HOME", Path.home() / ".auditor")
).expanduser()
MODELS_DIR = AUDITOR_HOME / "models"
REGISTRY_PATH = AUDITOR_HOME / "registry.json"

#: Ports handed out to managed model servers.
PORT_RANGE = range(8090, 8130)

#: A 7B model takes a little while to load into memory the first time.
SERVER_START_TIMEOUT_S = 180.0


class PrepareError(RuntimeError):
    """Something went wrong preparing a model. The message is user-facing."""


@dataclass
class PreparedModel:
    name: str
    path: str
    kind: str                 # "fused" | "folder"
    base_model: str | None = None
    adapter_path: str | None = None
    created_at: float = 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "created_at": self.created_at,
        }

    @property
    def spec(self) -> str:
        return f"prepared:{self.name}"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _load_registry() -> dict[str, PreparedModel]:
    try:
        raw = json.loads(REGISTRY_PATH.read_text())
    except (OSError, ValueError):
        return {}

    out: dict[str, PreparedModel] = {}
    for entry in raw.get("models", []):
        try:
            model = PreparedModel(**entry)
        except TypeError:
            continue
        # Drop entries whose files were deleted behind our back rather than
        # offering the user a model that cannot load.
        if Path(model.path).exists():
            out[model.name] = model
    return out


def _save_registry(models: dict[str, PreparedModel]) -> None:
    AUDITOR_HOME.mkdir(parents=True, exist_ok=True)
    payload = {"models": [m.to_json() for m in models.values()]}
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(REGISTRY_PATH)


def list_prepared() -> list[PreparedModel]:
    return sorted(_load_registry().values(), key=lambda m: -m.created_at)


def get_prepared(name: str) -> PreparedModel | None:
    return _load_registry().get(name)


def register(model: PreparedModel) -> None:
    models = _load_registry()
    models[model.name] = model
    _save_registry(models)


def unregister(name: str, delete_files: bool = False) -> bool:
    models = _load_registry()
    model = models.pop(name, None)
    if model is None:
        return False
    _save_registry(models)

    if delete_files:
        path = Path(model.path)
        # Only ever delete inside our own directory.
        if MODELS_DIR in path.parents:
            shutil.rmtree(path, ignore_errors=True)
    return True


# --------------------------------------------------------------------------
# base model resolution
# --------------------------------------------------------------------------

def resolve_base_model(reference: str) -> str:
    """Turn a base-model reference into something mlx_lm can load offline.

    A local path is used as-is. A Hugging Face repo id is resolved to its
    cached snapshot directory, because mlx_lm's offline lookup rejects a
    snapshot that is missing even a README, which is common in real caches.
    """
    candidate = Path(reference).expanduser()
    if candidate.is_dir():
        return str(candidate)

    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    if cache_root.name != "hub":
        cache_root = cache_root / "hub"

    repo_dir = cache_root / f"models--{reference.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        # Newest snapshot; a repo usually has exactly one.
        options = [d for d in snapshots.iterdir() if d.is_dir()]
        if options:
            newest = max(options, key=lambda d: d.stat().st_mtime)
            if (newest / "config.json").exists():
                return str(newest)

    # Not cached. Hand back the repo id so mlx_lm can download it if online.
    return reference


# --------------------------------------------------------------------------
# fusing
# --------------------------------------------------------------------------

def _unique_name(preferred: str) -> str:
    existing = set(_load_registry())
    if preferred not in existing and not (MODELS_DIR / preferred).exists():
        return preferred
    for n in range(2, 100):
        candidate = f"{preferred}-{n}"
        if candidate not in existing and not (MODELS_DIR / candidate).exists():
            return candidate
    raise PrepareError(f"Too many models named like {preferred!r}.")


async def fuse_adapters(
    *, adapter_path: str, base_model: str, name: str
) -> PreparedModel:
    """Combine LoRA adapters with their base model into one runnable model."""
    resolved_base = resolve_base_model(base_model)
    final_name = _unique_name(name)
    save_path = MODELS_DIR / final_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", resolved_base,
        "--adapter-path", adapter_path,
        "--save-path", str(save_path),
    ]
    log.info("fusing %s + %s -> %s", resolved_base, adapter_path, save_path)

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = await process.communicate()
    output = (stdout or b"").decode(errors="replace")

    if process.returncode != 0:
        shutil.rmtree(save_path, ignore_errors=True)
        raise PrepareError(_explain_fuse_failure(output, base_model))

    if not (save_path / "config.json").exists():
        shutil.rmtree(save_path, ignore_errors=True)
        raise PrepareError(
            "Fusing finished but produced no model. Check that the adapter "
            "folder matches the base model it was trained on."
        )

    model = PreparedModel(
        name=final_name,
        path=str(save_path),
        kind="fused",
        base_model=base_model,
        adapter_path=adapter_path,
        created_at=time.time(),
    )
    register(model)
    return model


def _explain_fuse_failure(output: str, base_model: str) -> str:
    """Turn an mlx_lm traceback into something actionable."""
    tail = output.strip().splitlines()[-1] if output.strip() else ""

    if "IncompleteSnapshotError" in output or "local_files_only" in output:
        return (
            f"The base model {base_model} is not fully downloaded. Connect to "
            "the internet and try again, or download it first."
        )
    if "RepositoryNotFoundError" in output or "404" in output:
        return (
            f"Could not find the base model {base_model}. Check the name in "
            "your adapter_config.json."
        )
    if "size mismatch" in output.lower() or "shape" in output.lower():
        return (
            "These adapters do not match that base model — the layer shapes "
            "differ. They were probably trained on a different model."
        )
    if "No module named" in output:
        return (
            "mlx-lm is not installed in Auditor's environment. Install it "
            "with: pip install mlx-lm"
        )
    return f"Fusing failed. {tail}" if tail else "Fusing failed."


# --------------------------------------------------------------------------
# exporting to Ollama
# --------------------------------------------------------------------------

#: mlx_lm can only write GGUF for these architectures.
GGUF_EXPORTABLE = {"llama", "mistral", "mixtral"}

#: Peak disk during export: fp16 weights, the f16 GGUF, and Ollama's copy.
#: Roughly 4.5x the 4-bit model, with headroom.
_EXPORT_DISK_MULTIPLIER = 5.0


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


async def export_to_ollama(
    name: str, *, tag: str | None = None, quantize: str | None = "q4_K_M"
) -> str:
    """Convert a prepared model to GGUF and register it with Ollama.

    This is the heavyweight path. It re-fuses from the original base and
    adapters with the weights de-quantized, because GGUF cannot represent
    MLX's 4-bit format, then hands the result to Ollama.

    Returns the Ollama tag. The model then behaves like any other Ollama
    model -- it survives a restart and needs no managed server.
    """
    model = get_prepared(name)
    if model is None:
        raise PrepareError(f"No prepared model named {name!r}.")
    if not model.base_model or not model.adapter_path:
        raise PrepareError(
            f"{name} was not built from adapters, so it cannot be re-exported. "
            "Point Auditor at the original adapter folder instead."
        )

    config_path = Path(model.path) / "config.json"
    try:
        model_type = json.loads(config_path.read_text()).get("model_type", "")
    except (OSError, ValueError):
        model_type = ""
    if model_type not in GGUF_EXPORTABLE:
        raise PrepareError(
            f"Ollama export supports {', '.join(sorted(GGUF_EXPORTABLE))} "
            f"models; this one is {model_type or 'an unknown type'}. Use it as "
            "a prepared model instead — it audits exactly the same."
        )

    needed = int(_dir_size(Path(model.path)) * _EXPORT_DISK_MULTIPLIER)
    free = _free_bytes(MODELS_DIR)
    if free < needed:
        raise PrepareError(
            f"Not enough disk space: this needs about {needed / 1e9:.0f} GB "
            f"free and there is {free / 1e9:.0f} GB. The model already works "
            "as a prepared model without exporting."
        )

    converter = find_gguf_converter()
    if converter is None:
        raise PrepareError(
            "Converting to GGUF needs llama.cpp's convert_hf_to_gguf.py, which "
            "is not installed. Set LLAMA_CPP_PATH to a llama.cpp checkout, or "
            "skip this — the model already audits fine as a prepared model."
        )

    out_tag = gguf_safe_tag(tag or name)
    staging = MODELS_DIR / f".export-{out_tag}"
    shutil.rmtree(staging, ignore_errors=True)

    try:
        # Step 1: rebuild at full precision. GGUF cannot represent MLX's 4-bit
        # format, so the weights have to be de-quantized on the way out.
        log.info("exporting %s: de-quantizing", name)
        await _run(
            [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", resolve_base_model(model.base_model),
                "--adapter-path", model.adapter_path,
                "--save-path", str(staging),
                "--dequantize",
            ],
            lambda out: _explain_fuse_failure(out, model.base_model),
        )

        # Step 2: convert with llama.cpp. mlx_lm has its own --export-gguf, but
        # in 0.31 it writes every tensor with shape (0,) -- a silently empty
        # model. Auditing a corrupt export would be worse than not exporting.
        log.info("exporting %s: converting to gguf", name)
        gguf_path = staging / "model-f16.gguf"
        await _run(
            [
                sys.executable, str(converter), str(staging),
                "--outfile", str(gguf_path),
                "--outtype", "f16",
            ],
            lambda out: f"GGUF conversion failed. {_last_line(out)}",
        )

        if not gguf_path.exists() or gguf_path.stat().st_size < 100_000_000:
            raise PrepareError(
                "Conversion produced no usable GGUF file. The model was not "
                "exported."
            )

        from .gguf import import_gguf

        return await import_gguf(str(gguf_path), out_tag, quantize=quantize)
    finally:
        # Ollama copies what it needs into its own store, so the ~30 GB of
        # intermediates are pure waste once import succeeds -- or once it fails.
        shutil.rmtree(staging, ignore_errors=True)


def _last_line(output: str) -> str:
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


async def _run(cmd: list[str], explain) -> str:
    """Run a subprocess, raising PrepareError with a readable message."""
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = await process.communicate()
    output = (stdout or b"").decode(errors="replace")
    if process.returncode != 0:
        raise PrepareError(explain(output))
    return output


def find_gguf_converter() -> Path | None:
    """Locate llama.cpp's convert_hf_to_gguf.py, if it is available."""
    candidates: list[Path] = []
    configured = os.environ.get("LLAMA_CPP_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates += [
        Path.home() / "llama.cpp",
        Path("/opt/homebrew/share/llama.cpp"),
        Path("/usr/local/share/llama.cpp"),
    ]

    for base in candidates:
        script = base / "convert_hf_to_gguf.py"
        if script.is_file():
            return script
        if base.name == "convert_hf_to_gguf.py" and base.is_file():
            return base
    return None


def gguf_safe_tag(name: str) -> str:
    from .gguf import safe_tag

    return safe_tag(name)


def register_folder(*, path: str, name: str) -> PreparedModel:
    """Register an already-complete model folder, without copying it."""
    folder = Path(path).expanduser()
    if not (folder / "config.json").exists():
        raise PrepareError(
            f"{folder.name} has no config.json, so it is not a complete model "
            "folder."
        )
    model = PreparedModel(
        name=_unique_name(name),
        path=str(folder),
        kind="folder",
        created_at=time.time(),
    )
    register(model)
    return model


# --------------------------------------------------------------------------
# managed servers
# --------------------------------------------------------------------------

def _free_port() -> int:
    for port in PORT_RANGE:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise PrepareError("No free port available for a local model server.")


@dataclass
class _Server:
    port: int
    process: asyncio.subprocess.Process
    model_path: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


#: Running servers, keyed by prepared-model name.
_servers: dict[str, _Server] = {}
_start_lock = asyncio.Lock()


async def _wait_until_ready(port: int, process, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise PrepareError(
                    "The model server stopped immediately after starting. "
                    "The model may be too large for available memory."
                )
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.75)
    raise PrepareError(
        f"The model server did not come up within {int(timeout)}s."
    )


async def ensure_server(name: str) -> tuple[str, str]:
    """Start (or reuse) a server for a prepared model.

    Returns the API base URL and the model id that server expects.
    """
    model = get_prepared(name)
    if model is None:
        raise PrepareError(f"No prepared model named {name!r}.")

    async with _start_lock:
        running = _servers.get(name)
        if running and running.process.returncode is None:
            return running.base_url, running.model_path

        port = _free_port()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mlx_lm", "server",
            "--model", model.path,
            "--port", str(port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            await _wait_until_ready(port, process, SERVER_START_TIMEOUT_S)
        except PrepareError:
            process.terminate()
            raise

        server = _Server(port=port, process=process, model_path=model.path)
        _servers[name] = server
        log.info("model server for %s ready on :%d", name, port)
        return server.base_url, server.model_path


async def stop_all_servers() -> None:
    """Shut down every managed server. Called on application shutdown."""
    for name, server in list(_servers.items()):
        if server.process.returncode is None:
            server.process.terminate()
            try:
                await asyncio.wait_for(server.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                server.process.kill()
        _servers.pop(name, None)


def mlx_available() -> bool:
    """Whether this machine can fuse and serve MLX models."""
    try:
        import mlx_lm  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return sys.platform == "darwin"
