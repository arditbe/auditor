"""Auditor HTTP API.

No auth by design -- this is a public demo. The one thing that is guarded is
run creation, which is rate-limited per client so a public URL cannot be used
to burn Vertex AI credits.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent.prompts import SUITES
from .agent.validator_agent import reset_cache as reset_agent_cache
from .config import settings
from .context_upload import extract_uploaded_context
from .events import STREAM_END, EventType, bus
from .models.schemas import RunConfig
from .orchestrator import ACTIVE, MAX_PROBES, start_run
from .providers import (
    MAX_FILES,
    MAX_UPLOAD_BYTES,
    GgufImportError,
    PrepareError,
    export_to_ollama,
    find_gguf_converter,
    fuse_adapters,
    import_gguf,
    inspect_source,
    is_useful,
    get_validator,
    list_ollama_models,
    list_prepared,
    list_validators,
    mlx_available,
    new_upload_dir,
    ollama_available,
    prune_uploads,
    register_folder,
    safe_component,
    safe_relative_path,
    stop_all_servers,
    unregister,
    UploadError,
)
from .store import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("auditor")

# ADK reads these from the environment, not from our Settings object.
if settings.google_cloud_project:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.google_cloud_project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)
    os.environ.setdefault(
        "GOOGLE_GENAI_USE_VERTEXAI",
        "1" if settings.google_genai_use_vertexai else "0",
    )

app = FastAPI(title="Auditor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------

RATE_LIMIT_RUNS = 10
RATE_LIMIT_WINDOW_S = 300
_run_starts: dict[str, deque[float]] = defaultdict(deque)


def _enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.time()
    history = _run_starts[client]
    while history and now - history[0] > RATE_LIMIT_WINDOW_S:
        history.popleft()
    if len(history) >= RATE_LIMIT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit: {RATE_LIMIT_RUNS} runs per "
                f"{RATE_LIMIT_WINDOW_S // 60} minutes."
            ),
        )
    history.append(now)


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    target_model: str = Field(..., description="e.g. 'ollama:qwen2:0.5b'")
    validator_model: str = "local-gemma"
    suite: str = "general"
    num_probes: int = Field(default=8, ge=1, le=MAX_PROBES)
    model_purpose: str = ""
    system_prompt: str | None = None


class SwitchValidatorRequest(BaseModel):
    validator_model: str


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "store": settings.store_backend,
        "cloud_demo": settings.auditor_cloud_demo,
        "vertex_configured": settings.vertex_configured,
        "project": settings.google_cloud_project or None,
    }


@app.get("/api/models/target")
async def target_models() -> dict:
    """Models available to audit. Today: whatever Ollama has pulled."""
    if settings.auditor_cloud_demo:
        return {"models": [], "error": None}

    try:
        models = await list_ollama_models()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not reach ollama: %s", exc)
        return {
            "models": [],
            "error": (
                f"Could not reach Ollama at {settings.ollama_host}. "
                "Is `ollama serve` running?"
            ),
        }
    return {"models": models, "error": None}


@app.get("/api/models/validator")
async def validator_models() -> dict:
    validators = list_validators()
    if settings.auditor_cloud_demo:
        validators = [
            validator
            for validator in validators
            if validator["provider"] == "ai-studio"
        ]
    return {"validators": validators}


@app.get("/api/suites")
async def suites() -> dict:
    return {
        "suites": [
            {"key": key, "label": value["label"], "focus": value["focus"]}
            for key, value in SUITES.items()
        ]
    }


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

class ApiKeyRequest(BaseModel):
    google_api_key: str


def _mask(secret: str) -> str:
    """Enough to recognise a key, never enough to use it."""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}{'•' * 8}{secret[-4:]}"


@app.get("/api/settings")
async def get_settings() -> dict:
    """What is configured. Never returns the key itself."""
    return {
        "google_api_key_set": settings.api_key_configured,
        "google_api_key_hint": (
            _mask(settings.google_api_key) if settings.api_key_configured else None
        ),
        "vertex_configured": settings.vertex_configured,
        "ollama_host": settings.ollama_host,
        "mlx_available": mlx_available(),
        "ollama_available": await ollama_available(),
    }


@app.put("/api/settings/google-api-key")
async def set_google_api_key(body: ApiKeyRequest) -> dict:
    """Apply a Google AI Studio key to the running process.

    The desktop app owns persistence -- it stores the key encrypted and passes
    it in at launch. This endpoint exists so changing the key takes effect
    immediately instead of after a restart.
    """
    key = body.google_api_key.strip()
    settings.google_api_key = key
    # Cached agents captured the old key when they were built.
    reset_agent_cache()
    log.info("google api key %s", "set" if key else "cleared")
    return {
        "google_api_key_set": bool(key),
        "google_api_key_hint": _mask(key) if key else None,
    }


# --------------------------------------------------------------------------
# bringing your own model
# --------------------------------------------------------------------------

class DetectRequest(BaseModel):
    source: str


class PrepareRequest(BaseModel):
    source: str
    #: What to call it in the dropdown. Defaults to the detected suggestion.
    name: str = ""


@app.post("/api/models/detect")
async def detect_model(body: DetectRequest) -> dict:
    """Identify whatever the user pasted, without changing anything."""
    if settings.auditor_cloud_demo and not body.source.strip().startswith(
        ("http://", "https://")
    ):
        raise HTTPException(
            status_code=400,
            detail="This web demo audits model server URLs only.",
        )

    if settings.auditor_cloud_demo:
        tags = set()
    else:
        try:
            tags = {m["name"] for m in await list_ollama_models()}
        except Exception:  # noqa: BLE001 - detection works without Ollama
            tags = set()

    detection = inspect_source(body.source, known_ollama_tags=tags)
    payload = detection.model_dump(mode="json")

    # Tell the UI up front if the action it is about to offer cannot work.
    if detection.kind == "mlx_adapters" and not mlx_available():
        payload["readiness"] = "blocked"
        payload["detail"] = (
            "Auditor found your adapters, but mlx-lm is not installed in its "
            "environment. Install it with: pip install mlx-lm"
        )
        payload["action_label"] = None
    elif detection.kind == "gguf_file" and not await ollama_available():
        payload["readiness"] = "blocked"
        payload["detail"] = (
            "Auditor found the file, but Ollama is not installed. Get it from "
            "ollama.com, then try again."
        )
        payload["action_label"] = None

    return payload


@app.post("/api/models/upload", status_code=201)
async def upload_model(files: list[UploadFile] = File(...)) -> dict:
    """Receive a model folder picked in the browser, then identify it.

    This is the path that works whether Auditor runs on your laptop or on
    Cloud Run, because the files come from the browser rather than from a
    filesystem the server may not share.
    """
    if settings.auditor_cloud_demo:
        raise HTTPException(
            status_code=403,
            detail="Model uploads are disabled on the Google Cloud web demo.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"That folder has too many files (limit {MAX_FILES}). "
                   "Pick the folder containing adapter_config.json.",
        )

    prune_uploads()
    folder = new_upload_dir()
    total = 0
    stored = 0
    origin_folder = ""

    try:
        for upload in files:
            try:
                relative = safe_relative_path(
                    upload.filename or "file"
                )
            except UploadError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if not is_useful(relative):
                continue

            if not origin_folder:
                raw_parts = (upload.filename or "").replace("\\", "/").split("/")
                if len(raw_parts) > 1:
                    origin_folder = safe_component(raw_parts[0])

            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            with destination.open("wb") as sink:
                # Streamed in chunks so a large file cannot balloon memory,
                # and the size cap is enforced mid-transfer rather than after.
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"That is over the "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
                                "Upload only the adapter folder, not the base "
                                "model."
                            ),
                        )
                    sink.write(chunk)
            stored += 1

        if stored == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "None of those files look like a model. Pick the folder "
                    "your training run saved, containing adapter_config.json."
                ),
            )

        detection = inspect_source(str(folder))
        payload = detection.model_dump(mode="json")
        payload["uploaded_bytes"] = total
        payload["uploaded_files"] = stored

        # Name it after the folder the person picked, not the upload id. The
        # picker sends "adapters_v2/adapters.safetensors"; that first component
        # is the only human-meaningful name we get.
        if origin_folder:
            payload["suggested_name"] = origin_folder

        if detection.kind == "mlx_adapters" and not mlx_available():
            payload["readiness"] = "blocked"
            payload["detail"] = (
                "Your adapters uploaded fine, but this server cannot fuse MLX "
                "models -- that needs Apple Silicon. Run Auditor locally to "
                "audit this model, or point it at a deployed endpoint."
            )
            payload["action_label"] = None

        return payload
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


@app.post("/api/models/prepare", status_code=201)
async def prepare_model(body: PrepareRequest) -> dict:
    """Do whatever it takes to make the detected source auditable.

    Fusing a 7B adapter takes a few seconds; loading it into memory on first
    use takes longer. Both happen before this returns, so the client gets a
    model it can immediately audit.
    """
    if settings.auditor_cloud_demo and not body.source.strip().startswith(
        ("http://", "https://")
    ):
        raise HTTPException(
            status_code=403,
            detail="This web demo can only audit model server URLs.",
        )

    if settings.auditor_cloud_demo:
        tags = set()
    else:
        try:
            tags = {m["name"] for m in await list_ollama_models()}
        except Exception:  # noqa: BLE001
            tags = set()

    detection = inspect_source(body.source, known_ollama_tags=tags)
    name = body.name.strip() or detection.suggested_name

    if detection.readiness == "ready" and detection.spec:
        return {"spec": detection.spec, "name": name, "kind": detection.kind}

    if detection.readiness != "needs_prepare":
        raise HTTPException(status_code=400, detail=detection.detail)

    try:
        if detection.kind == "mlx_adapters":
            model = await fuse_adapters(
                adapter_path=detection.resolved_path,
                base_model=detection.base_model,
                name=name,
            )
            return {"spec": model.spec, "name": model.name, "kind": "fused"}

        if detection.kind == "gguf_file":
            tag = await import_gguf(detection.resolved_path, name)
            return {"spec": f"ollama:{tag}", "name": tag, "kind": "ollama"}

        if detection.kind == "model_dir":
            model = register_folder(path=detection.resolved_path, name=name)
            return {"spec": model.spec, "name": model.name, "kind": "folder"}

    except (PrepareError, GgufImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    raise HTTPException(
        status_code=400, detail=f"Cannot prepare {detection.kind}."
    )


@app.get("/api/models/prepared")
async def prepared_models() -> dict:
    if settings.auditor_cloud_demo:
        return {
            "models": [],
            "mlx_available": False,
            "can_export_to_ollama": False,
        }

    return {
        "models": [
            {
                "spec": m.spec,
                "name": m.name,
                "kind": m.kind,
                "base_model": m.base_model,
                "created_at": m.created_at,
            }
            for m in list_prepared()
        ],
        "mlx_available": mlx_available(),
        # Drives whether the UI offers "Keep in Ollama" at all.
        "can_export_to_ollama": find_gguf_converter() is not None,
    }


@app.delete("/api/models/prepared/{name}")
async def delete_prepared(name: str, delete_files: bool = False) -> dict:
    if settings.auditor_cloud_demo:
        raise HTTPException(
            status_code=403,
            detail="Prepared models are disabled on the Google Cloud web demo.",
        )

    if not unregister(name, delete_files=delete_files):
        raise HTTPException(status_code=404, detail=f"No model named {name!r}.")
    return {"removed": name}


@app.post("/api/models/prepared/{name}/export-to-ollama", status_code=201)
async def export_prepared_to_ollama(name: str) -> dict:
    """Convert a prepared model to GGUF and hand it to Ollama.

    Slow and disk-hungry, but the result is a permanent Ollama model that
    needs no managed server. Offered as an option, never required.
    """
    if settings.auditor_cloud_demo:
        raise HTTPException(
            status_code=403,
            detail="Prepared models are disabled on the Google Cloud web demo.",
        )

    if not await ollama_available():
        raise HTTPException(
            status_code=422,
            detail="Ollama is not installed. Get it from ollama.com.",
        )
    try:
        tag = await export_to_ollama(name)
    except (PrepareError, GgufImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"spec": f"ollama:{tag}", "name": tag, "kind": "ollama"}


# --------------------------------------------------------------------------
# uploaded validator context
# --------------------------------------------------------------------------

@app.post("/api/validator/context")
async def upload_validator_context(file: UploadFile = File(...)) -> dict:
    try:
        context = await extract_uploaded_context(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"context": context.model_dump(mode="json")}


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------

@app.post("/api/runs", status_code=201)
async def create_run(body: StartRunRequest, request: Request) -> dict:
    _enforce_rate_limit(request)
    if settings.auditor_cloud_demo:
        if not body.target_model.strip().startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail="This web demo can only audit model server URLs.",
            )
        try:
            validator = get_validator(body.validator_model)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if validator.provider != "ai-studio":
            raise HTTPException(
                status_code=400,
                detail="This web demo uses Gemini API-key judges only.",
            )

    try:
        run = await start_run(RunConfig(**body.model_dump()))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run.run_id, "status": run.status.value}


@app.get("/api/runs")
async def recent_runs(limit: int = 20) -> dict:
    runs = await get_store().list_recent(limit=min(limit, 100))
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "target_model": r.config.target_model,
                "validator_model": r.config.validator_model,
                "suite": r.config.suite,
                "overall_score": r.score.overall,
                "created_at_ms": r.created_at_ms,
            }
            for r in runs
        ]
    }


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = await get_store().get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No such run")
    return run.model_dump(mode="json")


@app.post("/api/runs/{run_id}/validator")
async def switch_validator(run_id: str, body: SwitchValidatorRequest) -> dict:
    """Repoint an in-flight run at a different validator."""
    if settings.auditor_cloud_demo:
        try:
            validator = get_validator(body.validator_model)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if validator.provider != "ai-studio":
            raise HTTPException(
                status_code=400,
                detail="This web demo uses Gemini API-key judges only.",
            )

    controller = ACTIVE.get(run_id)
    if not controller:
        raise HTTPException(
            status_code=409, detail="Run is not active; nothing to switch."
        )
    try:
        previous = controller.switch_validator(body.validator_model)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await bus.publish(
        run_id,
        EventType.VALIDATOR_SWITCHED,
        {"from": previous, "to": body.validator_model},
    )
    return {"from": previous, "to": body.validator_model}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    controller = ACTIVE.get(run_id)
    if not controller:
        raise HTTPException(status_code=409, detail="Run is not active.")
    controller.cancel()
    return {"run_id": run_id, "status": "cancelling"}


# --------------------------------------------------------------------------
# live stream
# --------------------------------------------------------------------------

def _sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request, since: int = 0):
    """Server-sent events for one run.

    `since` is the last seq the client saw, so a reconnecting browser resumes
    without replaying the whole run.
    """
    run = await get_store().get(run_id)
    if not run and run_id not in ACTIVE:
        raise HTTPException(status_code=404, detail="No such run")

    queue, backlog, already_closed = await bus.subscribe(run_id, since_seq=since)

    async def generator():
        try:
            for event in backlog:
                yield _sse(event.type.value, event.model_dump(mode="json"))

            if already_closed:
                return

            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keep-alive: proxies drop idle connections, and a probe can
                    # legitimately take longer than that to come back.
                    yield ": keep-alive\n\n"
                    continue

                if item is STREAM_END:
                    break
                yield _sse(item.type.value, item.model_dump(mode="json"))
        finally:
            await bus.unsubscribe(run_id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    for controller in list(ACTIVE.values()):
        controller.cancel()
    # Managed model servers are our child processes; leaving them running
    # would hold gigabytes of memory after the API is gone.
    await stop_all_servers()
