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
from .providers import list_ollama_models, list_validators
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
        "vertex_configured": settings.vertex_configured,
        "project": settings.google_cloud_project or None,
    }


@app.get("/api/models/target")
async def target_models() -> dict:
    """Models available to audit. Today: whatever Ollama has pulled."""
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
    return {"validators": list_validators()}


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
    try:
        await list_ollama_models()
        ollama_is_available = True
    except Exception:  # noqa: BLE001
        ollama_is_available = False
    return {
        "google_api_key_set": settings.api_key_configured,
        "google_api_key_hint": (
            _mask(settings.google_api_key) if settings.api_key_configured else None
        ),
        "vertex_configured": settings.vertex_configured,
        "ollama_host": settings.ollama_host,
        "ollama_available": ollama_is_available,
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
