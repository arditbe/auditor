"""Standing instructions: audit this model on a schedule, unattended.

A *watch* is the thing that makes Auditor autonomous rather than interactive.
You describe what to audit and how often; after that nobody has to open the
dashboard for it to keep happening.

Schedules are stored here rather than as Cloud Scheduler jobs. One scheduler
job pings `/api/scheduled/tick`, and this module decides which watches are due.
That keeps the app in charge of its own schedule -- the UI can create and edit
watches without any Google Cloud permissions -- and it runs identically on a
laptop with a cron entry.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from .config import settings

log = logging.getLogger(__name__)

Cadence = Literal["hourly", "daily", "weekly"]

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000

INTERVALS: dict[str, int] = {
    "hourly": _HOUR_MS,
    "daily": _DAY_MS,
    "weekly": 7 * _DAY_MS,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


class Watch(BaseModel):
    watch_id: str = Field(default_factory=lambda: f"watch_{uuid.uuid4().hex[:10]}")
    name: str = ""
    enabled: bool = True

    # --- what to audit ---
    target_model: str
    validator_model: str = "gemini-flash"
    suite: str = "general"
    num_probes: int = 6
    model_purpose: str = ""

    # --- when ---
    cadence: Cadence = "daily"
    #: Hour of day (UTC) for daily/weekly. Ignored for hourly.
    hour_utc: int = 3

    # --- what to do with the result ---
    #: Build a training dataset from the failures each time it runs.
    build_dataset: bool = False
    #: Only interesting if the score drops; a stable model needs no dataset.
    dataset_on_regression_only: bool = True

    # --- state, written by the scheduler ---
    next_due_ms: int = 0
    last_run_at_ms: int | None = None
    last_run_id: str | None = None
    last_score: float | None = None
    last_summary: str = ""
    consecutive_failures: int = 0
    created_at_ms: int = Field(default_factory=_now_ms)

    def is_due(self, now_ms: int | None = None) -> bool:
        now = now_ms if now_ms is not None else _now_ms()
        return self.enabled and now >= self.next_due_ms

    def schedule_next(self, now_ms: int | None = None) -> int:
        """Set the next due time, and return it.

        Daily and weekly land on the requested hour. Anything already in the
        past is pushed forward a whole interval, so a watch that was disabled
        for a week does not fire seven times when it comes back.
        """
        now = now_ms if now_ms is not None else _now_ms()
        interval = INTERVALS[self.cadence]

        if self.cadence == "hourly":
            self.next_due_ms = now + interval
            return self.next_due_ms

        # Midnight UTC of the current day, plus the requested hour.
        midnight = (now // _DAY_MS) * _DAY_MS
        candidate = midnight + self.hour_utc * _HOUR_MS
        while candidate <= now:
            candidate += interval
        self.next_due_ms = candidate
        return self.next_due_ms


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

class WatchStore:
    """Firestore-backed when configured, in-memory otherwise.

    Kept separate from the run store: runs are an append-only log, watches are
    mutable configuration, and mixing them makes both harder to reason about.
    """

    def __init__(self) -> None:
        self._memory: dict[str, Watch] = {}
        self._lock = asyncio.Lock()
        self._client = None

        if settings.store_backend == "firestore":
            try:
                from google.cloud import firestore

                kwargs = {}
                if settings.google_cloud_project:
                    kwargs["project"] = settings.google_cloud_project
                self._client = firestore.Client(**kwargs)
                log.info("watch store: firestore")
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                log.warning("watch store falling back to memory: %s", exc)

    @property
    def _collection(self) -> str:
        return f"{settings.firestore_collection}_watches"

    async def save(self, watch: Watch) -> None:
        if self._client is None:
            async with self._lock:
                self._memory[watch.watch_id] = watch.model_copy(deep=True)
            return
        doc = self._client.collection(self._collection).document(watch.watch_id)
        await asyncio.to_thread(doc.set, watch.model_dump(mode="json"))

    async def get(self, watch_id: str) -> Watch | None:
        if self._client is None:
            async with self._lock:
                w = self._memory.get(watch_id)
                return w.model_copy(deep=True) if w else None
        doc = self._client.collection(self._collection).document(watch_id)
        snap = await asyncio.to_thread(doc.get)
        return Watch.model_validate(snap.to_dict()) if snap.exists else None

    async def list_all(self) -> list[Watch]:
        if self._client is None:
            async with self._lock:
                items = [w.model_copy(deep=True) for w in self._memory.values()]
        else:
            docs = await asyncio.to_thread(
                lambda: list(self._client.collection(self._collection).stream())
            )
            items = []
            for d in docs:
                try:
                    items.append(Watch.model_validate(d.to_dict()))
                except Exception as exc:  # noqa: BLE001
                    log.warning("skipping unreadable watch %s: %s", d.id, exc)
        return sorted(items, key=lambda w: w.created_at_ms, reverse=True)

    async def delete(self, watch_id: str) -> bool:
        if self._client is None:
            async with self._lock:
                return self._memory.pop(watch_id, None) is not None
        doc = self._client.collection(self._collection).document(watch_id)
        snap = await asyncio.to_thread(doc.get)
        if not snap.exists:
            return False
        await asyncio.to_thread(doc.delete)
        return True

    async def due(self, now_ms: int | None = None) -> list[Watch]:
        return [w for w in await self.list_all() if w.is_due(now_ms)]


_store: WatchStore | None = None


def get_watch_store() -> WatchStore:
    global _store
    if _store is None:
        _store = WatchStore()
    return _store


def reset_watch_store() -> None:
    global _store
    _store = None
