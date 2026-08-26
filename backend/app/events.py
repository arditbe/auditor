"""In-process pub/sub for live run events.

The dashboard is the product here, so events are the primary output of a run,
not a side effect of it. Each run keeps a replay log: a browser that connects
late (or reconnects mid-run) is caught up to the current state before it starts
receiving new events, which is what makes refresh-during-demo safe.

This is deliberately in-process. Cloud Run must therefore run a run and its SSE
stream on the same instance -- see docs/DEPLOY.md for the session-affinity
setting that guarantees that, and the Firestore-listener alternative if you
outgrow it.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    PLAN_READY = "plan.ready"
    FOCUS_STARTED = "focus.started"
    PROBE_STARTED = "probe.started"
    PROBE_ANSWERED = "probe.answered"
    PROBE_EVALUATED = "probe.evaluated"
    SCORE_UPDATED = "score.updated"
    VALIDATOR_SWITCHED = "validator.switched"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    LOG = "log"


class RunEvent(BaseModel):
    seq: int = 0
    type: EventType
    run_id: str
    at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    data: dict[str, Any] = Field(default_factory=dict)


#: Sentinel pushed to subscriber queues when a run's stream is finished.
STREAM_END = object()


class _RunChannel:
    def __init__(self) -> None:
        self.log: list[RunEvent] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.closed = False
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq


class EventBus:
    def __init__(self) -> None:
        self._channels: dict[str, _RunChannel] = {}
        self._lock = asyncio.Lock()

    def _channel(self, run_id: str) -> _RunChannel:
        if run_id not in self._channels:
            self._channels[run_id] = _RunChannel()
        return self._channels[run_id]

    async def publish(
        self, run_id: str, type_: EventType, data: dict[str, Any] | None = None
    ) -> RunEvent:
        async with self._lock:
            channel = self._channel(run_id)
            event = RunEvent(
                seq=channel.next_seq(), type=type_, run_id=run_id, data=data or {}
            )
            channel.log.append(event)
            subscribers = list(channel.subscribers)

        for queue in subscribers:
            # Never let one wedged browser stall the run. Queues are unbounded,
            # so put_nowait only fails if a subscriber was torn down mid-publish.
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                pass
        return event

    async def subscribe(
        self, run_id: str, since_seq: int = 0
    ) -> tuple[asyncio.Queue, list[RunEvent], bool]:
        """Register a subscriber.

        Returns its queue, the backlog it missed, and whether the run's stream
        has already closed (in which case the backlog is the whole story).
        """
        async with self._lock:
            channel = self._channel(run_id)
            backlog = [e for e in channel.log if e.seq > since_seq]
            queue: asyncio.Queue = asyncio.Queue()
            channel.subscribers.add(queue)
            return queue, backlog, channel.closed

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            channel = self._channels.get(run_id)
            if channel:
                channel.subscribers.discard(queue)

    async def close(self, run_id: str) -> None:
        """Mark a run's stream finished and release every listener."""
        async with self._lock:
            channel = self._channel(run_id)
            channel.closed = True
            subscribers = list(channel.subscribers)

        for queue in subscribers:
            queue.put_nowait(STREAM_END)

    async def discard(self, run_id: str) -> None:
        """Drop a run's replay log. Called when the run is evicted from memory."""
        async with self._lock:
            self._channels.pop(run_id, None)


bus = EventBus()
