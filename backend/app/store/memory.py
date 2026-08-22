"""In-memory run store. Bounded so a long-lived demo instance cannot grow
without limit."""
from __future__ import annotations

import asyncio
from collections import OrderedDict

from ..models.schemas import ValidationRun
from .base import RunStore

MAX_RUNS = 200


class MemoryRunStore(RunStore):
    def __init__(self, max_runs: int = MAX_RUNS) -> None:
        self._runs: OrderedDict[str, ValidationRun] = OrderedDict()
        self._max = max_runs
        self._lock = asyncio.Lock()

    async def save(self, run: ValidationRun) -> None:
        async with self._lock:
            # Store a snapshot: the orchestrator keeps mutating its own copy,
            # and a store that aliases live objects is not a store.
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._runs.move_to_end(run.run_id)
            while len(self._runs) > self._max:
                self._runs.popitem(last=False)

    async def get(self, run_id: str) -> ValidationRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run else None

    async def list_recent(self, limit: int = 20) -> list[ValidationRun]:
        async with self._lock:
            runs = sorted(
                self._runs.values(), key=lambda r: r.created_at_ms, reverse=True
            )
            return [r.model_copy(deep=True) for r in runs[:limit]]

    async def delete(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)
