"""Tests for the two guarantees the live dashboard rests on:

  * a browser that connects late or reloads still sees the whole run
  * the store holds snapshots, not aliases of the run the loop is mutating
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.events import STREAM_END, EventBus, EventType  # noqa: E402
from app.models.schemas import (  # noqa: E402
    Dimension,
    Evaluation,
    RunConfig,
    RunStatus,
    ValidationRun,
)
from app.store.memory import MemoryRunStore  # noqa: E402

pytestmark = pytest.mark.asyncio


class TestEventBus:
    async def test_late_subscriber_gets_the_whole_backlog(self):
        bus = EventBus()
        await bus.publish("r1", EventType.RUN_STARTED, {"n": 1})
        await bus.publish("r1", EventType.SCORE_UPDATED, {"n": 2})

        _, backlog, closed = await bus.subscribe("r1")
        assert [e.data["n"] for e in backlog] == [1, 2]
        assert closed is False

    async def test_since_resumes_without_replaying_everything(self):
        bus = EventBus()
        for i in range(1, 4):
            await bus.publish("r1", EventType.LOG, {"n": i})

        _, backlog, _ = await bus.subscribe("r1", since_seq=2)
        assert [e.data["n"] for e in backlog] == [3]

    async def test_sequence_numbers_are_monotonic_per_run(self):
        bus = EventBus()
        seqs = [
            (await bus.publish("r1", EventType.LOG, {})).seq for _ in range(5)
        ]
        assert seqs == [1, 2, 3, 4, 5]

    async def test_runs_do_not_share_a_sequence(self):
        bus = EventBus()
        a = await bus.publish("r1", EventType.LOG, {})
        b = await bus.publish("r2", EventType.LOG, {})
        assert a.seq == b.seq == 1

    async def test_live_subscriber_receives_new_events(self):
        bus = EventBus()
        queue, _, _ = await bus.subscribe("r1")
        await bus.publish("r1", EventType.SCORE_UPDATED, {"overall": 42})
        event = await queue.get()
        assert event.data["overall"] == 42

    async def test_close_releases_listeners_and_marks_the_run_finished(self):
        bus = EventBus()
        queue, _, _ = await bus.subscribe("r1")
        await bus.close("r1")
        assert await queue.get() is STREAM_END

        # A subscriber arriving after close still gets the backlog, and is told
        # not to wait for more.
        _, backlog, closed = await bus.subscribe("r1")
        assert closed is True
        assert backlog == []

    async def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        queue, _, _ = await bus.subscribe("r1")
        await bus.unsubscribe("r1", queue)
        await bus.publish("r1", EventType.LOG, {})
        assert queue.empty()


def _run() -> ValidationRun:
    return ValidationRun(
        config=RunConfig(target_model="ollama:x", validator_model="local-gemma")
    )


class TestMemoryStore:
    async def test_save_and_get_round_trip(self):
        store = MemoryRunStore()
        run = _run()
        await store.save(run)
        assert (await store.get(run.run_id)).run_id == run.run_id

    async def test_stored_run_is_a_snapshot_not_an_alias(self):
        store = MemoryRunStore()
        run = _run()
        await store.save(run)

        # The orchestrator keeps mutating its own object after saving.
        run.status = RunStatus.COMPLETE
        run.evaluations["p1"] = Evaluation(
            probe_id="p1", scores={Dimension.ACCURACY: 5.0}
        )

        stored = await store.get(run.run_id)
        assert stored.status is RunStatus.PENDING
        assert stored.evaluations == {}

    async def test_mutating_a_fetched_run_does_not_corrupt_the_store(self):
        store = MemoryRunStore()
        run = _run()
        await store.save(run)

        fetched = await store.get(run.run_id)
        fetched.status = RunStatus.FAILED

        assert (await store.get(run.run_id)).status is RunStatus.PENDING

    async def test_missing_run_is_none(self):
        assert await MemoryRunStore().get("nope") is None

    async def test_list_recent_is_newest_first(self):
        store = MemoryRunStore()
        first, second = _run(), _run()
        second.created_at_ms = first.created_at_ms + 1000
        await store.save(first)
        await store.save(second)

        assert [r.run_id for r in await store.list_recent()] == [
            second.run_id,
            first.run_id,
        ]

    async def test_oldest_runs_are_evicted_past_the_cap(self):
        store = MemoryRunStore(max_runs=2)
        runs = [_run() for _ in range(3)]
        for run in runs:
            await store.save(run)

        assert await store.get(runs[0].run_id) is None
        assert await store.get(runs[2].run_id) is not None

    async def test_delete(self):
        store = MemoryRunStore()
        run = _run()
        await store.save(run)
        await store.delete(run.run_id)
        assert await store.get(run.run_id) is None
