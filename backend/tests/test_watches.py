"""Tests for scheduling audits nobody is watching.

The scheduling arithmetic is the risky part. A watch that fires too often
burns credits silently; one that never fires looks like it is working right
up until you need it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.watches import (  # noqa: E402
    INTERVALS,
    Watch,
    WatchStore,
    reset_watch_store,
)

_HOUR = 3_600_000
_DAY = 86_400_000

# A round midnight UTC, so the arithmetic is readable.
MIDNIGHT = 1_787_702_400_000


def _watch(**kw) -> Watch:
    kw.setdefault("target_model", "ollama:x")
    return Watch(**kw)


class TestScheduling:
    def test_daily_lands_on_the_requested_hour(self):
        w = _watch(cadence="daily", hour_utc=3)
        due = w.schedule_next(now_ms=MIDNIGHT + 1 * _HOUR)
        assert due == MIDNIGHT + 3 * _HOUR

    def test_daily_rolls_to_tomorrow_once_the_hour_has_passed(self):
        w = _watch(cadence="daily", hour_utc=3)
        due = w.schedule_next(now_ms=MIDNIGHT + 5 * _HOUR)
        assert due == MIDNIGHT + _DAY + 3 * _HOUR

    def test_hourly_is_one_hour_out(self):
        w = _watch(cadence="hourly")
        now = MIDNIGHT + 90 * 60 * 1000
        assert w.schedule_next(now_ms=now) == now + _HOUR

    def test_weekly_steps_a_whole_week(self):
        w = _watch(cadence="weekly", hour_utc=3)
        first = w.schedule_next(now_ms=MIDNIGHT + 5 * _HOUR)
        second = w.schedule_next(now_ms=first + 1000)
        assert second - first == INTERVALS["weekly"]

    def test_a_long_disabled_watch_does_not_fire_a_backlog(self):
        # Away for a week: it should run once tonight, not seven times.
        w = _watch(cadence="daily", hour_utc=3)
        w.schedule_next(now_ms=MIDNIGHT)
        much_later = MIDNIGHT + 7 * _DAY + 5 * _HOUR

        due = w.schedule_next(now_ms=much_later)
        assert due > much_later
        assert due - much_later < _DAY

    def test_next_due_is_always_in_the_future(self):
        for cadence in INTERVALS:
            for hour in (0, 3, 12, 23):
                w = _watch(cadence=cadence, hour_utc=hour)
                now = MIDNIGHT + 11 * _HOUR
                assert w.schedule_next(now_ms=now) > now


class TestIsDue:
    def test_due_once_the_time_arrives(self):
        w = _watch()
        w.next_due_ms = MIDNIGHT
        assert w.is_due(MIDNIGHT) is True
        assert w.is_due(MIDNIGHT - 1) is False

    def test_a_disabled_watch_is_never_due(self):
        w = _watch(enabled=False)
        w.next_due_ms = MIDNIGHT
        assert w.is_due(MIDNIGHT + _DAY) is False


@pytest.mark.asyncio
class TestWatchStore:
    @pytest.fixture(autouse=True)
    def _clean(self):
        reset_watch_store()
        yield
        reset_watch_store()

    async def test_round_trip(self):
        store = WatchStore()
        w = _watch(name="nightly")
        await store.save(w)

        got = await store.get(w.watch_id)
        assert got is not None and got.name == "nightly"

    async def test_stored_watch_is_a_snapshot(self):
        store = WatchStore()
        w = _watch(name="before")
        await store.save(w)

        w.name = "after"
        again = await store.get(w.watch_id)
        assert again.name == "before"

    async def test_due_returns_only_what_should_run(self):
        store = WatchStore()
        ready = _watch(name="ready")
        ready.next_due_ms = MIDNIGHT
        later = _watch(name="later")
        later.next_due_ms = MIDNIGHT + _DAY
        off = _watch(name="off", enabled=False)
        off.next_due_ms = 0
        for w in (ready, later, off):
            await store.save(w)

        names = {w.name for w in await store.due(MIDNIGHT + _HOUR)}
        assert names == {"ready"}

    async def test_delete(self):
        store = WatchStore()
        w = _watch()
        await store.save(w)
        assert await store.delete(w.watch_id) is True
        assert await store.delete(w.watch_id) is False

    async def test_newest_first(self):
        store = WatchStore()
        old = _watch(name="old", created_at_ms=1000)
        new = _watch(name="new", created_at_ms=2000)
        await store.save(old)
        await store.save(new)
        assert [w.name for w in await store.list_all()] == ["new", "old"]


class TestDatasetPathSafety:
    """The download endpoint takes a filename straight from the URL."""

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "/etc/passwd", "..%2f..%2fetc", "....//etc/passwd"],
    )
    def test_traversal_is_refused(self, name):
        from app.datasets import dataset_path

        # Either rejected outright, or resolved to something harmless inside
        # the dataset directory -- never a file elsewhere on disk.
        resolved = dataset_path(name)
        assert resolved is None or "etc/passwd" not in str(resolved)
