"""Tests for the decisions the agent makes without being asked.

Two behaviours matter here and both are easy to get subtly wrong:
  * choosing which weakness to investigate, and knowing when to stop
  * judging whether tonight's score is worse than last night's

The second one runs with nobody watching, so a wrong answer is either a false
alarm at 3am or a regression nobody hears about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.models.schemas import (  # noqa: E402
    Dimension,
    Evaluation,
    Probe,
    RunConfig,
    RunStatus,
    ScoreSnapshot,
    ValidationRun,
    Verdict,
)
from app.orchestrator import (  # noqa: E402
    aggregate,
    detect_regression,
    pick_weak_dimension,
)
from app.store import reset_store  # noqa: E402
from app.store.memory import MemoryRunStore  # noqa: E402


def _run(target: str = "ollama:x", overall: float = 50.0, **kw) -> ValidationRun:
    run = ValidationRun(
        config=RunConfig(target_model=target, validator_model="gemini-flash")
    )
    run.score = ScoreSnapshot(overall=overall, completed=kw.pop("completed", 3), total=3)
    for k, v in kw.items():
        setattr(run, k, v)
    return run


class TestPickWeakDimension:
    """Where the agent decides to dig."""

    def _with(self, dims: dict[Dimension, float]) -> ValidationRun:
        run = _run()
        run.score.dimensions = dims
        return run

    def test_picks_the_weakest_dimension(self):
        run = self._with({
            Dimension.ACCURACY: 2.5,
            Dimension.HALLUCINATION_RESISTANCE: 0.5,
            Dimension.SAFETY: 4.0,
        })
        picked, mean = pick_weak_dimension(run, set())
        assert picked is Dimension.HALLUCINATION_RESISTANCE
        assert mean == 0.5

    def test_a_strong_model_is_left_alone(self):
        run = self._with({Dimension.ACCURACY: 4.5, Dimension.SAFETY: 5.0})
        assert pick_weak_dimension(run, set()) is None

    def test_exactly_at_the_threshold_is_not_weak(self):
        run = self._with({Dimension.ACCURACY: settings.adaptive_threshold})
        assert pick_weak_dimension(run, set()) is None

    def test_does_not_chase_the_same_weakness_twice(self):
        # Without this the agent would drill the worst dimension forever.
        run = self._with({
            Dimension.ACCURACY: 2.0,
            Dimension.COHERENCE: 1.0,
        })
        first, _ = pick_weak_dimension(run, set())
        assert first is Dimension.COHERENCE

        second, _ = pick_weak_dimension(run, {Dimension.COHERENCE})
        assert second is Dimension.ACCURACY

        assert pick_weak_dimension(
            run, {Dimension.COHERENCE, Dimension.ACCURACY}
        ) is None

    def test_can_be_switched_off(self, monkeypatch):
        monkeypatch.setattr(settings, "adaptive_probing", False)
        run = self._with({Dimension.ACCURACY: 0.0})
        assert pick_weak_dimension(run, set()) is None

    def test_no_scores_yet(self):
        assert pick_weak_dimension(self._with({}), set()) is None


@pytest.mark.asyncio
class TestDetectRegression:
    """The call a scheduled run has to make on its own."""

    @pytest.fixture(autouse=True)
    def _store(self, monkeypatch):
        store = MemoryRunStore()
        import app.orchestrator as orch

        monkeypatch.setattr(orch, "get_store", lambda: store)
        yield store
        reset_store()

    async def test_first_ever_audit_has_no_baseline(self, _store):
        result = await detect_regression(_run(overall=40.0))
        assert result["regressed"] is False
        assert result["baseline"] is None
        assert "First audit" in result["summary"]

    async def test_a_real_drop_is_flagged(self, _store):
        old = _run(overall=80.0, status=RunStatus.COMPLETE)
        await _store.save(old)

        result = await detect_regression(_run(overall=50.0))
        assert result["regressed"] is True
        assert result["delta"] == -30.0
        assert result["baseline"] == 80.0
        assert "Regression" in result["summary"]

    async def test_small_wobble_is_not_a_regression(self, _store):
        # Judges are language models; a couple of points of noise between runs
        # is expected and must not page anyone.
        await _store.save(_run(overall=70.0, status=RunStatus.COMPLETE))

        result = await detect_regression(_run(overall=67.0))
        assert result["regressed"] is False
        assert "Stable" in result["summary"]

    async def test_improvement_is_reported_but_not_a_regression(self, _store):
        await _store.save(_run(overall=40.0, status=RunStatus.COMPLETE))

        result = await detect_regression(_run(overall=75.0))
        assert result["regressed"] is False
        assert "Improved" in result["summary"]

    async def test_compares_against_the_same_model_only(self, _store):
        # A different model scoring badly must not look like this one regressing.
        await _store.save(
            _run(target="ollama:other", overall=95.0, status=RunStatus.COMPLETE)
        )

        result = await detect_regression(_run(target="ollama:x", overall=50.0))
        assert result["baseline"] is None

    async def test_ignores_failed_and_unscored_runs(self, _store):
        await _store.save(_run(overall=90.0, status=RunStatus.FAILED))
        await _store.save(_run(overall=90.0, status=RunStatus.COMPLETE, completed=0))

        result = await detect_regression(_run(overall=50.0))
        assert result["baseline"] is None, "only completed, scored runs count"

    async def test_a_broken_store_does_not_fail_the_run(self, monkeypatch):
        class Broken:
            async def list_recent(self, limit=50):
                raise RuntimeError("firestore down")

        import app.orchestrator as orch

        monkeypatch.setattr(orch, "get_store", lambda: Broken())
        # Losing history is not a reason to lose the audit.
        assert await detect_regression(_run()) is None


class TestFocusRoundsInTheReport:
    def test_report_records_what_the_agent_chose_to_investigate(self):
        from app.orchestrator import build_report

        run = _run()
        run.probes = [
            Probe(index=0, question="q0", dimension=Dimension.ACCURACY, round=0),
            Probe(index=1, question="q1", dimension=Dimension.SAFETY, round=1),
        ]
        for p in run.probes:
            run.evaluations[p.probe_id] = Evaluation(
                probe_id=p.probe_id,
                scores={p.dimension: 2.0},
                verdict=Verdict.FAIL,
            )
        run.score = aggregate(run)

        report = build_report(run)
        assert report["focus_rounds"] == [1]
        assert report["focused_on"] == ["safety"]
