"""Tests for the parts that decide a model's score.

These are the pieces where a silent bug produces a plausible-looking but wrong
number, which is the worst failure mode a validation tool can have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.parsing import (  # noqa: E402
    extract_json,
    extract_object,
    salvage_objects,
)
from app.models.schemas import (  # noqa: E402
    Dimension,
    Evaluation,
    Probe,
    RunConfig,
    ValidationRun,
    Verdict,
    weighted_score,
)
from app.orchestrator import (  # noqa: E402
    _coerce_evaluation,
    _coerce_probes,
    aggregate,
    build_report,
)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

class TestParsing:
    def test_plain_object(self):
        assert extract_object('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_and_trailer(self):
        text = 'Sure! Here is the score:\n{"a": 1}\nLet me know if you need more.'
        assert extract_object(text) == {"a": 1}

    def test_trailing_comma(self):
        assert extract_object('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        text = '{"reasoning": "it emitted a } and then a {", "score": 3}'
        assert extract_object(text) == {
            "reasoning": "it emitted a } and then a {",
            "score": 3,
        }

    def test_escaped_quote_inside_string(self):
        assert extract_object(r'{"r": "said \"no\" clearly"}') == {
            "r": 'said "no" clearly'
        }

    def test_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    @pytest.mark.parametrize("text", ["", "no json at all", "{unclosed"])
    def test_gives_up_rather_than_inventing(self, text):
        assert extract_object(text) is None

    def test_object_helper_rejects_arrays(self):
        assert extract_object("[1,2]") is None


class TestSalvage:
    def test_recovers_intact_elements_from_a_broken_array(self):
        # The real failure that motivated this: gemma3 dropped the `", "`
        # before `is_trap`, corrupting one element and the enclosing array.
        text = """```json
{"probes": [
  {"question": "q1", "dimension": "accuracy", "is_trap": false},
  {"question": "q2", "criteria": "says None is fine. is_trap": false},
  {"question": "q3", "dimension": "safety", "is_trap": true}
]}
```"""
        assert extract_object(text) is None, "strict parse should still refuse"

        salvaged = salvage_objects(text)
        questions = [o.get("question") for o in salvaged if "question" in o]
        assert "q1" in questions and "q3" in questions

    def test_returns_empty_when_nothing_parses(self):
        assert salvage_objects("total garbage {{{") == []
        assert salvage_objects("") == []

    def test_ignores_braces_inside_strings(self):
        text = '[{"a": "a } brace"}, {"b": 2}]'
        assert salvage_objects(text) == [{"a": "a } brace"}, {"b": 2}]

    def test_finds_nested_objects_only_at_top_level(self):
        text = '{"outer": {"inner": 1}}'
        assert salvage_objects(text) == [{"outer": {"inner": 1}}]


# --------------------------------------------------------------------------
# weighted score
# --------------------------------------------------------------------------

class TestWeightedScore:
    def test_all_perfect_is_100(self):
        scores = {d: 5.0 for d in Dimension}
        assert weighted_score(scores) == 100.0

    def test_all_zero_is_zero(self):
        assert weighted_score({d: 0.0 for d in Dimension}) == 0.0

    def test_empty_is_zero_not_a_crash(self):
        assert weighted_score({}) == 0.0

    def test_renormalises_over_tested_dimensions_only(self):
        # Only accuracy tested, scored 5/5 -> 100, not 30 (its raw weight).
        assert weighted_score({Dimension.ACCURACY: 5.0}) == 100.0

    def test_weighting_favours_the_heavier_dimension(self):
        # accuracy (0.30) perfect + coherence (0.10) zero should beat the
        # reverse, because accuracy carries more weight.
        high_accuracy = weighted_score(
            {Dimension.ACCURACY: 5.0, Dimension.COHERENCE: 0.0}
        )
        high_coherence = weighted_score(
            {Dimension.ACCURACY: 0.0, Dimension.COHERENCE: 5.0}
        )
        assert high_accuracy > high_coherence


# --------------------------------------------------------------------------
# coercion of model output
# --------------------------------------------------------------------------

class TestCoerceEvaluation:
    def _probe(self) -> Probe:
        return Probe(index=0, question="q", dimension=Dimension.ACCURACY)

    def test_happy_path(self):
        raw = {
            "scores": {"accuracy": 4},
            "verdict": "pass",
            "reasoning": "fine",
            "flags": ["refused-correctly"],
        }
        ev = _coerce_evaluation(raw, self._probe(), "local-gemma", 10)
        assert ev.scores == {Dimension.ACCURACY: 4.0}
        assert ev.verdict is Verdict.PASS
        assert ev.flags == ["refused-correctly"]
        assert ev.judged_by == "local-gemma"

    def test_out_of_range_scores_are_clamped(self):
        raw = {"scores": {"accuracy": 9, "safety": -3}, "verdict": "pass"}
        ev = _coerce_evaluation(raw, self._probe(), "v", 0)
        assert ev.scores[Dimension.ACCURACY] == 5.0
        assert ev.scores[Dimension.SAFETY] == 0.0

    def test_unknown_dimension_is_dropped(self):
        raw = {"scores": {"accuracy": 3, "vibes": 5}, "verdict": "warn"}
        ev = _coerce_evaluation(raw, self._probe(), "v", 0)
        assert set(ev.scores) == {Dimension.ACCURACY}

    def test_bad_verdict_is_derived_from_the_numbers(self):
        raw = {"scores": {"accuracy": 1}, "verdict": "catastrophic"}
        ev = _coerce_evaluation(raw, self._probe(), "v", 0)
        assert ev.verdict is Verdict.FAIL

    def test_no_usable_scores_returns_none(self):
        assert _coerce_evaluation({"scores": {}}, self._probe(), "v", 0) is None
        assert _coerce_evaluation(None, self._probe(), "v", 0) is None

    def test_non_numeric_score_is_skipped(self):
        raw = {"scores": {"accuracy": "good", "safety": 4}, "verdict": "pass"}
        ev = _coerce_evaluation(raw, self._probe(), "v", 0)
        assert set(ev.scores) == {Dimension.SAFETY}


class TestCoerceProbes:
    def test_reads_the_probes_key(self):
        raw = {"probes": [{"question": "q1", "dimension": "safety"}]}
        probes = _coerce_probes(raw, 5)
        assert len(probes) == 1
        assert probes[0].dimension is Dimension.SAFETY

    def test_accepts_a_bare_list(self):
        assert len(_coerce_probes([{"question": "q"}], 5)) == 1

    def test_respects_the_limit(self):
        raw = {"probes": [{"question": f"q{i}"} for i in range(20)]}
        assert len(_coerce_probes(raw, 3)) == 3

    def test_blank_questions_are_dropped_and_indices_stay_contiguous(self):
        raw = {"probes": [{"question": "a"}, {"question": "   "}, {"question": "b"}]}
        probes = _coerce_probes(raw, 10)
        assert [p.question for p in probes] == ["a", "b"]
        assert [p.index for p in probes] == [0, 1]

    def test_unknown_dimension_falls_back_to_accuracy(self):
        probes = _coerce_probes({"probes": [{"question": "q", "dimension": "x"}]}, 5)
        assert probes[0].dimension is Dimension.ACCURACY

    def test_garbage_yields_nothing(self):
        assert _coerce_probes("not json", 5) == []
        assert _coerce_probes({"nope": 1}, 5) == []


# --------------------------------------------------------------------------
# aggregation and reporting
# --------------------------------------------------------------------------

def _run_with(evaluations: list[Evaluation]) -> ValidationRun:
    run = ValidationRun(
        config=RunConfig(target_model="ollama:x", validator_model="local-gemma")
    )
    for i, ev in enumerate(evaluations):
        run.probes.append(
            Probe(
                probe_id=ev.probe_id,
                index=i,
                question=f"q{i}",
                dimension=next(iter(ev.scores)),
            )
        )
        run.evaluations[ev.probe_id] = ev
    return run


class TestAggregate:
    def test_averages_each_dimension_independently(self):
        run = _run_with(
            [
                Evaluation(
                    probe_id="p1",
                    scores={Dimension.ACCURACY: 4.0},
                    verdict=Verdict.PASS,
                ),
                Evaluation(
                    probe_id="p2",
                    scores={Dimension.ACCURACY: 2.0},
                    verdict=Verdict.FAIL,
                ),
            ]
        )
        snapshot = aggregate(run)
        assert snapshot.dimensions[Dimension.ACCURACY] == 3.0
        assert snapshot.completed == 2
        assert snapshot.passes == 1
        assert snapshot.fails == 1

    def test_untested_dimensions_are_absent_not_zero(self):
        run = _run_with(
            [Evaluation(probe_id="p1", scores={Dimension.SAFETY: 5.0})]
        )
        snapshot = aggregate(run)
        assert Dimension.ACCURACY not in snapshot.dimensions
        # A single perfect safety score must read 100, not be dragged down by
        # dimensions nothing has measured.
        assert snapshot.overall == 100.0

    def test_no_evaluations_is_zero_not_a_crash(self):
        snapshot = aggregate(_run_with([]))
        assert snapshot.overall == 0.0
        assert snapshot.completed == 0


class TestReport:
    def test_weakest_dimension_is_the_lowest_mean(self):
        run = _run_with(
            [
                Evaluation(
                    probe_id="p1",
                    scores={Dimension.ACCURACY: 5.0},
                    verdict=Verdict.PASS,
                ),
                Evaluation(
                    probe_id="p2",
                    scores={Dimension.SAFETY: 1.0},
                    verdict=Verdict.FAIL,
                    reasoning="unsafe",
                    judged_by="local-gemma",
                ),
            ]
        )
        run.score = aggregate(run)
        report = build_report(run)
        assert report["weakest_dimension"] == "safety"
        assert report["dimensions"][0]["dimension"] == "safety"

    def test_only_failures_are_listed(self):
        run = _run_with(
            [
                Evaluation(
                    probe_id="p1",
                    scores={Dimension.ACCURACY: 5.0},
                    verdict=Verdict.PASS,
                ),
                Evaluation(
                    probe_id="p2",
                    scores={Dimension.ACCURACY: 1.0},
                    verdict=Verdict.FAIL,
                ),
            ]
        )
        run.score = aggregate(run)
        report = build_report(run)
        assert [f["probe_id"] for f in report["failures"]] == ["p2"]

    def test_grade_boundaries(self):
        from app.orchestrator import _grade

        assert _grade(90) == "A"
        assert _grade(89.9) == "B"
        assert _grade(59.9) == "F"

    def test_records_every_validator_that_judged(self):
        run = _run_with(
            [
                Evaluation(
                    probe_id="p1",
                    scores={Dimension.ACCURACY: 5.0},
                    judged_by="local-gemma",
                ),
                Evaluation(
                    probe_id="p2",
                    scores={Dimension.ACCURACY: 3.0},
                    judged_by="gemini-flash",
                ),
            ]
        )
        run.score = aggregate(run)
        report = build_report(run)
        assert report["validators_used"] == ["gemini-flash", "local-gemma"]
