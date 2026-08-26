"""The validation run loop.

One run = generate probes, then for each probe: ask the model under test, ask
the validator agent to score the answer, update the live score. Every step
publishes an event before moving on, because the dashboard is the deliverable.

A run survives partial failure. If the target errors on one probe, or the judge
returns unparseable output, that probe is recorded as a failure and the run
continues -- an auditor that aborts on first error cannot audit a bad model,
which is exactly the case it exists for.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import Counter

from .agent import prompts
from .agent.parsing import extract_object, salvage_objects
from .agent.validator_agent import invoke as agent_invoke
from .config import settings
from .events import EventType, bus
from .models.schemas import (
    DIMENSION_WEIGHTS,
    Dimension,
    Evaluation,
    Probe,
    RunConfig,
    RunStatus,
    ScoreSnapshot,
    TargetResponse,
    ValidationRun,
    Verdict,
    weighted_score,
)
from .providers import TargetModel, build_target, get_validator
from .store import get_store

log = logging.getLogger(__name__)

MAX_PROBES = 25


class RunCancelled(Exception):
    """Raised inside the loop when the user cancels."""


class RunController:
    """Live handle on an in-flight run.

    Holds the mutable bits the API can poke at while the loop is running:
    the active validator (switchable mid-run) and the cancel flag.
    """

    def __init__(self, run: ValidationRun) -> None:
        self.run = run
        self.validator_key = run.config.validator_model
        self._cancelled = False
        self.task: asyncio.Task | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise RunCancelled()

    def switch_validator(self, new_key: str) -> str:
        """Point the rest of the run at a different validator. Returns the old key."""
        get_validator(new_key)  # raises KeyError if unknown
        previous, self.validator_key = self.validator_key, new_key
        return previous


#: Runs currently executing, by run_id.
ACTIVE: dict[str, RunController] = {}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def aggregate(run: ValidationRun) -> ScoreSnapshot:
    """Recompute the live score from every evaluation recorded so far."""
    per_dimension: dict[Dimension, list[float]] = {}
    verdicts: Counter[Verdict] = Counter()

    for evaluation in run.evaluations.values():
        verdicts[evaluation.verdict] += 1
        for dimension, value in evaluation.scores.items():
            per_dimension.setdefault(dimension, []).append(value)

    means = {d: round(statistics.fmean(v), 2) for d, v in per_dimension.items()}
    return ScoreSnapshot(
        overall=weighted_score(means),
        dimensions=means,
        completed=len(run.evaluations),
        total=len(run.probes) or run.config.num_probes,
        passes=verdicts[Verdict.PASS],
        warns=verdicts[Verdict.WARN],
        fails=verdicts[Verdict.FAIL],
    )


def _failed_evaluation(probe_id: str, judged_by: str, reason: str) -> Evaluation:
    """The model under test could not produce a usable answer.

    Scored zero rather than skipped: a model that cannot answer has failed the
    probe, and silently dropping it would flatter the final score.
    """
    return Evaluation(
        probe_id=probe_id,
        scores={Dimension.ACCURACY: 0.0},
        verdict=Verdict.FAIL,
        reasoning=reason,
        flags=["empty-response"],
        judged_by=judged_by,
    )


def _unscored_evaluation(probe_id: str, judged_by: str, reason: str) -> Evaluation:
    """The *judge* failed -- a timeout, a quota limit, unparseable output.

    Deliberately carries no dimension scores, so it contributes nothing to the
    result. Scoring the model zero here would blame it for Google rate-limiting
    us, which is how a rate limit ends up looking like a bad model.
    """
    return Evaluation(
        probe_id=probe_id,
        scores={},
        verdict=Verdict.WARN,
        reasoning=reason,
        flags=["not-scored"],
        judged_by=judged_by,
    )


#: Vertex hands a brand-new project very little Gemini quota, so 429s are
#: routine rather than exceptional. They are worth waiting out.
_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")


def _is_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _RATE_LIMIT_MARKERS)


# --------------------------------------------------------------------------
# probe generation
# --------------------------------------------------------------------------

def _coerce_probes(raw: object, limit: int) -> list[Probe]:
    """Turn the generator agent's JSON into validated Probe objects."""
    items: list[dict] = []
    if isinstance(raw, dict):
        candidate = raw.get("probes")
        if isinstance(candidate, list):
            items = [i for i in candidate if isinstance(i, dict)]
    elif isinstance(raw, list):
        items = [i for i in raw if isinstance(i, dict)]

    probes: list[Probe] = []
    for index, item in enumerate(items[:limit]):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        try:
            dimension = Dimension(str(item.get("dimension", "accuracy")).strip())
        except ValueError:
            dimension = Dimension.ACCURACY
        probes.append(
            Probe(
                index=len(probes),
                question=question,
                dimension=dimension,
                difficulty=str(item.get("difficulty", "medium")),
                criteria=str(item.get("criteria", "")).strip(),
                is_trap=bool(item.get("is_trap", False)),
            )
        )
    return probes


#: Probe generation is one long structured reply, and a smaller judge model
#: gets it wrong often enough that a single attempt is not good enough.
PROBE_GENERATION_ATTEMPTS = 3

#: Kept low: a probe is cheap to re-judge, but a run stalls if every probe
#: burns three judge calls. Rate limits get their own, more patient, retries.
JUDGE_ATTEMPTS = 3

#: First backoff after a 429; doubles each attempt.
RATE_LIMIT_BACKOFF_S = 8.0

_JSON_CORRECTION = (
    "\n\nYour previous reply could not be parsed as JSON. Common causes: a "
    "missing comma between fields, an unescaped double quote inside a string, "
    "or commentary outside the JSON. Reply again with the probe set as a "
    "single valid JSON object and nothing else."
)


def pick_weak_dimension(
    run: ValidationRun, already_probed: set[Dimension]
) -> tuple[Dimension, float] | None:
    """The agent's own judgement about where to dig.

    Returns the weakest dimension worth investigating, or None if the model
    held up. Dimensions already drilled into are skipped so a follow-up round
    cannot chase the same weakness forever.
    """
    if not settings.adaptive_probing:
        return None

    candidates = [
        (d, mean)
        for d, mean in run.score.dimensions.items()
        if mean < settings.adaptive_threshold and d not in already_probed
    ]
    if not candidates:
        return None
    # Weakest first: that is where the model is most likely to keep failing.
    return min(candidates, key=lambda kv: kv[1])


async def generate_focused_probes(
    controller: RunController,
    dimension: Dimension,
    mean_score: float,
    round_no: int,
) -> list[Probe]:
    """Ask the agent for harder probes aimed at one weakness."""
    config = controller.run.config
    message = prompts.focused_probe_prompt(
        dimension=dimension.value,
        mean_score=mean_score,
        num_probes=settings.adaptive_probes_per_round,
        model_purpose=config.model_purpose,
        target_spec=config.target_model,
        asked=[p.question for p in controller.run.probes],
    ) + prompts.FOCUSED_INSTRUCTION_SUFFIX

    raw = await agent_invoke(
        validator_key=controller.validator_key,
        role="generator",
        message=message,
        session_id=f"{controller.run.run_id}-focus-{round_no}",
        timeout=settings.judge_timeout_s * 2,
    )
    probes = _coerce_probes(
        extract_object(raw), settings.adaptive_probes_per_round
    ) or _coerce_probes(
        {"probes": salvage_objects(raw)}, settings.adaptive_probes_per_round
    )

    # Renumber onto the end of the existing plan and record why they exist.
    start = len(controller.run.probes)
    for offset, probe in enumerate(probes):
        probe.index = start + offset
        probe.round = round_no
        probe.dimension = dimension
        probe.reason = (
            f"{dimension.value} scored {mean_score:.2f}/5 in the previous "
            f"round, so the agent went deeper."
        )
    return probes


async def generate_probes(controller: RunController) -> list[Probe]:
    """Ask the agent for a probe set, retrying on unparseable output.

    Falls back to salvaging whatever individual probes parsed rather than
    failing the run outright -- five good probes beat no audit at all.
    """
    config = controller.run.config
    base_message = prompts.probe_generation_prompt(
        suite=config.suite,
        num_probes=config.num_probes,
        model_purpose=config.model_purpose,
        target_spec=config.target_model,
    )

    last_raw = ""
    for attempt in range(1, PROBE_GENERATION_ATTEMPTS + 1):
        controller.check_cancelled()
        message = base_message + (_JSON_CORRECTION if attempt > 1 else "")
        last_raw = await agent_invoke(
            validator_key=controller.validator_key,
            role="generator",
            message=message,
            # A fresh session each attempt, so a malformed reply is not left in
            # context for the model to imitate.
            session_id=f"{controller.run.run_id}-gen-{attempt}",
            timeout=settings.judge_timeout_s * 2,
        )

        probes = _coerce_probes(extract_object(last_raw), config.num_probes)
        if probes:
            return probes

        salvaged = _coerce_probes(
            {"probes": salvage_objects(last_raw)}, config.num_probes
        )
        # Only accept a partial set once retries are spent; a clean set is
        # better than a salvaged one if another attempt can produce it.
        if salvaged and attempt == PROBE_GENERATION_ATTEMPTS:
            log.warning(
                "run %s: salvaged %d probes from malformed output",
                controller.run.run_id,
                len(salvaged),
            )
            return salvaged

        log.warning(
            "run %s: probe generation attempt %d/%d was unparseable",
            controller.run.run_id,
            attempt,
            PROBE_GENERATION_ATTEMPTS,
        )

    raise RuntimeError(
        f"The validator model did not return a usable probe set after "
        f"{PROBE_GENERATION_ATTEMPTS} attempts. Its last reply began: "
        f"{last_raw[:200]!r}"
    )


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------

def _coerce_evaluation(
    raw: dict | None, probe: Probe, judged_by: str, latency_ms: int
) -> Evaluation | None:
    if not raw:
        return None

    scores: dict[Dimension, float] = {}
    for key, value in (raw.get("scores") or {}).items():
        try:
            dimension = Dimension(str(key).strip())
        except ValueError:
            continue
        try:
            # Clamp: models occasionally emit 7/5 or -1.
            scores[dimension] = max(0.0, min(5.0, float(value)))
        except (TypeError, ValueError):
            continue

    if not scores:
        return None

    try:
        verdict = Verdict(str(raw.get("verdict", "")).strip().lower())
    except ValueError:
        # Derive it from the numbers rather than trusting a malformed label.
        lowest = min(scores.values())
        verdict = (
            Verdict.PASS if lowest >= 4 else Verdict.FAIL if lowest <= 2 else Verdict.WARN
        )

    flags = [str(f) for f in (raw.get("flags") or []) if isinstance(f, (str, int))]

    return Evaluation(
        probe_id=probe.probe_id,
        scores=scores,
        verdict=verdict,
        reasoning=str(raw.get("reasoning", "")).strip(),
        flags=flags,
        judged_by=judged_by,
        latency_ms=latency_ms,
    )


async def judge_response(
    controller: RunController, probe: Probe, answer: str
) -> Evaluation:
    validator_key = controller.validator_key
    message = prompts.judge_prompt(
        question=probe.question,
        criteria=probe.criteria,
        answer=answer,
        is_trap=probe.is_trap,
        dimension=probe.dimension.value,
    )
    started = time.perf_counter()

    # Retried because an unparseable verdict is the judge's failure, not the
    # model's -- scoring the model zero for it would corrupt the result.
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        try:
            raw = await agent_invoke(
                validator_key=validator_key,
                role="judge",
                message=message + (_JSON_CORRECTION if attempt > 1 else ""),
                # Fresh session per probe: the judge must not be primed by how
                # it scored the previous answer.
                session_id=f"{controller.run.run_id}-judge-{probe.probe_id}-{attempt}",
                timeout=settings.judge_timeout_s,
            )
        except asyncio.TimeoutError:
            return _unscored_evaluation(
                probe.probe_id,
                validator_key,
                "The judge timed out scoring this answer, so it was not scored.",
            )
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limited(exc) and attempt < JUDGE_ATTEMPTS:
                # Exponential backoff. A fresh Google Cloud project has very
                # little Gemini quota, and waiting is what actually fixes it.
                delay = RATE_LIMIT_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    "run %s probe %s: judge rate-limited, retrying in %.0fs",
                    controller.run.run_id,
                    probe.probe_id,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if _is_rate_limited(exc):
                return _unscored_evaluation(
                    probe.probe_id,
                    validator_key,
                    "The judge ran out of Google Cloud quota, so this answer "
                    "was not scored. Request more Gemini quota, or use a "
                    "local judge.",
                )
            return _unscored_evaluation(
                probe.probe_id,
                validator_key,
                f"The judge failed, so this answer was not scored. "
                f"{type(exc).__name__}: {exc}",
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        evaluation = _coerce_evaluation(
            extract_object(raw), probe, validator_key, latency_ms
        )
        if evaluation is not None:
            return evaluation

        log.warning(
            "run %s probe %s: judge reply %d/%d was unparseable",
            controller.run.run_id,
            probe.probe_id,
            attempt,
            JUDGE_ATTEMPTS,
        )

    return _unscored_evaluation(
        probe.probe_id,
        validator_key,
        "The judge's reply could not be read as a score, so this answer was "
        "not scored.",
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def build_report(run: ValidationRun) -> dict:
    score = run.score
    flags = Counter(
        flag for e in run.evaluations.values() for flag in e.flags
    )
    latencies = [r.latency_ms for r in run.responses.values() if not r.error]

    dimensions = []
    for dimension, mean in sorted(
        score.dimensions.items(), key=lambda kv: kv[1]
    ):
        dimensions.append(
            {
                "dimension": dimension.value,
                "mean_score": mean,
                "out_of": 5.0,
                "weight": DIMENSION_WEIGHTS[dimension],
                "probe_count": sum(
                    1 for e in run.evaluations.values() if dimension in e.scores
                ),
            }
        )

    failures = []
    for probe in run.probes:
        evaluation = run.evaluations.get(probe.probe_id)
        if not evaluation or evaluation.verdict != Verdict.FAIL:
            continue
        response = run.responses.get(probe.probe_id)
        failures.append(
            {
                "probe_id": probe.probe_id,
                "question": probe.question,
                "dimension": probe.dimension.value,
                "is_trap": probe.is_trap,
                "answer": (response.text if response else ""),
                "reasoning": evaluation.reasoning,
                "flags": evaluation.flags,
            }
        )

    return {
        "overall_score": score.overall,
        "grade": _grade(score.overall),
        "verdict_counts": {
            "pass": score.passes,
            "warn": score.warns,
            "fail": score.fails,
        },
        "dimensions": dimensions,
        # Sorted ascending, so the first entry is the weakest area.
        "weakest_dimension": dimensions[0]["dimension"] if dimensions else None,
        "flags": dict(flags.most_common()),
        "failures": failures,
        "target_model": run.config.target_model,
        "validators_used": sorted(
            {e.judged_by for e in run.evaluations.values() if e.judged_by}
        ),
        "suite": run.config.suite,
        "probes_run": len(run.evaluations),
        # What the agent chose to investigate without being asked.
        "focus_rounds": sorted({p.round for p in run.probes if p.round > 0}),
        "focused_on": sorted(
            {p.dimension.value for p in run.probes if p.round > 0}
        ),
        "target_latency_ms": {
            "mean": int(statistics.fmean(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "duration_ms": (run.completed_at_ms or int(time.time() * 1000))
        - run.created_at_ms,
    }


async def detect_regression(run: ValidationRun) -> dict | None:
    """Compare against the last completed audit of the same model.

    This is the decision a scheduled run exists to make: nobody is watching,
    so the agent has to work out on its own whether tonight's result is worth
    waking someone for.
    """
    try:
        history = await get_store().list_recent(limit=50)
    except Exception as exc:  # noqa: BLE001 - never fail a run over history
        log.warning("could not read history for regression check: %s", exc)
        return None

    previous = next(
        (
            r
            for r in history
            if r.run_id != run.run_id
            and r.config.target_model == run.config.target_model
            and r.status is RunStatus.COMPLETE
            and r.score.completed > 0
        ),
        None,
    )
    if previous is None:
        return {
            "baseline": None,
            "delta": None,
            "regressed": False,
            "summary": "First audit of this model — nothing to compare against.",
        }

    delta = round(run.score.overall - previous.score.overall, 1)
    regressed = delta <= -settings.regression_drop

    if regressed:
        summary = (
            f"Regression: down {abs(delta)} points from "
            f"{previous.score.overall} to {run.score.overall}."
        )
    elif delta >= settings.regression_drop:
        summary = f"Improved {delta} points since the last audit."
    else:
        summary = f"Stable, within {abs(delta)} points of the last audit."

    return {
        "baseline": previous.score.overall,
        "baseline_run_id": previous.run_id,
        "baseline_at_ms": previous.created_at_ms,
        "delta": delta,
        "regressed": regressed,
        "summary": summary,
    }


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# --------------------------------------------------------------------------
# the run loop
# --------------------------------------------------------------------------

async def execute_run(controller: RunController) -> None:
    run = controller.run
    store = get_store()
    target: TargetModel | None = None

    async def emit(type_: EventType, data: dict | None = None) -> None:
        await bus.publish(run.run_id, type_, data or {})

    try:
        run.status = RunStatus.RUNNING
        await store.save(run)
        await emit(
            EventType.RUN_STARTED,
            {
                "run_id": run.run_id,
                "config": run.config.model_dump(),
                "validator": controller.validator_key,
            },
        )

        # Built inside the try: a prepared model has to start a local server
        # here, and if that fails the run must report it rather than vanish.
        await emit(EventType.LOG, {"message": "Connecting to the model..."})
        target = await build_target(run.config.target_model)

        # 1. The agent designs the test.
        await emit(EventType.LOG, {"message": "Designing probe set..."})
        controller.check_cancelled()
        run.probes = await generate_probes(controller)
        run.score = aggregate(run)
        await store.save(run)
        await emit(
            EventType.PLAN_READY,
            {
                "probes": [p.model_dump(mode="json") for p in run.probes],
                "count": len(run.probes),
            },
        )

        async def run_probe(probe: Probe) -> None:
            """Ask one question, judge the answer, publish both."""
            controller.check_cancelled()
            await emit(
                EventType.PROBE_STARTED,
                {"probe": probe.model_dump(mode="json")},
            )

            completion = await target.generate(
                probe.question, system=run.config.system_prompt
            )
            response = TargetResponse(
                probe_id=probe.probe_id,
                text=completion.text,
                latency_ms=completion.latency_ms,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                error=completion.error,
            )
            run.responses[probe.probe_id] = response
            await emit(
                EventType.PROBE_ANSWERED,
                {
                    "probe_id": probe.probe_id,
                    "response": response.model_dump(mode="json"),
                },
            )

            controller.check_cancelled()
            if completion.error:
                evaluation = _failed_evaluation(
                    probe.probe_id,
                    controller.validator_key,
                    f"Model under test failed to respond: {completion.error}",
                )
            else:
                evaluation = await judge_response(controller, probe, completion.text)

            run.evaluations[probe.probe_id] = evaluation
            run.score = aggregate(run)
            await emit(
                EventType.PROBE_EVALUATED,
                {
                    "probe_id": probe.probe_id,
                    "evaluation": evaluation.model_dump(mode="json"),
                },
            )
            await emit(EventType.SCORE_UPDATED, run.score.model_dump(mode="json"))
            await store.save(run)

            if settings.probe_delay_s:
                await asyncio.sleep(settings.probe_delay_s)

        # 2. The opening pass.
        for probe in list(run.probes):
            await run_probe(probe)

        # 3. Follow-up rounds the agent decides on for itself.
        #
        # This is the part that makes it an agent rather than a script: having
        # seen the first results, it judges whether any dimension is weak
        # enough to be worth investigating, writes harder probes aimed at it,
        # and runs those too. Nobody asked it to.
        drilled: set[Dimension] = set()
        for round_no in range(1, settings.adaptive_max_rounds + 1):
            controller.check_cancelled()
            weak = pick_weak_dimension(run, drilled)
            if weak is None:
                break

            dimension, mean = weak
            drilled.add(dimension)
            await emit(
                EventType.FOCUS_STARTED,
                {
                    "round": round_no,
                    "dimension": dimension.value,
                    "mean_score": mean,
                    "reason": (
                        f"{dimension.value.replace('_', ' ')} scored "
                        f"{mean:.2f}/5 — going deeper"
                    ),
                },
            )

            follow_ups = await generate_focused_probes(
                controller, dimension, mean, round_no
            )
            if not follow_ups:
                log.warning("run %s: focus round %d produced no probes",
                            run.run_id, round_no)
                break

            run.probes.extend(follow_ups)
            run.score = aggregate(run)
            await emit(
                EventType.PLAN_READY,
                {
                    "probes": [p.model_dump(mode="json") for p in run.probes],
                    "count": len(run.probes),
                    "round": round_no,
                },
            )
            for probe in follow_ups:
                await run_probe(probe)

        # 4. Final report.
        run.status = RunStatus.COMPLETE
        run.completed_at_ms = int(time.time() * 1000)
        run.report = build_report(run)
        run.report["regression"] = await detect_regression(run)
        await store.save(run)
        await emit(EventType.RUN_COMPLETED, {"report": run.report})

    except RunCancelled:
        run.status = RunStatus.CANCELLED
        run.completed_at_ms = int(time.time() * 1000)
        # A cancelled run still gets a report for whatever it managed to score.
        run.report = build_report(run)
        await store.save(run)
        await emit(EventType.RUN_CANCELLED, {"report": run.report})

    except Exception as exc:  # noqa: BLE001
        log.exception("run %s failed", run.run_id)
        run.status = RunStatus.FAILED
        run.error = f"{type(exc).__name__}: {exc}"
        run.completed_at_ms = int(time.time() * 1000)
        await store.save(run)
        await emit(EventType.RUN_FAILED, {"error": run.error})

    finally:
        if target is not None:
            await target.close()
        await bus.close(run.run_id)
        ACTIVE.pop(run.run_id, None)


async def start_run(config: RunConfig) -> ValidationRun:
    """Validate the request, register the run, and kick off the loop."""
    get_validator(config.validator_model)  # fail fast on a bad validator key
    config.num_probes = max(1, min(MAX_PROBES, config.num_probes))
    if config.suite not in prompts.SUITES:
        config.suite = prompts.DEFAULT_SUITE

    run = ValidationRun(config=config)
    run.score = ScoreSnapshot(total=config.num_probes)
    controller = RunController(run)
    ACTIVE[run.run_id] = controller

    await get_store().save(run)
    controller.task = asyncio.create_task(execute_run(controller))
    return run
