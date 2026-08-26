"""Turning an audit into training data.

Finding out a model is bad is only half a job. This closes the loop: every
probe the model failed becomes a training example with a *corrected* answer,
written by the same judge that failed it — so the output is something you can
fine-tune on rather than a list of complaints.

Two formats are written because the two are used differently:
  * JSONL  — what a trainer consumes (mlx_lm, axolotl, PEFT all read it)
  * CSV    — what a human opens to check the corrections before training
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel

from .agent.parsing import extract_object
from .agent.validator_agent import invoke as agent_invoke
from .config import settings
from .models.schemas import ValidationRun, Verdict

log = logging.getLogger(__name__)

DATA_DIR = Path(
    os.environ.get("AUDITOR_HOME", Path.home() / ".auditor")
).expanduser() / "datasets"

#: A correction is a short answer, not an essay.
CORRECTION_TIMEOUT_S = 90.0

CORRECTION_INSTRUCTION = """\
You are writing training data to repair a model's specific failure.

You are given a question, the criteria a correct answer must meet, and the
wrong answer the model gave. Write the answer it *should* have given.

Rules:
- Satisfy every constraint in the criteria, including format and length ones.
- If the question is a trap -- unanswerable, or resting on a false premise --
  the correct answer declines and says why. Do not answer it anyway.
- Write only the answer itself. No preamble, no explanation of your reasoning,
  no quotes around it.

Respond with JSON only, no markdown fence:
{"answer": "...", "note": "one short sentence on what the model got wrong"}
"""


class DatasetFile(BaseModel):
    name: str
    path: str
    rows: int
    bytes: int
    run_id: str
    created_at_ms: int
    format: str


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.") or "dataset"


async def _correct(run: ValidationRun, probe, response, evaluation, validator: str) -> dict | None:
    """Ask the judge what the model should have said."""
    message = (
        f"QUESTION:\n{probe.question}\n\n"
        f"CRITERIA:\n{probe.criteria or '(none given)'}\n\n"
        f"{'This probe is a TRAP: the correct answer declines.' if probe.is_trap else ''}\n\n"
        f"THE MODEL'S WRONG ANSWER:\n{(response.text if response else '') or '(no answer)'}\n\n"
        f"WHY IT WAS WRONG:\n{evaluation.reasoning}\n\n"
        "Write the correct answer. JSON only."
    )
    try:
        raw = await agent_invoke(
            validator_key=validator,
            role="judge",
            message=CORRECTION_INSTRUCTION + "\n\n" + message,
            session_id=f"{run.run_id}-fix-{probe.probe_id}",
            timeout=CORRECTION_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the file
        log.warning("correction failed for %s: %s", probe.probe_id, exc)
        return None

    parsed = extract_object(raw)
    if not parsed or not str(parsed.get("answer", "")).strip():
        return None
    return {
        "answer": str(parsed["answer"]).strip(),
        "note": str(parsed.get("note", "")).strip(),
    }


async def build_dataset(
    run: ValidationRun, *, validator: str | None = None
) -> DatasetFile | None:
    """Write training data for everything this model got wrong.

    Returns None when there is nothing to fix, which is a good outcome rather
    than an error.
    """
    failures = [
        p
        for p in run.probes
        if (ev := run.evaluations.get(p.probe_id)) and ev.verdict is Verdict.FAIL
    ]
    if not failures:
        log.info("run %s had no failures; no dataset written", run.run_id)
        return None

    judge = validator or run.config.validator_model

    # Corrections are independent, so write them concurrently -- but bounded,
    # because a 20-failure run would otherwise open 20 parallel model calls.
    limit = asyncio.Semaphore(3)

    async def one(probe):
        async with limit:
            ev = run.evaluations[probe.probe_id]
            resp = run.responses.get(probe.probe_id)
            fix = await _correct(run, probe, resp, ev, judge)
            if fix is None:
                return None
            return {
                "prompt": probe.question,
                "completion": fix["answer"],
                "rejected": (resp.text if resp else "") or "",
                "dimension": probe.dimension.value,
                "is_trap": probe.is_trap,
                "why_it_failed": ev.reasoning,
                "note": fix["note"],
            }

    rows = [r for r in await asyncio.gather(*(one(p) for p in failures)) if r]
    if not rows:
        log.warning("run %s: every correction failed; no dataset written", run.run_id)
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    base = _safe_name(f"{run.config.target_model}-{stamp}")

    jsonl_path = DATA_DIR / f"{base}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Only the pair a trainer needs; the rest is review metadata.
            fh.write(json.dumps(
                {"prompt": row["prompt"], "completion": row["completion"]},
                ensure_ascii=False,
            ) + "\n")

    csv_path = DATA_DIR / f"{base}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info("run %s: wrote %d training rows to %s", run.run_id, len(rows), base)
    return DatasetFile(
        name=f"{base}.jsonl",
        path=str(jsonl_path),
        rows=len(rows),
        bytes=jsonl_path.stat().st_size,
        run_id=run.run_id,
        created_at_ms=int(time.time() * 1000),
        format="jsonl",
    )


def list_datasets() -> list[DatasetFile]:
    if not DATA_DIR.is_dir():
        return []
    out: list[DatasetFile] = []
    for path in DATA_DIR.glob("*.jsonl"):
        try:
            stat = path.stat()
            rows = sum(1 for _ in path.open(encoding="utf-8"))
        except OSError:
            continue
        out.append(
            DatasetFile(
                name=path.name,
                path=str(path),
                rows=rows,
                bytes=stat.st_size,
                run_id="",
                created_at_ms=int(stat.st_mtime * 1000),
                format="jsonl",
            )
        )
    return sorted(out, key=lambda d: d.created_at_ms, reverse=True)


def dataset_path(name: str) -> Path | None:
    """Resolve a dataset filename, refusing anything outside DATA_DIR."""
    candidate = (DATA_DIR / Path(name).name).resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
