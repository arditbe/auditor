"""Uploaded context for probe generation."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi import UploadFile
from pydantic import BaseModel

MAX_CONTEXT_CHARS = 12000
MAX_CELL_CHARS = 420
MAX_CSV_ROWS_INLINE = 24
CSV_CHUNK_COUNT = 4
CSV_ROWS_PER_CHUNK = 6


class UploadedContext(BaseModel):
    filename: str
    text: str
    chars: int
    truncated: bool = False


def _flatten_csv_context(text: str) -> str:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return text
    rows = list(reader)
    if not rows:
        return "CSV columns: " + ", ".join(reader.fieldnames)

    lines = [
        f"CSV context summary: {len(rows)} rows, columns: "
        + ", ".join(reader.fieldnames)
    ]

    if len(rows) <= MAX_CSV_ROWS_INLINE:
        lines.append("Rows included: all rows")
        lines.extend(_format_csv_rows(rows, start_index=1))
        return "\n".join(lines)

    spans = _csv_sample_spans(len(rows))
    lines.append(
        "Rows included: representative chunks only, not the full CSV, to keep "
        "validator token use bounded."
    )
    for chunk_index, (start, end) in enumerate(spans, start=1):
        lines.append(f"\npart {chunk_index}: rows {start + 1}-{end}")
        lines.extend(_format_csv_rows(rows[start:end], start_index=start + 1))
    return "\n".join(lines)


def _csv_sample_spans(total_rows: int) -> list[tuple[int, int]]:
    max_start = max(0, total_rows - CSV_ROWS_PER_CHUNK)
    starts = [
        round(i * max_start / (CSV_CHUNK_COUNT - 1))
        for i in range(CSV_CHUNK_COUNT)
    ]
    spans = []
    seen = set()
    for start in starts:
        end = min(total_rows, start + CSV_ROWS_PER_CHUNK)
        key = (start, end)
        if key not in seen:
            spans.append(key)
            seen.add(key)
    return spans


def _format_csv_rows(rows: list[dict], start_index: int) -> list[str]:
    lines = []
    for offset, row in enumerate(rows):
        parts = []
        for key, value in row.items():
            value = _clip_cell(str(value or "").strip())
            if key and value:
                parts.append(f"{key}: {value}")
        if parts:
            lines.append(f"row {start_index + offset}: " + " | ".join(parts))
    return lines


def _clip_cell(value: str) -> str:
    if len(value) <= MAX_CELL_CHARS:
        return value
    return value[:MAX_CELL_CHARS].rstrip() + "..."


def _flatten_json_context(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _bounded_text_context(text: str) -> tuple[str, bool]:
    text = text.strip()
    if len(text) <= MAX_CONTEXT_CHARS:
        return text, False

    part_budget = MAX_CONTEXT_CHARS // 3
    middle_start = max(0, len(text) // 2 - part_budget // 2)
    middle_end = min(len(text), middle_start + part_budget)
    end_start = max(0, len(text) - part_budget)
    parts = [
        ("beginning", text[:part_budget].rstrip()),
        ("middle", text[middle_start:middle_end].strip()),
        ("end", text[end_start:].lstrip()),
    ]
    sampled = [
        "Large file context sampled in parts, not included in full, to keep validator token use bounded."
    ]
    for label, content in parts:
        sampled.append(f"\npart: {label}\n{content}")
    return "\n".join(sampled), True


async def extract_uploaded_context(upload: UploadFile) -> UploadedContext:
    raw = await upload.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Could not read this file as text. Upload CSV, JSON, TXT, MD, or another UTF-8 text file."
        ) from exc

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".csv":
        text = _flatten_csv_context(text)
    elif suffix in {".json", ".jsonl"}:
        text = _flatten_json_context(text)

    text = text.strip()
    if not text:
        raise ValueError("Uploaded file did not contain readable text.")

    text, truncated = _bounded_text_context(text)
    return UploadedContext(
        filename=upload.filename or "uploaded-file",
        text=text,
        chars=len(text),
        truncated=truncated,
    )
