"""Tolerant JSON extraction from model output.

Small local models fence their JSON, prepend "Here you go:", or emit a trailing
comma. The auditor has to keep running when that happens, so parsing is
forgiving by design -- but it never invents data: if nothing parses, the caller
gets None and handles the failure explicitly.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _iter_balanced_spans(text: str, open_ch: str, close_ch: str):
    """Yield each balanced {...} / [...] region, ignoring delimiters in strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1


def _balanced_span(text: str, open_ch: str, close_ch: str) -> str | None:
    """The first balanced region, or None."""
    return next(_iter_balanced_spans(text, open_ch, close_ch), None)


def _candidates(text: str) -> list[str]:
    """Progressively less literal readings of the model's output."""
    out: list[str] = [text.strip()]
    out.extend(m.strip() for m in _FENCE.findall(text))
    for opener, closer in (("{", "}"), ("[", "]")):
        span = _balanced_span(text, opener, closer)
        if span:
            out.append(span)
    return [c for c in out if c]


def extract_json(text: str) -> Any | None:
    """Best-effort parse of the first JSON value in `text`. None if hopeless."""
    if not text:
        return None

    for candidate in _candidates(text):
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def extract_object(text: str) -> dict | None:
    """As `extract_json`, but only accepts a JSON object."""
    value = extract_json(text)
    return value if isinstance(value, dict) else None


def _try_load(span: str) -> Any | None:
    for attempt in (span, _TRAILING_COMMA.sub(r"\1", span)):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


#: Guards against pathological nesting in a runaway generation.
_MAX_SALVAGE_DEPTH = 6


def salvage_objects(text: str, _depth: int = 0) -> list[dict]:
    """Every individually well-formed JSON object in `text`.

    A smaller model writing a long array will occasionally corrupt one element
    -- a dropped quote, a missing comma -- which makes the enclosing array
    unparseable even though the other elements are perfectly good. Rather than
    discard the whole run, descend past the broken wrapper and take what parses.

    Only used as a fallback, after a strict parse has already failed.
    """
    if _depth > _MAX_SALVAGE_DEPTH:
        return []

    out: list[dict] = []
    for span in _iter_balanced_spans(text, "{", "}"):
        value = _try_load(span)
        if isinstance(value, dict):
            out.append(value)
            continue
        # This object is malformed. Its children may not be, so look inside
        # rather than throwing the whole span away.
        out.extend(salvage_objects(span[1:-1], _depth + 1))
    return out
