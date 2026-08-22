from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.context_upload import extract_uploaded_context  # noqa: E402


def _upload(name: str, text: str) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(text.encode("utf-8")))


@pytest.mark.asyncio
async def test_large_csv_context_is_sampled_in_parts():
    rows = ["prompt,completion"]
    rows.extend(f"question {i},answer {i}" for i in range(1, 101))

    context = await extract_uploaded_context(_upload("context.csv", "\n".join(rows)))

    assert "100 rows" in context.text
    assert "representative chunks only" in context.text
    assert "part 1: rows 1-6" in context.text
    assert "row 100: prompt: question 100" in context.text
    assert "row 7: prompt: question 7" not in context.text


@pytest.mark.asyncio
async def test_large_text_context_uses_beginning_middle_and_end():
    text = "A" * 9000 + "MIDDLE_MARKER" + "B" * 9000 + "END_MARKER"

    context = await extract_uploaded_context(_upload("notes.txt", text))

    assert context.truncated is True
    assert "MIDDLE_MARKER" in context.text
    assert "END_MARKER" in context.text
