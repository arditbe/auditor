"""Core domain types for a validation run.

A run is a sequence of probes. For each probe we ask the *target* model a
question, then ask the *validator* model to score the answer against a rubric.
Everything is streamed to the UI as it happens.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Dimension(str, Enum):
    """Scoring axes. Each probe targets one, but the judge scores all it can."""

    ACCURACY = "accuracy"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    INSTRUCTION_FOLLOWING = "instruction_following"
    SAFETY = "safety"
    COHERENCE = "coherence"


# Relative weight of each dimension in the headline score.
DIMENSION_WEIGHTS: dict[Dimension, float] = {
    Dimension.ACCURACY: 0.30,
    Dimension.HALLUCINATION_RESISTANCE: 0.25,
    Dimension.INSTRUCTION_FOLLOWING: 0.20,
    Dimension.SAFETY: 0.15,
    Dimension.COHERENCE: 0.10,
}

MAX_DIMENSION_SCORE = 5.0


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Probe(BaseModel):
    """A single question put to the model under test."""

    probe_id: str = Field(default_factory=lambda: _new_id("probe"))
    index: int
    question: str
    dimension: Dimension
    difficulty: str = "medium"
    # What a good answer must contain / avoid. Handed to the judge.
    criteria: str = ""
    # True when the question is deliberately unanswerable or premised on a
    # falsehood -- the correct behaviour is to decline, not to answer.
    is_trap: bool = False
    created_at_ms: int = Field(default_factory=_now_ms)


class TargetResponse(BaseModel):
    """What the model under test said."""

    probe_id: str
    text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    created_at_ms: int = Field(default_factory=_now_ms)


class Evaluation(BaseModel):
    """The validator agent's judgement of one response."""

    probe_id: str
    scores: dict[Dimension, float] = Field(default_factory=dict)
    verdict: Verdict = Verdict.WARN
    reasoning: str = ""
    flags: list[str] = Field(default_factory=list)
    # Which validator produced this -- runs can switch validators mid-flight.
    judged_by: str = ""
    latency_ms: int = 0
    created_at_ms: int = Field(default_factory=_now_ms)

    @property
    def weighted_score(self) -> float:
        """0-100 for this single evaluation, over whichever dims were scored."""
        return weighted_score(self.scores)


def weighted_score(scores: dict[Dimension, float]) -> float:
    """Weighted mean of dimension scores, renormalised to 0-100."""
    total_weight = sum(DIMENSION_WEIGHTS[d] for d in scores)
    if total_weight <= 0:
        return 0.0
    acc = sum(DIMENSION_WEIGHTS[d] * v for d, v in scores.items())
    return round((acc / total_weight) / MAX_DIMENSION_SCORE * 100, 1)


class ScoreSnapshot(BaseModel):
    """Live score after N completed probes."""

    overall: float = 0.0
    dimensions: dict[Dimension, float] = Field(default_factory=dict)
    completed: int = 0
    total: int = 0
    passes: int = 0
    warns: int = 0
    fails: int = 0


class RunConfig(BaseModel):
    """Everything the user chose on the launch screen."""

    target_model: str          # provider-qualified, e.g. "ollama:qwen2:0.5b"
    validator_model: str       # key into the validator registry
    suite: str = "general"
    num_probes: int = 8
    system_prompt: str | None = None
    # Free-text description of what the model was tuned for. Steers probes.
    model_purpose: str = ""


class ValidationRun(BaseModel):
    run_id: str = Field(default_factory=lambda: _new_id("run"))
    config: RunConfig
    status: RunStatus = RunStatus.PENDING
    score: ScoreSnapshot = Field(default_factory=ScoreSnapshot)
    probes: list[Probe] = Field(default_factory=list)
    responses: dict[str, TargetResponse] = Field(default_factory=dict)
    evaluations: dict[str, Evaluation] = Field(default_factory=dict)
    report: dict[str, Any] | None = None
    error: str | None = None
    created_at_ms: int = Field(default_factory=_now_ms)
    completed_at_ms: int | None = None
