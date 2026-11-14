"""Decision schema for the LLM-backed Jev (validated, never trusted raw)."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["JevScoreEntry", "JevDecision", "parse_jev_decision"]


class JevScoreEntry(BaseModel):
    """One oscillator's gate score in [0, 1]."""

    oscillatorId: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class JevDecision(BaseModel):
    """The cheap decision model's full response for one fire decision."""

    scores: list[JevScoreEntry]


def parse_jev_decision(payload: dict) -> JevDecision | None:
    """Validate a raw dict into a JevDecision; None on any schema failure.

    A failed parse means "no verified opinion" - callers must treat it as
    zero scores, never as a fire.
    """
    try:
        return JevDecision.model_validate(payload)
    except Exception:
        return None
