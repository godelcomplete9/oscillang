"""Decision schema for the semantic Jev reranker (validated, never trusted raw).

Each score is an alignment verdict in [0, 1]: how semantically aligned the
candidate mini skill is with the input. A failed parse means "no verified
opinion" - callers must treat it as zero verdicts, never as a fire.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["JevScoreEntry", "JevDecision", "parse_jev_decision"]


class JevScoreEntry(BaseModel):
    """One candidate's alignment verdict in [0, 1]."""

    oscillatorId: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class JevDecision(BaseModel):
    """The semantic gatekeeper's full response for one fire decision."""

    scores: list[JevScoreEntry]


def parse_jev_decision(payload: dict) -> JevDecision | None:
    """Validate a raw dict into a JevDecision; None on any schema failure."""
    try:
        return JevDecision.model_validate(payload)
    except Exception:
        return None
