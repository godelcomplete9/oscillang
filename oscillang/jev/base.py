"""Jev interface: the decision layer that gates mini skills.

Jev verifies semantic coherence and alignment between the user input and the
candidate behavioral fragments. It is a reranker and the main gatekeeper of
a mini skill: candidates arrive with their actual skill texts, and its
verdict decides which of them fire. The oscillator layer contributes only
dynamics (phase lock, refractory, fatigue, coupling) as a hard gate on top -
never a semantic signal, and no embedding cosine anywhere.

Two roles share this package:

- ``JevModel`` (``score``): a numeric dynamics scorer over ``JevContext``.
  ``RuleJev`` is the deterministic prefilter; ``MlpJev`` is the learned hook.
  A numeric model may also gate alone (dynamics-only mode).
- ``JevReranker`` (``rerank``): the semantic verifier over ``JevCandidate``
  texts. ``LlmJev`` is the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "JevContext",
    "JevCandidate",
    "JevVerdict",
    "JevModel",
    "JevReranker",
]


@dataclass(frozen=True)
class JevContext:
    """Dynamics-only feature vectors for one tick, aligned to lattice rows.

    The runtime assembles these; a JevModel must stay a pure function of the
    context (plus its own parameters). ``noise`` is pre-drawn by the runtime's
    seeded generator so every implementation stays deterministic under test.
    """

    ids: list[str]
    phase_prox: np.ndarray  # (N,) 1 - wrapped phase distance to reference / pi
    recency: np.ndarray  # (N,) 0..1 boost from the leaky recency trace
    amplitude: np.ndarray  # (N,) 1 - fatigue
    used_cost: np.ndarray  # (N,) 0..1 over-use penalty (+ flagged penalty)
    coupling: np.ndarray  # (N,) 0..1 normalized partner pressure
    refractory: np.ndarray  # (N,) bool
    noise: np.ndarray  # (N,) seeded, in (0, 0.2) sigma

    def __post_init__(self) -> None:
        n = len(self.ids)
        for name in ("phase_prox", "recency", "amplitude", "used_cost", "coupling", "noise"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"JevContext.{name} has shape {arr.shape[0]}, expected {n}")
        if self.refractory.shape[0] != n:
            raise ValueError("JevContext.refractory length mismatch")


@dataclass(frozen=True)
class JevCandidate:
    """One candidate mini skill offered to the semantic reranker."""

    oscillator_id: str
    skill_id: str
    trigger: str
    prompt_fragment: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class JevVerdict:
    """The reranker's semantic alignment verdict for one candidate."""

    oscillator_id: str
    score: float  # 0..1 semantic alignment with the input
    reason: str | None = None


class JevModel(Protocol):
    """A numeric dynamics scorer: one V score per oscillator per tick."""

    name: str

    def score(self, ctx: JevContext) -> np.ndarray: ...


class JevReranker(Protocol):
    """A semantic alignment verifier over candidate mini skills."""

    name: str

    def rerank(
        self,
        user_input: str,
        candidates: list[JevCandidate],
        context: str = "",
    ) -> list[JevVerdict]: ...
