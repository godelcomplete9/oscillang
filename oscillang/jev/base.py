"""Jev interface and per-tick context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["JevContext", "JevModel"]


@dataclass(frozen=True)
class JevContext:
    """Feature vectors for one tick, already aligned to lattice rows.

    The runtime assembles these; a JevModel must stay a pure function of the
    context (plus its own parameters). ``noise`` is pre-drawn by the runtime's
    seeded generator so every implementation stays deterministic under test.
    """

    ids: list[str]
    coherence: np.ndarray  # (N,) skill-text similarity to the state vector
    phase_prox: np.ndarray  # (N,) 1 - wrapped phase distance to reference / pi
    memory_sim: np.ndarray  # (N,) similarity to the leaky memory trace
    amplitude: np.ndarray  # (N,) 1 - fatigue
    used_cost: np.ndarray  # (N,) 0..1 over-use penalty (+ flagged penalty)
    coupling: np.ndarray  # (N,) 0..1 normalized partner pressure
    refractory: np.ndarray  # (N,) bool
    noise: np.ndarray  # (N,) seeded, in (0, 0.2) sigma

    def __post_init__(self) -> None:
        n = len(self.ids)
        for name in (
            "coherence",
            "phase_prox",
            "memory_sim",
            "amplitude",
            "used_cost",
            "coupling",
            "noise",
        ):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"JevContext.{name} has shape {arr.shape[0]}, expected {n}")
        if self.refractory.shape[0] != n:
            raise ValueError("JevContext.refractory length mismatch")


@runtime_checkable
class JevModel(Protocol):
    """Anything that can score one tick's oscillators."""

    name: str

    def score(self, ctx: JevContext) -> np.ndarray: ...
