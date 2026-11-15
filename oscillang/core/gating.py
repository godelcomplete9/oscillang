"""Lock gating: a phase-only resonance gate.

phaseDiff is wrapped to [0, pi]; an oscillator joins a fire only if
``phaseDiff < lockTolerance`` (0.4 rad). Semantic alignment is NOT measured
here - that is Jev's job (the decision layer reranks mini skills for
alignment with the input). The oscillator layer supplies pure dynamics.
All functions are batched across candidates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_phase_diff",
    "phase_diff_matrix",
    "phase_diff_to_reference",
    "lock_gate",
    "lock_gate_reference",
    "retrieval_top_k",
]

TWO_PI = 2.0 * float(np.pi)


def wrap_phase_diff(diff: np.ndarray) -> np.ndarray:
    """Absolute phase difference wrapped into [0, pi]."""
    d = np.abs(np.asarray(diff, dtype=np.float32)) % TWO_PI
    return np.where(d > np.pi, TWO_PI - d, d).astype(np.float32)


def phase_diff_matrix(phase: np.ndarray) -> np.ndarray:
    """(k, k) pairwise wrapped phase differences for a candidate set."""
    p = np.asarray(phase, dtype=np.float32)
    return wrap_phase_diff(p[None, :] - p[:, None])


def phase_diff_to_reference(phase: np.ndarray, reference: float) -> np.ndarray:
    """(k,) wrapped phase difference of each oscillator to a reference phase."""
    return wrap_phase_diff(np.asarray(phase, dtype=np.float32) - reference)


def lock_gate(
    phase_diff: np.ndarray,
    refractory: np.ndarray | None = None,
    lock_tolerance: float = 0.4,
    phase_gating: bool = True,
) -> np.ndarray:
    """Boolean gate mask: phase-locked AND not refractory.

    Shapes of ``phase_diff`` and ``refractory`` broadcast against each other.
    With ``phase_gating`` off, the phase check is skipped (the ablation for
    dynamics-free runs) and only refractory vetoes.
    """
    if phase_gating:
        gate = np.asarray(phase_diff, dtype=np.float32) < lock_tolerance
    else:
        gate = np.ones(np.asarray(phase_diff).shape, dtype=bool)
    if refractory is not None:
        gate = gate & ~np.broadcast_to(np.asarray(refractory, dtype=bool), gate.shape)
    return gate


def lock_gate_reference(
    phase: np.ndarray,
    reference_phase: float,
    refractory: np.ndarray | None = None,
    lock_tolerance: float = 0.4,
    phase_gating: bool = True,
) -> np.ndarray:
    """(k,) gate mask of candidates against a single reference phase."""
    return lock_gate(
        phase_diff_to_reference(phase, reference_phase),
        refractory,
        lock_tolerance,
        phase_gating,
    )


def retrieval_top_k(
    scores: np.ndarray,
    k: int,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One batched top-K selection by score (Jev verdicts et al).

    Returns (indices, scores) sorted descending, length min(k, n).
    """
    s = np.asarray(scores, dtype=np.float32).copy()
    if exclude_mask is not None:
        s[exclude_mask] = -np.inf
    k = min(int(k), s.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    part = np.argpartition(-s, k - 1)[:k]
    order = np.argsort(-s[part])
    idx = part[order]
    return idx.astype(np.int64), s[idx]
