"""Lock gating: gate only when phase-lock AND coherence agree.

phaseDiff is wrapped to [0, pi]; an oscillator gates only if
``phaseDiff < lockTolerance`` (0.4 rad) AND ``coherence > lockThreshold``
(0.6). High similarity with low phase lock must NOT gate. All functions are
batched across candidates.
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
    coherence: np.ndarray,
    refractory: np.ndarray | None = None,
    lock_tolerance: float = 0.4,
    lock_threshold: float = 0.6,
    phase_gating: bool = True,
) -> np.ndarray:
    """Boolean gate mask: phase-locked AND coherent AND not refractory.

    Shapes of ``phase_diff`` and ``coherence`` must match ((k, k) pairwise or
    (k,) to-reference). ``refractory`` broadcasts.
    """
    if phase_gating:
        gate = (phase_diff < lock_tolerance) & (coherence > lock_threshold)
    else:
        gate = coherence > lock_threshold
    if refractory is not None:
        gate = gate & ~np.broadcast_to(np.asarray(refractory, dtype=bool), gate.shape)
    return gate


def lock_gate_reference(
    phase: np.ndarray,
    reference_phase: float,
    coherence: np.ndarray,
    refractory: np.ndarray | None = None,
    lock_tolerance: float = 0.4,
    lock_threshold: float = 0.6,
    phase_gating: bool = True,
) -> np.ndarray:
    """(k,) gate mask of candidates against a single reference phase."""
    return lock_gate(
        phase_diff_to_reference(phase, reference_phase),
        coherence,
        refractory,
        lock_tolerance,
        lock_threshold,
        phase_gating,
    )


def retrieval_top_k(
    scores: np.ndarray,
    k: int,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One batched top-K selection by score (embedding similarity et al).

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
