"""Lock-resonance gating tests.

The critical invariant: high embedding similarity with low phase lock must
NOT gate. Gating requires BOTH phaseDiff < lockTolerance AND coherence >
lockThreshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillang.core.coherence import coherence_pairs
from oscillang.core.config import Config
from oscillang.core.gating import (
    lock_gate,
    lock_gate_reference,
    phase_diff_matrix,
    phase_diff_to_reference,
    wrap_phase_diff,
)

TWO_PI = 2.0 * float(np.pi)


def _coh(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    return coherence_pairs(np.vstack([emb_a, emb_b]), np.array([0, 1]))


def test_high_similarity_low_phase_lock_does_not_gate() -> None:
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([emb, emb])
    coh = _coh(embeddings[0], embeddings[1])
    assert coh[0, 1] > 0.99
    diff = phase_diff_matrix(np.array([0.0, 1.2], dtype=np.float32))
    gate = lock_gate(diff, coh)
    assert not gate[0, 1]


def test_high_similarity_and_locked_gates() -> None:
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([emb, emb])
    coh = _coh(embeddings[0], embeddings[1])
    diff = phase_diff_matrix(np.array([0.0, 0.15], dtype=np.float32))
    gate = lock_gate(diff, coh)
    assert gate[0, 1]


def test_locked_but_low_coherence_does_not_gate() -> None:
    emb_a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    emb_b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    coh = _coh(emb_a, emb_b)
    assert coh[0, 1] < 0.6
    diff = phase_diff_matrix(np.array([0.0, 0.05], dtype=np.float32))
    gate = lock_gate(diff, coh)
    assert not gate[0, 1]


def test_refractory_blocks_gate() -> None:
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([emb, emb])
    coh = _coh(embeddings[0], embeddings[1])
    diff = phase_diff_matrix(np.array([0.0, 0.1], dtype=np.float32))
    refractory = np.array([[False, True], [True, False]])
    gate = lock_gate(diff, coh, refractory)
    assert not gate[0, 1]


def test_phase_gating_disabled_falls_back_to_coherence() -> None:
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([emb, emb])
    coh = _coh(embeddings[0], embeddings[1])
    diff = phase_diff_matrix(np.array([0.0, 1.5], dtype=np.float32))
    gate = lock_gate(diff, coh, phase_gating=False)
    assert gate[0, 1]


def test_wrap_phase_diff_stays_in_zero_pi() -> None:
    phases = np.linspace(0.0, 2.0 * TWO_PI, 33, dtype=np.float32)
    diff = wrap_phase_diff(phases)
    assert np.all(diff >= 0.0)
    assert np.all(diff <= np.pi + 1e-6)


def test_batched_candidates_mixed_gates() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(6, 8)).astype(np.float32)
    embeddings[3] = embeddings[0]
    embeddings[4] = embeddings[0]
    phases = np.array([0.0, 0.0, 0.0, 0.1, 1.0, 0.05], dtype=np.float32)
    idx = np.arange(6)
    coh = coherence_pairs(embeddings, idx)
    diff = phase_diff_matrix(phases)
    refractory = np.array([False, False, False, False, False, True], dtype=bool)
    gate = lock_gate(diff, coh, refractory[:, None] | refractory[None, :])
    assert gate[0, 3]
    assert not gate[0, 4]
    assert not gate[0, 5]
    assert not gate[1, 2]


def test_reference_gating_matches_pairwise() -> None:
    phases = np.array([0.0, 0.2, 1.0, 0.3], dtype=np.float32)
    coherence = np.array([0.9, 0.9, 0.9, 0.3], dtype=np.float32)
    reference = 0.05
    gate_ref = lock_gate_reference(phases, reference, coherence)
    diff = phase_diff_to_reference(phases, reference)
    gate_pair = lock_gate(diff, coherence)
    assert np.array_equal(gate_ref, gate_pair)
    assert gate_ref[0] and gate_ref[1]
    assert not gate_ref[2] and not gate_ref[3]


def test_config_thresholds_applied() -> None:
    cfg = Config().validated()
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([emb, emb])
    coh = _coh(embeddings[0], embeddings[1])
    diff = phase_diff_matrix(np.array([0.0, 0.45], dtype=np.float32))
    gate = lock_gate(
        diff, coh, lock_tolerance=cfg.lock_tolerance, lock_threshold=cfg.lock_threshold
    )
    assert not gate[0, 1]


@pytest.mark.parametrize("diff_val", [0.39, 0.41])
def test_tolerance_boundary(diff_val: float) -> None:
    diff = np.array([[0.0, diff_val]], dtype=np.float32)
    coh = np.array([[1.0, 0.9]], dtype=np.float32)
    gate = lock_gate(diff, coh)
    assert bool(gate[0, 1]) == (diff_val < 0.4)
