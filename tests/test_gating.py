"""Phase-only lock gating tests.

Gating requires phaseDiff < lockTolerance; semantic alignment is Jev's job
(the decision layer reranks mini skills) and never a property of this gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillang.core.config import Config
from oscillang.core.gating import (
    TWO_PI,
    lock_gate,
    lock_gate_reference,
    phase_diff_matrix,
    phase_diff_to_reference,
    wrap_phase_diff,
)


def test_unlocked_phases_do_not_gate() -> None:
    diff = phase_diff_matrix(np.array([0.0, 1.2], dtype=np.float32))
    gate = lock_gate(diff)
    assert not gate[0, 1]


def test_locked_phases_gate() -> None:
    diff = phase_diff_matrix(np.array([0.0, 0.15], dtype=np.float32))
    gate = lock_gate(diff)
    assert gate[0, 1]


def test_refractory_blocks_gate() -> None:
    diff = phase_diff_matrix(np.array([0.0, 0.1], dtype=np.float32))
    refractory = np.array([[False, True], [True, False]])
    gate = lock_gate(diff, refractory)
    assert not gate[0, 1]


def test_phase_gating_disabled_gates_everything() -> None:
    diff = phase_diff_matrix(np.array([0.0, 1.5], dtype=np.float32))
    gate = lock_gate(diff, phase_gating=False)
    assert gate.all()


def test_phase_gating_disabled_still_respects_refractory() -> None:
    diff = phase_diff_matrix(np.array([0.0, 1.5], dtype=np.float32))
    refractory = np.array([False, True])
    gate = lock_gate(diff, refractory[:, None] | refractory[None, :], phase_gating=False)
    assert gate[0, 0]
    assert not gate[1, 1]
    assert not gate[0, 1]


def test_wrap_phase_diff_stays_in_zero_pi() -> None:
    phases = np.linspace(0.0, 2.0 * TWO_PI, 33, dtype=np.float32)
    diff = wrap_phase_diff(phases)
    assert np.all(diff >= 0.0)
    assert np.all(diff <= np.pi + 1e-6)


def test_batched_candidates_mixed_gates() -> None:
    phases = np.array([0.0, 0.0, 0.0, 0.1, 1.0, 0.05], dtype=np.float32)
    diff = phase_diff_matrix(phases)
    refractory = np.array([False, False, False, False, False, True], dtype=bool)
    gate = lock_gate(diff, refractory[:, None] | refractory[None, :])
    assert gate[0, 3]
    assert gate[1, 2]  # identical phases lock even without any other signal
    assert not gate[0, 4]
    assert not gate[2, 4]
    assert not gate[0, 5]


def test_reference_gating_matches_pairwise() -> None:
    phases = np.array([0.0, 0.2, 1.0, 0.3], dtype=np.float32)
    reference = 0.05
    gate_ref = lock_gate_reference(phases, reference)
    diff = phase_diff_to_reference(phases, reference)
    gate_pair = lock_gate(diff)
    assert np.array_equal(gate_ref, gate_pair)
    assert gate_ref[0] and gate_ref[1] and gate_ref[3]
    assert not gate_ref[2]


def test_config_thresholds_applied() -> None:
    cfg = Config().validated()
    diff = phase_diff_matrix(np.array([0.0, 0.45], dtype=np.float32))
    gate = lock_gate(diff, lock_tolerance=cfg.lock_tolerance)
    assert not gate[0, 1]


@pytest.mark.parametrize("diff_val", [0.39, 0.41])
def test_tolerance_boundary(diff_val: float) -> None:
    diff = np.array([[0.0, diff_val]], dtype=np.float32)
    gate = lock_gate(diff)
    assert bool(gate[0, 1]) == (diff_val < 0.4)
