"""Unit tests for vectorized dynamics: phase, Kuramoto, fatigue, refractory."""

from __future__ import annotations

import numpy as np
import pytest

from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix
from oscillang.core.dynamics import (
    advance_phase,
    apply_use,
    drift_unused,
    kuramoto_term,
    recover_amplitude,
    step_lattice,
    update_refractory,
)
from oscillang.core.lattice import OscillatorLattice

TWO_PI = 2.0 * float(np.pi)


def test_advance_phase_wraps() -> None:
    phase = np.array([0.0, TWO_PI - 0.01, 1.0], dtype=np.float32)
    hz = np.array([0.5, 1.0, 0.2], dtype=np.float32)
    out = advance_phase(phase, hz, step=1.0)
    assert np.all(out >= 0.0) and np.all(out < TWO_PI)
    assert out[0] == pytest.approx(np.pi, abs=1e-5)


def test_advance_phase_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, TWO_PI, size=64).astype(np.float32)
    hz = rng.uniform(0.01, 1.5, size=64).astype(np.float32)
    out = advance_phase(phase, hz, step=2.0)
    expected = (phase + TWO_PI * hz * 2.0) % TWO_PI
    assert np.allclose(out, expected, atol=1e-5)


def test_apply_use_decays_fired_only() -> None:
    amp = np.array([1.0, 0.5], dtype=np.float32)
    fired = np.array([True, False])
    out = apply_use(amp, fired, decay_per_use=0.08, recovery_per_tick=0.02)
    assert out[0] == pytest.approx(0.08 + 0.02)
    assert out[1] == pytest.approx(0.5)


def test_apply_use_clamps_to_zero() -> None:
    amp = np.array([0.03], dtype=np.float32)
    out = apply_use(amp, np.array([True]), decay_per_use=0.08, recovery_per_tick=0.02)
    assert out[0] == pytest.approx(0.03 * 0.08 + 0.02)


def test_recover_and_drift() -> None:
    amp = np.array([0.4, 0.98], dtype=np.float32)
    fired = np.array([False, False])
    recovered = recover_amplitude(amp, fired, recovery_per_tick=0.02)
    assert recovered[0] == pytest.approx(0.42)
    assert recovered[1] == pytest.approx(1.0)
    amp3 = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    fired3 = np.array([False, False, False])
    drifted = drift_unused(amp3, fired3, 0.5, 0.1)
    assert drifted[0] == pytest.approx(0.23)
    assert drifted[1] == pytest.approx(0.77)
    assert drifted[2] == pytest.approx(0.5)


def test_update_refractory_fast_vs_slow() -> None:
    until = np.array([-1, -1, -1], dtype=np.int32)
    fired = np.array([True, True, False])
    hz = np.array([0.8, 0.2, 0.9], dtype=np.float32)
    out = update_refractory(until, fired, tick=7, hz=hz, fast_ticks=2, slow_ticks=5, fast_hz=0.6)
    assert out[0] == 9
    assert out[1] == 12
    assert out[2] == -1


def test_update_refractory_sets_fresh_window() -> None:
    until = np.array([50], dtype=np.int32)
    out = update_refractory(
        until,
        np.array([True]),
        tick=5,
        hz=np.array([0.1], dtype=np.float32),
        fast_ticks=2,
        slow_ticks=5,
        fast_hz=0.6,
    )
    assert out[0] == 10


def test_kuramoto_term_matches_naive() -> None:
    rng = np.random.default_rng(1)
    n = 12
    phase = rng.uniform(0, TWO_PI, size=n).astype(np.float32)
    cfg = Config(sparse_coupling_threshold=n + 1).validated()
    coupling = CouplingMatrix(n, cfg)
    rows = rng.integers(0, n, size=20)
    cols = rng.integers(0, n, size=20)
    coupling.apply_delta(rows, cols, rng.uniform(0.0, 1.0, size=20).astype(np.float32))
    mat = coupling.effective()
    naive = np.zeros(n)
    for i in range(n):
        for j in range(n):
            naive[i] += float(mat[i, j]) * np.sin(float(phase[j]) - float(phase[i]))
    fast = kuramoto_term(phase, coupling, k=1.0)
    assert np.allclose(fast, naive.astype(np.float32), atol=1e-4)
    sparse_check = kuramoto_term(phase, mat, k=1.0)
    assert np.allclose(fast, sparse_check, atol=1e-4)


def test_kuramoto_zero_for_identical_phases() -> None:
    phase = np.full(6, 0.7, dtype=np.float32)
    cfg = Config().validated()
    coupling = CouplingMatrix(6, cfg)
    coupling.apply_delta(np.arange(6), np.arange(6), np.ones(6, dtype=np.float32))
    assert np.allclose(kuramoto_term(phase, coupling, 1.0), 0.0, atol=1e-6)


def test_step_lattice_integration_defaults() -> None:
    cfg = Config().validated()
    lat = OscillatorLattice.generate(50, config=cfg, seed=9)
    fired = np.zeros(50, dtype=bool)
    fired[[2, 10]] = True
    amp_before = lat.amplitude.copy()
    step_lattice(lat, tick=0, config=cfg, fire_mask=fired)
    assert lat.amplitude[2] == pytest.approx(
        min(1.0, amp_before[2] * lat.decay_rate[2] + cfg.recovery_per_tick), abs=1e-6
    )
    assert lat.refractory_until_tick[10] == (2 if lat.hz[10] >= cfg.fast_hz else 5)
    assert np.allclose(lat.fatigue, 1.0 - lat.amplitude)


def test_step_lattice_ablation_flags() -> None:
    lat = OscillatorLattice.generate(20, seed=10)
    cfg = Config(fatigue=False).validated()
    amp_before = lat.amplitude.copy()
    step_lattice(lat, 0, cfg)
    assert np.allclose(lat.amplitude, np.clip(amp_before + cfg.recovery_per_tick, 0, 1))

    lat2 = OscillatorLattice.generate(20, seed=10)
    cfg2 = Config(refractory=False).validated()
    step_lattice(lat2, 0, cfg2, fire_mask=np.ones(20, dtype=bool))
    assert np.all(lat2.refractory_until_tick == -1)


def test_step_lattice_kuramoto_moves_phase() -> None:
    cfg = Config(kuramoto=True).validated()
    lat = OscillatorLattice.generate(40, config=cfg, seed=12)
    rows = np.arange(40)
    lat.coupling.apply_delta(rows, rows, np.ones(40, dtype=np.float32))
    before = lat.phase.copy()
    step_lattice(lat, 0, cfg)
    assert not np.allclose(before, lat.phase)
