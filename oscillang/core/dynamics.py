"""Vectorized core dynamics: phase, Kuramoto term, fatigue, refractory.

Every function operates on whole arrays; there are no Python loops over
oscillators anywhere in core (enforced by tests/test_static_no_loops.py).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix
from oscillang.core.lattice import OscillatorLattice

__all__ = [
    "advance_phase",
    "kuramoto_term",
    "apply_use",
    "recover_amplitude",
    "drift_unused",
    "update_refractory",
    "step_lattice",
]

TWO_PI = 2.0 * float(np.pi)


def advance_phase(phase: np.ndarray, hz: np.ndarray, step: float) -> np.ndarray:
    """phase = (phase + 2*pi*hz*step) mod 2*pi, fully vectorized."""
    return np.mod(phase + TWO_PI * hz * step, TWO_PI).astype(np.float32)


def kuramoto_term(
    phase: np.ndarray,
    coupling: np.ndarray | sparse.csr_matrix | CouplingMatrix,
    k: float,
) -> np.ndarray:
    """K * sum_j C_ij * sin(phase_j - phase_i) for every i at once.

    Uses the identity sin(b - a) = sin(b)cos(a) - cos(b)sin(a) so the term is
    two sparse/dense matvecs instead of an (N, N) trig matrix.
    """
    if isinstance(coupling, CouplingMatrix):
        mat = coupling.effective()
    else:
        mat = coupling
    sin_p = np.sin(phase)
    cos_p = np.cos(phase)
    if sparse.issparse(mat):
        c_sin = np.asarray(mat @ sin_p).ravel()
        c_cos = np.asarray(mat @ cos_p).ravel()
    else:
        c_sin = mat @ sin_p
        c_cos = mat @ cos_p
    return (k * (cos_p * c_sin - sin_p * c_cos)).astype(np.float32)


def apply_use(
    amplitude: np.ndarray,
    fire_mask: np.ndarray,
    decay_per_use: float | np.ndarray,
    recovery_per_tick: float,
) -> np.ndarray:
    """On use: amp = amp*decayPerUse + recoveryPerTick; clamped to [0, 1].

    ``decay_per_use`` may be a per-oscillator array (the lattice stores a
    decayRate per row).
    """
    updated = np.where(
        fire_mask,
        amplitude * decay_per_use + recovery_per_tick,
        amplitude,
    )
    return np.clip(updated, 0.0, 1.0).astype(np.float32)


def recover_amplitude(
    amplitude: np.ndarray,
    fire_mask: np.ndarray,
    recovery_per_tick: float,
) -> np.ndarray:
    """Unused oscillators recover: amp += recoveryPerTick, clamped to [0, 1]."""
    updated = np.where(fire_mask, amplitude, amplitude + recovery_per_tick)
    return np.clip(updated, 0.0, 1.0).astype(np.float32)


def drift_unused(
    amplitude: np.ndarray,
    fire_mask: np.ndarray,
    baseline: float,
    drift_rate: float,
) -> np.ndarray:
    """Unused oscillators drift toward the baseline amplitude (default 0.5)."""
    target = np.where(fire_mask, amplitude, baseline)
    updated = np.where(fire_mask, amplitude, amplitude + (target - amplitude) * drift_rate)
    return np.clip(updated, 0.0, 1.0).astype(np.float32)


def update_refractory(
    refractory_until: np.ndarray,
    fire_mask: np.ndarray,
    tick: int,
    hz: np.ndarray,
    fast_ticks: int,
    slow_ticks: int,
    fast_hz: float,
) -> np.ndarray:
    """Fired oscillators go refractory: 2 ticks fast (hz >= 0.6), 5 ticks slow."""
    duration = np.where(hz >= fast_hz, fast_ticks, slow_ticks).astype(np.int32)
    until = np.where(fire_mask, tick + duration, refractory_until)
    return until.astype(np.int32)


def step_lattice(
    lattice: OscillatorLattice,
    tick: int,
    config: Config,
    fire_mask: np.ndarray | None = None,
    apply_coupling_decay: bool = True,
) -> np.ndarray:
    """One vectorized lattice update: phase, amplitude/fatigue, refractory.

    Returns the fire mask actually applied (all False if none given). Kuramoto
    coupling is applied only when ``config.kuramoto`` is set.
    """
    n = lattice.n
    if fire_mask is None:
        fire_mask = np.zeros(n, dtype=bool)
    else:
        fire_mask = np.asarray(fire_mask, dtype=bool)

    phase = lattice.phase[:n]
    if config.kuramoto and n > 0:
        phase = phase + kuramoto_term(phase, lattice.coupling, config.kuramoto_k) * config.step
    lattice.phase[:n] = advance_phase(phase, lattice.hz[:n], config.step)

    if config.fatigue:
        used = apply_use(
            lattice.amplitude[:n],
            fire_mask,
            lattice.decay_rate[:n],
            config.recovery_per_tick,
        )
        recovered = recover_amplitude(used, fire_mask, config.recovery_per_tick)
        lattice.amplitude[:n] = drift_unused(
            recovered, fire_mask, config.baseline_amplitude, config.unused_drift_rate
        )
    else:
        lattice.amplitude[:n] = np.clip(
            lattice.amplitude[:n] + config.recovery_per_tick, 0.0, 1.0
        ).astype(np.float32)

    if config.refractory:
        lattice.refractory_until_tick[:n] = update_refractory(
            lattice.refractory_until_tick[:n],
            fire_mask,
            tick,
            lattice.hz[:n],
            config.refractory_fast,
            config.refractory_slow,
            config.fast_hz,
        )

    if apply_coupling_decay and config.coupling_plasticity and n > 0:
        lattice.coupling.decay()
    return fire_mask
