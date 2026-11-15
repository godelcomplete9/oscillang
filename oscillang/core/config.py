"""Central configuration for oscillang.

Every default from the spec lives here so ablations can flip one flag at a
time. All randomness in the library flows through a seeded
``numpy.random.Generator``; noise is never zero (default 0.05, allowed range
(0.0, 0.2)).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Config"]

MIN_NOISE = 0.01
MAX_NOISE = 0.2


@dataclass
class Config:
    """All tunable dynamics constants. Defaults match the spec."""

    # --- integration -------------------------------------------------------
    step: float = 1.0

    # --- coupling ----------------------------------------------------------
    kuramoto: bool = False
    kuramoto_k: float = 1.0
    coupling_delta_limit: float = 0.1
    coupling_decay: float = 0.999
    coupling_saturation: float = 0.95
    # Coupling is stored as a dense (N, N) matrix below this size and as a
    # scipy.sparse matrix at or above it (20k dense would be 1.6 GB and too
    # slow to touch per tick).
    sparse_coupling_threshold: int = 20000

    # --- lock gating -------------------------------------------------------
    lock_tolerance: float = 0.4
    top_k: int = 12  # retrieval K, valid range 8-16

    # --- fatigue / refractory ----------------------------------------------
    decay_per_use: float = 0.08
    recovery_per_tick: float = 0.02
    refractory_fast: int = 2
    refractory_slow: int = 5
    fast_hz: float = 0.6
    baseline_amplitude: float = 0.5
    unused_drift_rate: float = 0.01

    # --- Jev scoring -------------------------------------------------------
    # The prefilter scores dynamics only; semantic alignment is the
    # reranker's verdict (scored 0-1 against ``v_threshold``).
    noise: float = 0.05
    w_phase: float = 0.8
    w_rec: float = 0.4
    w_fat: float = 0.7
    w_cost: float = 0.5
    w_coup: float = 0.4
    v_threshold: float = 0.65
    jev_shortlist: int = 24

    # --- input gains -------------------------------------------------------
    g_ext: float = 1.0

    # --- lattice -----------------------------------------------------------
    embedding_dim: int = 32

    # --- ablation switches (False = mechanism disabled) ---------------------
    # phase_gating off removes the phase-lock hard gate (refractory only).
    # world_model passes the previous beat to Jev as rerank context.
    # leaky_memory enables the recency boost for recently fired skills.
    phase_gating: bool = True
    coupling_plasticity: bool = True
    fatigue: bool = True
    refractory: bool = True
    noise_enabled: bool = True
    leaky_memory: bool = True
    world_model: bool = True
    learned_jev: bool = True

    # --- guards (runtime milestone) ----------------------------------------
    silence_watchdog_ticks: int = 10
    dead_coherence_ticks: int = 20
    no_fire_ticks: int = 10
    fire_every_tick_window: int = 20
    guard_vth_step: float = 0.05
    guard_noise_inject: float = 0.1

    # --- recency memory (leaky trace) --------------------------------------
    memory_alpha: float = 0.95
    memory_fade_ticks: int = 20
    memory_prune_size: int = 10000

    def validated(self) -> Config:
        """Return a copy with invariants enforced (noise clamped, K in 8-16)."""
        cfg = replace(self)
        if cfg.noise <= 0.0 or cfg.noise >= MAX_NOISE:
            floor = MIN_NOISE if cfg.noise <= 0.0 else cfg.noise
            cfg.noise = max(MIN_NOISE, min(floor, MAX_NOISE - 1e-6))
        cfg.top_k = max(8, min(16, int(cfg.top_k)))
        return cfg
