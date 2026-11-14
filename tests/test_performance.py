"""Performance tests for the vectorized core.

Budgets (spec): 20,000 oscillators / 100 ticks -> p50 tick < 150ms (no Main
Brain); 2,000 -> < 50ms; 200,000 composed -> < 400ms. Each tick exercises the
full core path: phase/amplitude/refractory dynamics, retrieval top-K,
candidate coherence, phase-diff lock gating, coupling row sums and decay.
"""

from __future__ import annotations

import time

import numpy as np

from oscillang.core.coherence import coherence_pairs
from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix
from oscillang.core.dynamics import step_lattice
from oscillang.core.gating import lock_gate, phase_diff_matrix, retrieval_top_k
from oscillang.core.lattice import OscillatorLattice


def core_tick(lat: OscillatorLattice, tick: int, cfg: Config, query: np.ndarray) -> int:
    step_lattice(lat, tick, cfg)
    scores = lat.normalized_embeddings @ query
    idx, _ = retrieval_top_k(scores, cfg.top_k)
    coh = coherence_pairs(lat.embeddings, idx)
    diff = phase_diff_matrix(lat.phase[idx])
    refractory = (lat.refractory_until_tick[idx] > tick)[:, None] | (
        lat.refractory_until_tick[idx] > tick
    )[None, :]
    gate = lock_gate(diff, coh, refractory, cfg.lock_tolerance, cfg.lock_threshold)
    lat.coupling.row_sums()
    lat.coupling.decay()
    return int(np.count_nonzero(gate))


def run_ticks(n: int, ticks: int, seed: int = 0, cfg: Config | None = None) -> list[float]:
    config = (cfg or Config()).validated()
    rng = np.random.default_rng(seed)
    lat = OscillatorLattice.generate(n, config=config, seed=seed)
    query = rng.normal(size=config.embedding_dim).astype(np.float32)
    core_tick(lat, -1, config, query)  # warm-up
    times: list[float] = []
    for tick in range(ticks):
        query += rng.normal(0.0, 0.05, size=config.embedding_dim).astype(np.float32)
        start = time.perf_counter()
        core_tick(lat, tick, config, query)
        times.append(time.perf_counter() - start)
    return times


def p50(times: list[float]) -> float:
    return float(np.percentile(np.asarray(times), 50)) * 1000.0


def test_coupling_backend_threshold() -> None:
    cfg = Config().validated()
    lat = OscillatorLattice.generate(100, config=cfg, seed=0)
    assert lat.coupling.backend == "dense"
    big = OscillatorLattice.generate(20001, config=cfg, seed=0)
    assert big.coupling.backend == "sparse"


def test_2000_oscillators_100_ticks_p50_under_50ms() -> None:
    times = run_ticks(2000, 100, seed=1)
    elapsed = p50(times)
    assert elapsed < 50.0, f"2k p50 tick {elapsed:.1f}ms >= 50ms"


def test_20000_oscillators_100_ticks_p50_under_150ms() -> None:
    times = run_ticks(20000, 100, seed=2)
    elapsed = p50(times)
    assert elapsed < 150.0, f"20k p50 tick {elapsed:.1f}ms >= 150ms"


def test_200000_composed_under_400ms() -> None:
    cfg = Config().validated()
    rng = np.random.default_rng(3)
    lat = OscillatorLattice(config=cfg)
    start = time.perf_counter()
    lat.extend_composed(200_000, rng)
    built = time.perf_counter() - start
    query = rng.normal(size=cfg.embedding_dim).astype(np.float32)
    start = time.perf_counter()
    gate_counts = [core_tick(lat, tick, cfg, query) for tick in range(3)]
    ticked = time.perf_counter() - start
    total = (built + ticked) * 1000.0
    assert lat.n == 200_000
    assert all(count >= 0 for count in gate_counts)
    assert total < 400.0, f"200k composed build+3 ticks {total:.1f}ms >= 400ms"


def test_sparse_coupling_row_sums_match_dense() -> None:
    cfg = Config(sparse_coupling_threshold=5).validated()
    dense = CouplingMatrix(4, cfg)
    sparse = CouplingMatrix(4, cfg)
    rows = np.array([0, 1, 2, 2])
    cols = np.array([1, 2, 0, 0])
    deltas = np.array([0.3, -0.2, 0.9, 0.1], dtype=np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    assert np.allclose(dense.row_sums(), sparse.row_sums(), atol=1e-6)
    for _ in range(3):
        dense.decay()
        sparse.decay()
    assert np.allclose(dense.row_sums(), sparse.row_sums(), atol=1e-6)
