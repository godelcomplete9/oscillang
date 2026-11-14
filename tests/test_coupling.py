"""Unit tests for the coupling matrix (dense and sparse backends)."""

from __future__ import annotations

import numpy as np
import pytest

from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix


def make_pair(n: int = 8) -> tuple[Config, CouplingMatrix, CouplingMatrix]:
    dense_cfg = Config(sparse_coupling_threshold=n + 1).validated()
    sparse_cfg = Config(sparse_coupling_threshold=1).validated()
    return dense_cfg, CouplingMatrix(n, dense_cfg), CouplingMatrix(n, sparse_cfg)


def test_backends_selected_by_threshold() -> None:
    _, dense, sparse = make_pair()
    assert dense.backend == "dense"
    assert sparse.backend == "sparse"


def test_delta_clamped_to_limit_and_unit_range() -> None:
    _, dense, sparse = make_pair()
    rows = np.array([0, 1])
    cols = np.array([1, 2])
    deltas = np.array([5.0, -5.0], dtype=np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(0.1, abs=1e-6)
        assert m.get(1, 2) == pytest.approx(0.0, abs=1e-6)


def test_clamp_high_and_low() -> None:
    _, dense, sparse = make_pair()
    rows = np.array([0, 0, 1, 1])
    cols = np.array([1, 1, 2, 2])
    deltas = np.array([1.0, 1.0, -0.5, -1.0], dtype=np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(0.2, abs=1e-6)
        assert m.get(1, 2) == pytest.approx(0.0, abs=1e-6)
    assert dense.get(1, 2) >= 0.0 and sparse.get(1, 2) >= 0.0


def test_duplicate_pairs_accumulate() -> None:
    _, dense, sparse = make_pair()
    rows = np.array([0, 0])
    cols = np.array([1, 1])
    deltas = np.array([0.06, 0.06], dtype=np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(0.12, abs=1e-6)


def test_lazy_decay_matches_materialized() -> None:
    _, dense, sparse = make_pair()
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    deltas = np.array([0.6, 0.4], dtype=np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    rate = 0.999
    for m in (dense, sparse):
        m.decay()
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(0.1 * rate, abs=1e-6)
    for _ in range(5):
        dense.decay()
        sparse.decay()
    dense.materialize()
    sparse.materialize()
    expected = 0.1 * rate**6
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(expected, abs=1e-6)
        assert m.get(2, 3) == pytest.approx(expected, abs=1e-6)


def test_row_sums_dense_sparse_agree() -> None:
    _, dense, sparse = make_pair()
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 8, size=30)
    cols = rng.integers(0, 8, size=30)
    deltas = rng.uniform(-0.1, 0.1, size=30).astype(np.float32)
    dense.apply_delta(rows, cols, deltas)
    sparse.apply_delta(rows, cols, deltas)
    assert np.allclose(dense.row_sums(), sparse.row_sums(), atol=1e-5)


def test_scale_global_guard() -> None:
    _, dense, sparse = make_pair()
    rows = np.array([0])
    cols = np.array([1])
    dense.apply_delta(rows, cols, np.array([1.0], dtype=np.float32))
    sparse.apply_delta(rows, cols, np.array([1.0], dtype=np.float32))
    dense.scale_global(0.99)
    sparse.scale_global(0.99)
    for m in (dense, sparse):
        assert m.get(0, 1) == pytest.approx(0.1 * 0.99, abs=1e-6)


def test_empty_delta_is_noop() -> None:
    _, dense, _ = make_pair()
    dense.apply_delta(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    )
    assert np.allclose(dense.row_sums(), 0.0)


def test_effective_full_matrix() -> None:
    _, dense, _ = make_pair()
    dense.apply_delta(np.array([0]), np.array([1]), np.array([0.5], dtype=np.float32))
    eff = dense.effective()
    assert eff[0, 1] == pytest.approx(0.1, abs=1e-6)
