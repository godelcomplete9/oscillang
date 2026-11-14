"""Coupling matrix with dense and sparse backends.

Coupling is an (N, N) matrix in [0, 1]. Above the configured size threshold
(default 20,000) a scipy.sparse CSR backend is used. Decay is applied lazily
via a scalar factor so a per-tick decay is O(1) for the dense backend instead
of touching the full matrix; deltas materialize the factor only on the
entries they touch.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from oscillang.core.config import Config

__all__ = ["CouplingMatrix"]

FloatArray = np.ndarray


class CouplingMatrix:
    """(N, N) coupling weights in [0, 1], dense or sparse, lazily decayed."""

    def __init__(self, n: int, config: Config) -> None:
        self.n = int(n)
        self.config = config
        self._factor: float = 1.0
        if self._use_sparse(n, config):
            self._sparse: sparse.csr_matrix | None = sparse.csr_matrix((n, n), dtype=np.float32)
            self._dense: FloatArray | None = None
        else:
            self._sparse = None
            self._dense = np.zeros((n, n), dtype=np.float32)

    @staticmethod
    def _use_sparse(n: int, config: Config) -> bool:
        return n >= config.sparse_coupling_threshold

    @property
    def backend(self) -> str:
        return "sparse" if self._sparse is not None else "dense"

    # --- decay --------------------------------------------------------------
    def decay(self) -> None:
        """Apply one tick of exponential decay (x0.999) lazily."""
        self._factor *= float(self.config.coupling_decay)

    def materialize(self) -> None:
        """Fold the pending decay factor into the stored values."""
        if self._factor == 1.0:
            return
        if self._dense is not None:
            self._dense *= self._factor
        else:
            assert self._sparse is not None
            self._sparse.data *= self._factor
        self._factor = 1.0

    # --- reads --------------------------------------------------------------
    def row_sums(self) -> FloatArray:
        """Per-oscillator coupling aggregate, shape (N,)."""
        if self._dense is not None:
            return self._dense.sum(axis=1) * self._factor
        assert self._sparse is not None
        return np.asarray(self._sparse.sum(axis=1)).ravel() * self._factor

    def effective(self) -> FloatArray | sparse.csr_matrix:
        """Full matrix with decay folded in (used by the Kuramoto term)."""
        if self._dense is not None:
            return self._dense * self._factor
        assert self._sparse is not None
        out = self._sparse.copy()
        out.data = out.data * self._factor
        return out

    def get(self, row: int, col: int) -> float:
        """Single entry read with decay folded in."""
        if self._dense is not None:
            return float(self._dense[row, col]) * self._factor
        assert self._sparse is not None
        return float(self._sparse[row, col]) * self._factor

    # --- writes -------------------------------------------------------------
    def apply_delta(self, rows: np.ndarray, cols: np.ndarray, deltas: np.ndarray) -> None:
        """STDP-style weight update: ``w += clip(delta)`` on the given pairs.

        Deltas are clamped to [-limit, +limit]; results clamped to [0, 1].
        Vectorized; duplicate pairs accumulate.
        """
        if rows.size == 0:
            return
        limit = float(self.config.coupling_delta_limit)
        d = np.clip(np.asarray(deltas, dtype=np.float32), -limit, limit)
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        if self._dense is not None:
            cur = self._dense[r, c] * self._factor
            self._dense[r, c] = np.clip(cur, 0.0, 1.0)
            np.add.at(self._dense, (r, c), d)
            self._dense[r, c] = np.clip(self._dense[r, c], 0.0, 1.0)
        else:
            assert self._sparse is not None
            self.materialize()
            delta_mat = sparse.coo_matrix((d, (r, c)), shape=(self.n, self.n), dtype=np.float32)
            self._sparse = (self._sparse + delta_mat).tocsr()
            self._sparse.data = np.clip(self._sparse.data, 0.0, 1.0)

    def set_entries(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Direct assignment of coupling entries (initial ontology coupling).

        Bypasses the STDP delta limit; values are clamped to [0, 1]. Vectorized;
        pairs must be unique (they are when built from per-row top-K).
        """
        if rows.size == 0:
            return
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        if self._dense is not None:
            self._dense[r, c] = v * self._factor
        else:
            assert self._sparse is not None
            self.materialize()
            value_mat = sparse.coo_matrix((v, (r, c)), shape=(self.n, self.n), dtype=np.float32)
            self._sparse = (self._sparse + value_mat).tocsr()
            self._sparse.data = np.clip(self._sparse.data, 0.0, 1.0)

    def scale_global(self, factor: float) -> None:
        """Additional multiplicative decay (guard against saturation)."""
        self._factor *= float(factor)

    def dense_copy(self) -> FloatArray:
        """Dense (N, N) view with decay folded in. Only for small N / tests."""
        if self._dense is not None:
            return self._dense * self._factor
        assert self._sparse is not None
        return np.asarray(self._sparse.todense()) * self._factor
