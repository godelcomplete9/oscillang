"""Leaky memory trace: exponential moving average of fired oscillator embeddings.

The trace steers the Jev score (``w_mem`` term) and the query-aware state
vector (``g_sim`` gain). It fades toward zero by construction: every update
multiplies the old vector by ``alpha`` (0.95 default), so a beat that stops
firing dissolves after roughly ``memory_fade_ticks``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MemoryTrace"]


class MemoryTrace:
    """Unit-norm EMA vector over recently fired oscillator embeddings."""

    def __init__(self, dim: int, alpha: float = 0.95) -> None:
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.vec: np.ndarray = np.zeros(self.dim, dtype=np.float32)

    @property
    def active(self) -> bool:
        return bool(np.any(self.vec != 0.0))

    def update(self, embeddings: np.ndarray) -> None:
        """Blend toward the (normalized) mean of the just-fired embeddings."""
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.size == 0:
            return
        norms = np.maximum(np.linalg.norm(mat, axis=1), 1e-8)
        mean = (mat / norms[:, None]).mean(axis=0)
        mean_norm = max(float(np.linalg.norm(mean)), 1e-8)
        mean = (mean / mean_norm).astype(np.float32)
        self.vec = (self.alpha * self.vec + (1.0 - self.alpha) * mean).astype(np.float32)
        norm = max(float(np.linalg.norm(self.vec)), 1e-8)
        self.vec = (self.vec / norm).astype(np.float32)

    def similarity(self, normalized_embeddings: np.ndarray) -> np.ndarray:
        """(N,) cosine of every oscillator embedding against the trace."""
        if not self.active:
            return np.zeros(normalized_embeddings.shape[0], dtype=np.float32)
        return (normalized_embeddings @ self.vec).astype(np.float32)
