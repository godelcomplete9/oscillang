"""Coherence: V @ V.T on L2-normalized embeddings, plus query-aware variants.

Full (N, N) coherence is opt-in (``coherence_full``) because it is quadratic;
per-tick code uses either the cached full matrix, small candidate slices, or
the O(N*d) query projection.
"""

from __future__ import annotations

import numpy as np

from oscillang.core.lattice import l2_normalize_rows

__all__ = [
    "coherence_full",
    "coherence_pairs",
    "coherence_query",
    "compose_state_vector",
]


def coherence_full(embeddings: np.ndarray) -> np.ndarray:
    """(N, N) cosine-similarity matrix from L2-normalized embeddings."""
    return (l2_normalize_rows(embeddings) @ l2_normalize_rows(embeddings).T).astype(np.float32)


def coherence_pairs(embeddings: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(k, k) coherence among a candidate subset; batched and small."""
    sub = l2_normalize_rows(embeddings[indices])
    return (sub @ sub.T).astype(np.float32)


def coherence_query(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    """(N,) cosine similarity of every oscillator to a query/state vector."""
    q = np.asarray(query, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    return (l2_normalize_rows(embeddings) @ q).astype(np.float32)


def compose_state_vector(
    components: list[tuple[np.ndarray, float]],
    embedding_dim: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Query-aware state vector: weighted sum of L2-normalized components.

    Generic helper: components are e.g. an input encoding, a scene objective,
    and recent thoughts. The result is L2-normalized.
    """
    if not components:
        return np.zeros(embedding_dim, dtype=np.float32)
    mat = np.stack([np.asarray(vec, dtype=np.float32) for vec, _ in components])
    weights = np.asarray([weight for _, weight in components], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(mat, axis=1), eps)
    state = ((weights / norms) @ mat).astype(np.float32)
    return (state / max(float(np.linalg.norm(state)), eps)).astype(np.float32)
