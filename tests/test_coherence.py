"""Unit tests for coherence and the Config invariants."""

from __future__ import annotations

import numpy as np
import pytest

from oscillang.core.coherence import (
    coherence_full,
    coherence_pairs,
    coherence_query,
    compose_state_vector,
)
from oscillang.core.config import Config


def test_coherence_full_identity_and_orthogonality() -> None:
    emb = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071, 0.7071, 0.0]],
        dtype=np.float32,
    )
    coh = coherence_full(emb)
    assert np.allclose(np.diag(coh), 1.0, atol=1e-5)
    assert coh[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert coh[0, 2] == pytest.approx(2**-0.5, abs=1e-3)
    assert np.allclose(coh, coh.T, atol=1e-6)


def test_coherence_pairs_subset() -> None:
    emb = np.eye(5, dtype=np.float32)
    idx = np.array([1, 3])
    coh = coherence_pairs(emb, idx)
    assert coh.shape == (2, 2)
    assert coh[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert coh[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_coherence_query_ranking() -> None:
    emb = np.eye(4, dtype=np.float32)
    query = np.array([0.1, 1.0, 0.0, 0.0], dtype=np.float32)
    scores = coherence_query(emb, query)
    assert scores[1] == pytest.approx(1.0, abs=0.01)
    assert scores[0] < scores[1]
    assert scores[2] == pytest.approx(0.0, abs=1e-6)


def test_compose_state_vector_normalized() -> None:
    a = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    state = compose_state_vector([(a, 1.0), (b, 1.0)], embedding_dim=3)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-5)
    assert state[0] == pytest.approx(2**-0.5, abs=1e-5)


def test_compose_state_vector_weighted() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    state = compose_state_vector([(a, 3.0), (b, 1.0)], embedding_dim=2)
    assert state[0] > state[1]


def test_compose_state_vector_empty() -> None:
    state = compose_state_vector([], embedding_dim=4)
    assert state.shape == (4,)
    assert np.all(state == 0.0)


def test_config_noise_never_zero() -> None:
    assert Config(noise=0.0).validated().noise == pytest.approx(0.01, abs=1e-9)
    assert Config(noise=0.05).validated().noise == pytest.approx(0.05)
    assert Config(noise=0.5).validated().noise < 0.2


def test_config_top_k_range() -> None:
    assert Config(top_k=2).validated().top_k == 8
    assert Config(top_k=99).validated().top_k == 16
    assert Config(top_k=12).validated().top_k == 12
