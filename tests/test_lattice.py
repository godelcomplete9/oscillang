"""Unit tests for the struct-of-arrays OscillatorLattice."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from oscillang.core.banks import (
    FUNC_BANK,
    FUNC_HZ,
    Bank,
    Domain,
    Func,
)
from oscillang.core.config import Config
from oscillang.core.lattice import (
    MODES,
    MOVES,
    PURPOSES,
    STANCES,
    Oscillator,
    OscillatorLattice,
)

DOMAINS = tuple(Domain)
FUNCS = tuple(Func)


def test_generate_shapes_and_ranges() -> None:
    cfg = Config().validated()
    lat = OscillatorLattice.generate(100, config=cfg, seed=11)
    assert lat.n == 100
    assert lat.embeddings.shape == (100, cfg.embedding_dim)
    assert np.all(lat.hz >= 0.01) and np.all(lat.hz <= 1.5)
    assert np.all(lat.phase >= 0.0) and np.all(lat.phase < 2.0 * np.pi)
    assert np.all(lat.amplitude >= 0.0) and np.all(lat.amplitude <= 1.0)
    assert len(set(lat.ids.tolist())) == 100


def test_fatigue_is_one_minus_amplitude() -> None:
    lat = OscillatorLattice.generate(50, seed=5)
    assert np.allclose(lat.fatigue, 1.0 - lat.amplitude)


def test_snapshot_roundtrip_and_validation() -> None:
    lat = OscillatorLattice.generate(20, seed=6)
    snap = lat.to_oscillator(3)
    assert isinstance(snap, Oscillator)
    assert snap.domain in {d.value for d in DOMAINS}
    assert snap.func in {f.value for f in FUNCS}
    assert snap.mode in MODES
    assert snap.stance in STANCES
    assert snap.move in MOVES
    assert snap.fatigue == pytest.approx(1.0 - snap.amplitude)
    with pytest.raises(IndexError):
        lat.to_oscillator(20)
    with pytest.raises(ValidationError):
        Oscillator(**{**snap.model_dump(), "hz": 5.0})


def test_append_bulk_and_ids_continue() -> None:
    lat = OscillatorLattice.generate(10, seed=1)
    k = 5
    ids = lat.append(
        np.zeros(k, dtype=np.int16),
        np.zeros(k, dtype=np.int16),
        np.full(k, 0.5, dtype=np.float32),
        np.zeros(k, dtype=np.float32),
        np.full(k, 0.5, dtype=np.float32),
        np.ones((k, lat.embedding_dim), dtype=np.float32),
    )
    assert lat.n == 15
    assert ids[0] == "osc-10"
    snap = lat.to_oscillator(10)
    assert snap.hz == pytest.approx(0.5)


def test_append_clips_and_wraps() -> None:
    lat = OscillatorLattice(config=Config(embedding_dim=3).validated())
    lat.append(
        np.zeros(2, dtype=np.int16),
        np.zeros(2, dtype=np.int16),
        np.array([0.0, 9.0], dtype=np.float32),
        np.array([-1.0, 8.0], dtype=np.float32),
        np.array([-0.5, 2.0], dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )
    assert np.all(lat.hz >= 0.01) and np.all(lat.hz <= 1.5)
    assert np.all(lat.phase >= 0.0) and np.all(lat.phase < 2.0 * np.pi)
    assert np.all(lat.amplitude >= 0.0) and np.all(lat.amplitude <= 1.0)


def test_extend_composed_vectorized() -> None:
    cfg = Config().validated()
    lat = OscillatorLattice(config=cfg)
    rng = np.random.default_rng(2)
    lat.extend_composed(500, rng)
    assert lat.n == 500
    assert len(set(lat.ids.tolist())) == 500
    assert np.all(np.linalg.norm(lat.embeddings, axis=1) > 0.0)
    for f in FUNCS:
        assert FUNC_HZ[f] >= 0.01 and FUNC_HZ[f] <= 1.5


def test_coupling_resize_preserves_entries() -> None:
    lat = OscillatorLattice.generate(10, seed=3)
    coupling = lat.coupling
    rows = np.array([0, 1])
    cols = np.array([1, 2])
    coupling.apply_delta(rows, cols, np.array([0.5, 0.4], dtype=np.float32))
    assert lat.coupling.get(0, 1) == pytest.approx(0.1, abs=1e-6)
    lat.append(
        np.zeros(3, dtype=np.int16),
        np.zeros(3, dtype=np.int16),
        np.full(3, 0.3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.full(3, 0.5, dtype=np.float32),
        np.ones((3, lat.embedding_dim), dtype=np.float32),
    )
    assert lat.coupling.n == 13
    assert lat.coupling.get(0, 1) == pytest.approx(0.1, abs=1e-6)


def test_normalized_embeddings_unit_norm() -> None:
    lat = OscillatorLattice.generate(30, seed=4)
    normed = lat.normalized_embeddings
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-5)
    lat.append(
        np.zeros(2, dtype=np.int16),
        np.zeros(2, dtype=np.int16),
        np.full(2, 0.4, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.full(2, 0.5, dtype=np.float32),
        np.full((2, lat.embedding_dim), 3.0, dtype=np.float32),
    )
    assert np.allclose(np.linalg.norm(lat.normalized_embeddings, axis=1), 1.0, atol=1e-5)


def test_bank_of_internal_external() -> None:
    lat = OscillatorLattice.generate(60, seed=8)
    banks = {lat.bank_of(i) for i in range(lat.n)}
    assert banks <= {Bank.INTERNAL, Bank.EXTERNAL}
    assert FUNC_BANK[Func.WORLD_FACTS] is Bank.EXTERNAL
    assert FUNC_BANK[Func.EMOTION] is Bank.INTERNAL
    assert set(PURPOSES) == {"texture", "pressure", "memory", "counter", "bridge", "anchor"}
