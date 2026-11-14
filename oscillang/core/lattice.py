"""Oscillator lattice: struct-of-arrays storage, never a list of objects.

All per-oscillator fields live in flat numpy arrays so every tick operation is
a single vectorized call. A pydantic ``Oscillator`` model exists only as a
snapshot/serialization view of a single row.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy import sparse

from oscillang.core.banks import (
    DOMAINS,
    FUNC_BANK,
    FUNC_HZ,
    FUNCS,
    Bank,
    Func,
)
from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix

__all__ = ["Oscillator", "OscillatorLattice", "MODES", "STANCES", "MOVES", "PURPOSES"]

MODES: tuple[str, ...] = ("neutral", "active", "suppressed", "amplified")
STANCES: tuple[str, ...] = ("toward", "away", "against", "with")
MOVES: tuple[str, ...] = ("advance", "retreat", "circle", "hold")
PURPOSES: tuple[str, ...] = ("texture", "pressure", "memory", "counter", "bridge", "anchor")


class Oscillator(BaseModel):
    """Snapshot view of one oscillator row in the lattice."""

    id: str
    domain: str
    func: str
    mode: str
    stance: str
    move: str
    hz: float = Field(ge=0.01, le=1.5)
    phase: float = Field(ge=0.0, lt=2.0 * float(np.pi))
    amplitude: float = Field(ge=0.0, le=1.0)
    templateId: str | None = None
    lastFiredTick: int = -1
    refractoryUntilTick: int = -1
    embedding: list[float]
    fatigue: float = Field(ge=0.0, le=1.0)
    decayRate: float = Field(gt=0.0, le=1.0)


class OscillatorLattice:
    """Struct-of-arrays oscillator store plus its coupling matrix."""

    def __init__(self, config: Config | None = None, embedding_dim: int | None = None) -> None:
        self.config = config or Config()
        self.embedding_dim = embedding_dim or self.config.embedding_dim
        self._n = 0
        self._capacity = 0
        self.ids: np.ndarray = np.empty(0, dtype=object)
        self.domain: np.ndarray = np.empty(0, dtype=np.int16)
        self.func: np.ndarray = np.empty(0, dtype=np.int16)
        self.mode: np.ndarray = np.empty(0, dtype=np.int16)
        self.stance: np.ndarray = np.empty(0, dtype=np.int16)
        self.move: np.ndarray = np.empty(0, dtype=np.int16)
        self.hz: np.ndarray = np.empty(0, dtype=np.float32)
        self.phase: np.ndarray = np.empty(0, dtype=np.float32)
        self.amplitude: np.ndarray = np.empty(0, dtype=np.float32)
        self.template_id: np.ndarray = np.empty(0, dtype=object)
        self.last_fired_tick: np.ndarray = np.empty(0, dtype=np.int32)
        self.refractory_until_tick: np.ndarray = np.empty(0, dtype=np.int32)
        self.decay_rate: np.ndarray = np.empty(0, dtype=np.float32)
        self.embeddings: np.ndarray = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._normed: np.ndarray | None = None
        self._coupling: CouplingMatrix | None = None

    # --- basic properties ---------------------------------------------------
    @property
    def n(self) -> int:
        return self._n

    @property
    def fatigue(self) -> np.ndarray:
        """fatigue = 1 - amplitude, computed on demand, never stored."""
        return 1.0 - self.amplitude

    @property
    def coupling(self) -> CouplingMatrix:
        if self._coupling is None or self._coupling.n != self._n:
            self._coupling = self._build_coupling()
        return self._coupling

    def _build_coupling(self) -> CouplingMatrix:
        old = self._coupling
        new = CouplingMatrix(self._n, self.config)
        if old is not None and old.n > 0 and self._n >= old.n:
            _grow_coupling(old, new)
        return new

    # --- embeddings ---------------------------------------------------------
    @property
    def normalized_embeddings(self) -> np.ndarray:
        if self._normed is None or self._normed.shape[0] != self._n:
            self._normed = l2_normalize_rows(self.embeddings[: self._n])
        return self._normed

    def invalidate_embeddings(self) -> None:
        self._normed = None

    # --- construction -------------------------------------------------------
    def reserve(self, min_capacity: int) -> None:
        if min_capacity <= self._capacity:
            return
        new_cap = max(min_capacity, self._capacity * 2)
        self.ids = _grow_object(self.ids, new_cap, None)
        self.template_id = _grow_object(self.template_id, new_cap, None)
        typed_names = (
            "domain",
            "func",
            "mode",
            "stance",
            "move",
            "hz",
            "phase",
            "amplitude",
            "decay_rate",
            "last_fired_tick",
            "refractory_until_tick",
        )
        self.__dict__.update(
            {
                name: _grow_typed(getattr(self, name), new_cap, getattr(self, name).dtype)
                for name in typed_names
            }
        )
        new_emb = np.zeros((new_cap, self.embedding_dim), dtype=np.float32)
        new_emb[: self._n] = self.embeddings
        self.embeddings = new_emb
        self._capacity = new_cap
        self._normed = None

    def append(
        self,
        domain: np.ndarray,
        func: np.ndarray,
        hz: np.ndarray,
        phase: np.ndarray,
        amplitude: np.ndarray,
        embeddings: np.ndarray,
        mode: np.ndarray | None = None,
        stance: np.ndarray | None = None,
        move: np.ndarray | None = None,
        template_ids: np.ndarray | None = None,
        decay_rate: np.ndarray | None = None,
        last_fired_tick: np.ndarray | None = None,
        refractory_until_tick: np.ndarray | None = None,
    ) -> np.ndarray:
        """Bulk-append rows; all inputs are arrays of length k. Returns new ids."""
        k = int(hz.shape[0])
        if k == 0:
            return np.empty(0, dtype=object)
        start = self._n
        self.reserve(start + k)
        end = start + k
        self.domain[start:end] = domain
        self.func[start:end] = func
        self.mode[start:end] = mode if mode is not None else 0
        self.stance[start:end] = stance if stance is not None else 0
        self.move[start:end] = move if move is not None else 0
        self.hz[start:end] = np.clip(hz, 0.01, 1.5)
        self.phase[start:end] = np.mod(phase, 2.0 * np.pi)
        self.amplitude[start:end] = np.clip(amplitude, 0.0, 1.0)
        if decay_rate is not None:
            self.decay_rate[start:end] = decay_rate
        else:
            self.decay_rate[start:end] = self.config.decay_per_use
        if last_fired_tick is not None:
            self.last_fired_tick[start:end] = last_fired_tick
        else:
            self.last_fired_tick[start:end] = -1
        if refractory_until_tick is not None:
            self.refractory_until_tick[start:end] = refractory_until_tick
        else:
            self.refractory_until_tick[start:end] = -1
        if template_ids is not None:
            self.template_id[start:end] = template_ids
        self.embeddings[start:end] = embeddings
        self.ids[start:end] = _make_ids(start, k)
        self._n = end
        self._normed = None
        if self._coupling is not None:
            self._coupling = self._build_coupling()
        return self.ids[start:end].copy()

    # --- generators ---------------------------------------------------------
    @classmethod
    def generate(cls, n: int, config: Config | None = None, seed: int = 0) -> OscillatorLattice:
        """Vectorized random lattice of n oscillators (reproducible via seed)."""
        cfg = (config or Config()).validated()
        rng = np.random.default_rng(seed)
        lat = cls(config=cfg)
        lat.reserve(n)
        domain = rng.integers(0, len(DOMAINS), size=n).astype(np.int16)
        func = rng.integers(0, len(FUNCS), size=n).astype(np.int16)
        hz = rng.uniform(0.01, 1.5, size=n).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        amp = rng.uniform(0.1, 0.9, size=n).astype(np.float32)
        decay = rng.uniform(0.05, 0.12, size=n).astype(np.float32)
        emb = rng.normal(0.0, 1.0, size=(n, cfg.embedding_dim)).astype(np.float32)
        lat.append(domain, func, hz, phase, amp, emb, decay_rate=decay)
        return lat

    def extend_composed(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Append ``count`` on-demand composed oscillators.

        V = V_domain + V_func + V_mode + V_stance + V_move + V_purpose, with
        component lookup tables drawn from the rng. Fully vectorized.
        """
        if count <= 0:
            return np.empty(0, dtype=object)
        d = self.embedding_dim
        n_dom, n_fun = len(DOMAINS), len(FUNCS)
        n_mod, n_sta, n_mov, n_pur = len(MODES), len(STANCES), len(MOVES), len(PURPOSES)
        v_domain = _unit_rows(rng, n_dom, d)
        v_func = _unit_rows(rng, n_fun, d)
        v_mode = _unit_rows(rng, n_mod, d)
        v_stance = _unit_rows(rng, n_sta, d)
        v_move = _unit_rows(rng, n_mov, d)
        v_purpose = _unit_rows(rng, n_pur, d)
        dc = rng.integers(0, n_dom, size=count).astype(np.int16)
        fc = rng.integers(0, n_fun, size=count).astype(np.int16)
        mc = rng.integers(0, n_mod, size=count).astype(np.int16)
        sc = rng.integers(0, n_sta, size=count).astype(np.int16)
        mvc = rng.integers(0, n_mov, size=count).astype(np.int16)
        pc = rng.integers(0, n_pur, size=count).astype(np.int16)
        emb = (
            v_domain[dc] + v_func[fc] + v_mode[mc] + v_stance[sc] + v_move[mvc] + v_purpose[pc]
        ).astype(np.float32)
        base_hz = np.array([FUNC_HZ[f] for f in Func], dtype=np.float32)
        hz = np.clip(base_hz[fc] + rng.normal(0.0, 0.05, size=count), 0.01, 1.5).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
        amp = np.clip(
            self.config.baseline_amplitude + rng.normal(0.0, 0.1, size=count), 0.0, 1.0
        ).astype(np.float32)
        return self.append(dc, fc, hz, phase, amp, emb, mode=mc, stance=sc, move=mvc)

    # --- snapshot -----------------------------------------------------------
    def to_oscillator(self, i: int) -> Oscillator:
        if i < 0 or i >= self._n:
            raise IndexError(f"oscillator index {i} out of range (n={self._n})")
        return Oscillator(
            id=str(self.ids[i]),
            domain=DOMAINS[int(self.domain[i])].value,
            func=FUNCS[int(self.func[i])].value,
            mode=MODES[int(self.mode[i])],
            stance=STANCES[int(self.stance[i])],
            move=MOVES[int(self.move[i])],
            hz=float(self.hz[i]),
            phase=float(self.phase[i]),
            amplitude=float(self.amplitude[i]),
            templateId=self.template_id[i],
            lastFiredTick=int(self.last_fired_tick[i]),
            refractoryUntilTick=int(self.refractory_until_tick[i]),
            embedding=[float(x) for x in self.embeddings[i]],
            fatigue=float(1.0 - self.amplitude[i]),
            decayRate=float(self.decay_rate[i]),
        )

    def bank_of(self, i: int) -> Bank:
        return FUNC_BANK[FUNCS[int(self.func[i])]]


# --- helpers -----------------------------------------------------------------
def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps).astype(mat.dtype)


def _unit_rows(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    mat = rng.normal(0.0, 1.0, size=(count, dim)).astype(np.float32)
    return l2_normalize_rows(mat)


def _make_ids(start: int, k: int) -> np.ndarray:
    return np.char.add("osc-", np.arange(start, start + k).astype(np.str_))


def _grow_object(arr: np.ndarray, new_cap: int, fill: object) -> np.ndarray:
    out = np.full(new_cap, fill, dtype=object)
    out[: arr.shape[0]] = arr
    return out


def _grow_typed(arr: np.ndarray, new_cap: int, dtype: np.dtype) -> np.ndarray:
    out = np.zeros(new_cap, dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _grow_coupling(old: CouplingMatrix, new: CouplingMatrix) -> None:
    """Copy old coupling entries into a larger matrix (same-backend fast path)."""
    if old.backend == "dense":
        old_dense = old.dense_copy()
        k = old.n
        if new.backend == "dense":
            assert new._dense is not None
            new._dense[:k, :k] = old_dense
        else:
            assert new._sparse is not None
            new._sparse = sparse.csr_matrix(old_dense)
    else:
        old.materialize()
        assert old._sparse is not None
        if new.backend == "sparse":
            assert new._sparse is not None
            new._sparse = old._sparse if old.n == new.n else _pad_sparse(old._sparse, new.n)
        else:
            assert new._dense is not None
            new._dense[: old.n, : old.n] = np.asarray(old._sparse.todense())
    new._factor = 1.0


def _pad_sparse(mat: sparse.csr_matrix, n: int) -> sparse.csr_matrix:
    out = sparse.csr_matrix((n, n), dtype=mat.dtype)
    k = mat.shape[0]
    out[:k, :k] = mat
    return out
