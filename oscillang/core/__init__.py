"""Core oscillator dynamics: lattice, coupling, coherence, gating, dynamics."""

from oscillang.core.banks import Bank, Domain, Func
from oscillang.core.coherence import (
    coherence_full,
    coherence_pairs,
    coherence_query,
    compose_state_vector,
)
from oscillang.core.config import Config
from oscillang.core.coupling import CouplingMatrix
from oscillang.core.dynamics import (
    advance_phase,
    apply_use,
    drift_unused,
    kuramoto_term,
    recover_amplitude,
    step_lattice,
    update_refractory,
)
from oscillang.core.gating import (
    lock_gate,
    lock_gate_reference,
    phase_diff_matrix,
    retrieval_top_k,
    wrap_phase_diff,
)
from oscillang.core.lattice import Oscillator, OscillatorLattice

__all__ = [
    "Bank",
    "Config",
    "CouplingMatrix",
    "Domain",
    "Func",
    "Oscillator",
    "OscillatorLattice",
    "advance_phase",
    "apply_use",
    "coherence_full",
    "coherence_pairs",
    "coherence_query",
    "compose_state_vector",
    "drift_unused",
    "kuramoto_term",
    "lock_gate",
    "lock_gate_reference",
    "phase_diff_matrix",
    "recover_amplitude",
    "retrieval_top_k",
    "step_lattice",
    "update_refractory",
    "wrap_phase_diff",
]
