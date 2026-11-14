"""Jev: the cheap decision layer that gates which oscillators fire.

Jev never renders prose - that is Main Brain's job, and only inside a fire
decision. Three implementations share one interface (``JevModel``):

- ``RuleJev``: the deterministic weighted sum (default; spec weights).
- ``LlmJev``: the cheap LLM scores a shortlist in ONE call per fire decision.
- ``MlpJev``: a tiny numpy perceptron; the ``learned_jev`` hook for training.

All of them take a ``JevContext`` (the per-tick feature vectors the runtime
assembled) and return one V score per oscillator.
"""

from oscillang.jev.base import JevContext, JevModel
from oscillang.jev.llm_jev import LlmJev
from oscillang.jev.mlp_jev import MlpJev
from oscillang.jev.rule_jev import RuleJev
from oscillang.jev.schema import JevDecision, JevScoreEntry, parse_jev_decision

__all__ = [
    "JevContext",
    "JevModel",
    "RuleJev",
    "LlmJev",
    "MlpJev",
    "JevDecision",
    "JevScoreEntry",
    "parse_jev_decision",
]
