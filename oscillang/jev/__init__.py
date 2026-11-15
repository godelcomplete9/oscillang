"""Jev: the decision layer that gates which mini skills fire.

Jev verifies semantic coherence and alignment between the input and the
candidate behavioral fragments; it is a reranker and the main gatekeeper of
a mini skill. It never renders prose - that is Main Brain's job, and only
inside a fire decision. The oscillator layer contributes pure dynamics as a
hard gate on top; no embedding cosine anywhere.

Two roles:

- ``JevModel`` (``score`` over ``JevContext``): numeric dynamics scorer.
  ``RuleJev`` is the deterministic prefilter; ``MlpJev`` is the learned hook.
- ``JevReranker`` (``rerank`` over ``JevCandidate`` texts): the semantic
  verifier. ``LlmJev`` verifies alignment in ONE call per fire decision.
"""

from oscillang.jev.base import (
    JevCandidate,
    JevContext,
    JevModel,
    JevReranker,
    JevVerdict,
)
from oscillang.jev.llm_jev import LlmJev
from oscillang.jev.mlp_jev import MlpJev
from oscillang.jev.rule_jev import RuleJev
from oscillang.jev.schema import JevDecision, JevScoreEntry, parse_jev_decision

__all__ = [
    "JevContext",
    "JevCandidate",
    "JevVerdict",
    "JevModel",
    "JevReranker",
    "RuleJev",
    "LlmJev",
    "MlpJev",
    "JevDecision",
    "JevScoreEntry",
    "parse_jev_decision",
]
