"""RuleJev: the deterministic dynamics prefilter (the spec's default scorer).

Scores dynamics only - phase proximity, recency, amplitude, minus use cost
and coupling pressure, plus noise. It carries NO semantic signal: alignment
is verified by the reranker (``LlmJev``). RuleJev's job is to shortlist
which mini skills the reranker sees, and to serve as the numeric fallback in
dynamics-only mode.
"""

from __future__ import annotations

import numpy as np

from oscillang.core.config import Config
from oscillang.jev.base import JevContext

__all__ = ["RuleJev"]


class RuleJev:
    """V = w·dynamics - costs + noise, straight from the spec weights."""

    name = "rule-jev"

    def __init__(self, config: Config) -> None:
        self.config = config

    def score(self, ctx: JevContext) -> np.ndarray:
        cfg = self.config
        rec_term = cfg.w_rec * ctx.recency if cfg.leaky_memory else 0.0
        noise = ctx.noise if cfg.noise_enabled else 0.0
        v = (
            cfg.w_phase * ctx.phase_prox
            + rec_term
            + cfg.w_fat * ctx.amplitude
            - cfg.w_cost * ctx.used_cost
            - cfg.w_coup * ctx.coupling
            + noise
        )
        return v.astype(np.float32)
