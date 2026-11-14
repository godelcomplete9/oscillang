"""RuleJev: the deterministic weighted sum (the spec's default scorer)."""

from __future__ import annotations

import numpy as np

from oscillang.core.config import Config
from oscillang.jev.base import JevContext

__all__ = ["RuleJev"]


class RuleJev:
    """V = w·features - costs + noise, straight from the spec weights."""

    name = "rule-jev"

    def __init__(self, config: Config) -> None:
        self.config = config

    def score(self, ctx: JevContext) -> np.ndarray:
        cfg = self.config
        mem_term = cfg.w_mem * ctx.memory_sim if cfg.leaky_memory else 0.0
        noise = ctx.noise if cfg.noise_enabled else 0.0
        v = (
            cfg.w_coh * ctx.coherence
            + cfg.w_phase * ctx.phase_prox
            + mem_term
            + cfg.w_fat * ctx.amplitude
            - cfg.w_cost * ctx.used_cost
            - cfg.w_coup * ctx.coupling
            + noise
        )
        return v.astype(np.float32)
