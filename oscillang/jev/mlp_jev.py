"""MlpJev: tiny numpy perceptron - the learned dynamics prefilter.

Two layers, tanh hidden, no torch dependency, fully deterministic from the
seed. It scores dynamics only (no semantic signal - alignment is the
reranker's job). ``learn`` does one SGD step toward per-oscillator targets
(reward signal lands with the eval milestone). Output is clipped to [0, 2]
so a wild network can never dominate the rule scorer's range.
"""

from __future__ import annotations

import numpy as np

from oscillang.jev.base import JevContext

__all__ = ["MlpJev"]

FEATURES = 5


def _feature_matrix(ctx: JevContext) -> np.ndarray:
    """(5, N): phase prox, recency, amplitude, -cost, -coupling."""
    return np.stack(
        [
            ctx.phase_prox,
            ctx.recency,
            ctx.amplitude,
            -ctx.used_cost,
            -ctx.coupling,
        ]
    ).astype(np.float32)


class MlpJev:
    name = "mlp-jev"

    def __init__(self, hidden: int = 8, seed: int = 0, scale: float = 0.5) -> None:
        rng = np.random.default_rng(seed)
        self.w1: np.ndarray = rng.normal(0.0, scale, (FEATURES, hidden)).astype(np.float32)
        self.b1: np.ndarray = np.zeros(hidden, dtype=np.float32)
        self.w2: np.ndarray = rng.normal(0.0, scale, (hidden,)).astype(np.float32)
        self.b2: float = 0.0
        self.rng = np.random.default_rng(seed + 1)

    def _hidden(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features.T @ self.w1 + self.b1).astype(np.float32)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self._hidden(features)
        v = (h @ self.w2 + self.b2).astype(np.float32)
        return v, h

    def score(self, ctx: JevContext) -> np.ndarray:
        features = _feature_matrix(ctx)
        v, _ = self._forward(features)
        return np.clip(v + ctx.noise, 0.0, 2.0).astype(np.float32)

    def learn(self, ctx: JevContext, targets: np.ndarray, lr: float = 0.05) -> float:
        """One SGD step on the output layer toward ``targets``; returns MSE."""
        features = _feature_matrix(ctx)
        v, h = self._forward(features)
        error = v - np.asarray(targets, dtype=np.float32)
        mse = float(np.mean(error**2))
        grad_w2 = (h.T @ error) / max(error.shape[0], 1)
        self.w2 = (self.w2 - lr * grad_w2).astype(np.float32)
        self.b2 = float(self.b2 - lr * float(np.mean(error)))
        return mse
