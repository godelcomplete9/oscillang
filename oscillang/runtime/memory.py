"""Leaky memory trace: a recency boost for recently fired mini skills.

The trace steers the Jev prefilter (``w_rec`` term). It fades with age: the
boost for a skill that fired ``age`` ticks ago is ``alpha**age``, cut to zero
after ``memory_fade_ticks``, so a beat that stops firing dissolves. No
embeddings: memory is a dynamics-side signal about what recently fired, not
a semantic vector.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RecencyTrace"]


class RecencyTrace:
    """Exponentially decaying per-oscillator last-fired registry."""

    def __init__(self, alpha: float = 0.95, fade_ticks: int = 20) -> None:
        self.alpha = float(alpha)
        self.fade_ticks = int(fade_ticks)
        self._last: dict[str, int] = {}

    @property
    def active(self) -> bool:
        return bool(self._last)

    def update(self, osc_ids: list[str], tick: int) -> None:
        """Record the oscillators that just fired at ``tick``."""
        for osc_id in osc_ids:
            self._last[str(osc_id)] = int(tick)

    def recency(self, ids: list[str], tick: int) -> np.ndarray:
        """(N,) recency boost in [0, 1] for each oscillator id."""
        out = np.zeros(len(ids), dtype=np.float32)
        for i, osc_id in enumerate(ids):
            last = self._last.get(str(osc_id))
            if last is None:
                continue
            age = int(tick) - last
            if 0 <= age <= self.fade_ticks:
                out[i] = np.float32(self.alpha**age)
        return out
