"""Runtime guards: the watchdogs that keep the loop from locking up.

Pure logic, no lattice access: ``GuardState.update`` takes the outcomes of
one tick (did anything fire, how strong was the best alignment verdict, did
Main Brain answer) and returns guard events. The engine reacts to events:

- ``no_fire_threshold_drop``: nothing fired for ``no_fire_ticks`` -> Jev
  threshold drops by ``guard_vth_step`` (floored at 0.05).
- ``overfire_threshold_raise``: something fired every tick for
  ``fire_every_tick_window`` -> threshold rises by the same step (capped 0.99).
- ``dead_coherence_noise_inject``: the best alignment verdict stayed under
  0.05 for ``dead_coherence_ticks`` -> engine injects ``guard_noise_inject``
  radians of phase drive next tick.
- ``silence_watchdog``: Main Brain returned nothing for
  ``silence_watchdog_ticks`` -> engine forces a re-render with a nudge next
  turn.
"""

from __future__ import annotations

from oscillang.core.config import Config

__all__ = ["GuardState", "DEAD_COHERENCE_FLOOR", "VTH_FLOOR", "VTH_CEIL"]

DEAD_COHERENCE_FLOOR = 0.05
VTH_FLOOR = 0.05
VTH_CEIL = 0.99


class GuardState:
    """Adaptive Jev threshold plus deadlock counters."""

    def __init__(self, config: Config, v_threshold: float | None = None) -> None:
        self.config = config
        self.v_threshold = float(v_threshold if v_threshold is not None else config.v_threshold)
        self._no_fire = 0
        self._overfire = 0
        self._dead_coh = 0
        self._silent = 0

    def update(self, fired: bool, max_verdict: float, responded: bool) -> list[str]:
        """Feed one tick's outcome; returns guard events raised this tick."""
        cfg = self.config
        events: list[str] = []

        self._no_fire = 0 if fired else self._no_fire + 1
        if self._no_fire >= cfg.no_fire_ticks:
            self.v_threshold = max(VTH_FLOOR, self.v_threshold - cfg.guard_vth_step)
            events.append("no_fire_threshold_drop")
            self._no_fire = 0

        self._overfire = self._overfire + 1 if fired else 0
        if self._overfire >= cfg.fire_every_tick_window:
            self.v_threshold = min(VTH_CEIL, self.v_threshold + cfg.guard_vth_step)
            events.append("overfire_threshold_raise")
            self._overfire = 0

        self._dead_coh = self._dead_coh + 1 if max_verdict < DEAD_COHERENCE_FLOOR else 0
        if self._dead_coh >= cfg.dead_coherence_ticks:
            events.append("dead_coherence_noise_inject")
            self._dead_coh = 0

        self._silent = 0 if responded else self._silent + 1
        if self._silent >= cfg.silence_watchdog_ticks:
            events.append("silence_watchdog")
            self._silent = 0

        return events
