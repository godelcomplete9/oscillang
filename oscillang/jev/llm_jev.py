"""LlmJev: the cheap decision model scores a shortlist in ONE call.

The shortlist is pre-filtered by RuleJev so the call stays cheap no matter
how large the lattice gets. Unparseable or schema-invalid responses score
zero - silence can never fire an oscillator, and Main Brain is never
consulted without a fire decision.
"""

from __future__ import annotations

import numpy as np

from oscillang.core.config import Config
from oscillang.jev.base import JevContext
from oscillang.jev.rule_jev import RuleJev
from oscillang.jev.schema import parse_jev_decision
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.skills.crafter import _extract_json_object

__all__ = ["LlmJev"]


class LlmJev:
    name = "llm-jev"

    def __init__(
        self,
        chat_model: ChatModel,
        config: Config,
        shortlist_k: int = 24,
        temperature: float = 0.0,
    ) -> None:
        self.chat_model = chat_model
        self.config = config
        self.shortlist_k = max(1, int(shortlist_k))
        self.temperature = temperature
        self._prefilter = RuleJev(config)
        self.calls: list[list[Message]] = []

    def _prompt(self, rows: list[int], ctx: JevContext, base: np.ndarray) -> str:
        lines = [
            "You are Jev, a cheap decision model. Score each candidate "
            "oscillator 0.0-1.0 for firing this tick: high when its coherence, "
            "phase and freshness agree with the input, low when it is tired, "
            "overused or off-topic.",
            "",
        ]
        for row in rows:
            lines.append(
                f"{ctx.ids[row]}: coherence={float(ctx.coherence[row]):.3f}, "
                f"phaseProx={float(ctx.phase_prox[row]):.3f}, "
                f"memory={float(ctx.memory_sim[row]):.3f}, "
                f"amplitude={float(ctx.amplitude[row]):.3f}, "
                f"usedCost={float(ctx.used_cost[row]):.3f}, "
                f"refractory={bool(ctx.refractory[row])}, "
                f"ruleScore={float(base[row]):.3f}"
            )
        lines.append("")
        lines.append(
            'Return JSON only: {"scores": [{"oscillatorId": "...", "score": 0.0,'
            ' "reason": "..."}]} with one entry per candidate above.'
        )
        return "\n".join(lines)

    def score(self, ctx: JevContext) -> np.ndarray:
        n = len(ctx.ids)
        base = self._prefilter.score(ctx)
        order = np.argsort(-base)[: self.shortlist_k]
        rows = [int(r) for r in order]
        prompt = self._prompt(rows, ctx, base)
        messages = [Message(role="user", content=prompt)]
        self.calls.append(list(messages))
        response = self.chat_model.invoke(messages, InvokeOptions(temperature=self.temperature))
        parsed = _extract_json_object(response.text) or {}
        decision = parse_jev_decision(parsed)
        out = np.zeros(n, dtype=np.float32)
        if decision is None:
            return out
        id_to_row = {osc_id: i for i, osc_id in enumerate(ctx.ids)}
        for entry in decision.scores:
            row = id_to_row.get(entry.oscillatorId)
            if row is not None:
                out[row] = float(np.clip(entry.score, 0.0, 1.0))
        return out + ctx.noise
