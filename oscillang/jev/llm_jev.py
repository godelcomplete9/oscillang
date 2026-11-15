"""LlmJev: the semantic gatekeeper - reranks candidate mini skills in ONE call.

Jev verifies semantic coherence and alignment: every candidate's actual
skill texts (trigger, fragment, examples) go into the prompt next to the
user input, and the model returns an alignment verdict per candidate.
Unparseable or schema-invalid responses yield no verdicts - silence can
never fire a mini skill, and Main Brain is never consulted without a fire
decision.
"""

from __future__ import annotations

import numpy as np

from oscillang.jev.base import JevCandidate, JevVerdict
from oscillang.jev.schema import parse_jev_decision
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.skills.crafter import _extract_json_object

__all__ = ["LlmJev"]


class LlmJev:
    name = "llm-jev"

    def __init__(self, chat_model: ChatModel, temperature: float = 0.0) -> None:
        self.chat_model = chat_model
        self.temperature = temperature
        self.calls: list[list[Message]] = []

    def _prompt(self, user_input: str, candidates: list[JevCandidate], context: str) -> str:
        lines = [
            "You are Jev, the semantic gatekeeper. Verify for each candidate "
            "behavioral fragment whether it is semantically aligned with the "
            "input: score 0.9-1.0 when its trigger and fragment clearly match "
            "the input's meaning, mid scores for partial relevance, 0.0 when "
            "unrelated. Judge alignment only - never style.",
            "",
        ]
        if context:
            lines.append(f"Previous beat: {context}")
            lines.append("")
        lines.append(f"Input: {user_input}")
        lines.append("")
        lines.append("Candidates:")
        for c in candidates:
            lines.append(f"{c.oscillator_id}: trigger={c.trigger}")
            lines.append(f"  fragment: {c.prompt_fragment}")
            if c.examples:
                lines.append(f"  examples: {' | '.join(c.examples)}")
        lines.append("")
        lines.append(
            'Return JSON only: {"scores": [{"oscillatorId": "...", "score": 0.0,'
            ' "reason": "..."}]} with one entry per candidate above.'
        )
        return "\n".join(lines)

    def rerank(
        self,
        user_input: str,
        candidates: list[JevCandidate],
        context: str = "",
    ) -> list[JevVerdict]:
        if not candidates:
            return []
        prompt = self._prompt(user_input, candidates, context)
        messages = [Message(role="user", content=prompt)]
        self.calls.append(list(messages))
        response = self.chat_model.invoke(messages, InvokeOptions(temperature=self.temperature))
        parsed = _extract_json_object(response.text) or {}
        decision = parse_jev_decision(parsed)
        if decision is None:
            return []
        offered = {c.oscillator_id for c in candidates}
        out: list[JevVerdict] = []
        seen: set[str] = set()
        for entry in decision.scores:
            if entry.oscillatorId not in offered or entry.oscillatorId in seen:
                continue
            seen.add(entry.oscillatorId)
            out.append(
                JevVerdict(
                    oscillator_id=entry.oscillatorId,
                    score=float(np.clip(entry.score, 0.0, 1.0)),
                    reason=entry.reason,
                )
            )
        return out
