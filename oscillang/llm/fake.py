"""Deterministic fake chat model for tests and offline runs.

No network, no randomness: the same input always yields the same output, so
seeded acceptance tests stay reproducible. Outputs are templated from the
last user message so downstream embedding-distance comparisons (same brain /
different lattice) remain meaningful.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from oscillang.llm.base import ChatResponse, InvokeOptions, Message

__all__ = ["FakeChatModel"]

Responder = Callable[[list[Message]], str]


class FakeChatModel:
    """ChatModel double with an optional custom responder and call log."""

    model_name: str

    def __init__(
        self,
        model_name: str = "fake",
        responder: Responder | None = None,
    ) -> None:
        self.model_name = model_name
        self.responder = responder
        self.calls: list[list[Message]] = []
        self.options_seen: list[InvokeOptions | None] = []

    def _default_text(self, messages: list[Message]) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return f"[{self.model_name}] {last_user}"

    def invoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        self.options_seen.append(options)
        if self.responder is not None:
            text = self.responder(messages)
        else:
            text = self._default_text(messages)
        return ChatResponse(text=text, model=self.model_name)

    async def ainvoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse:
        # A real async client yields to the loop on network I/O; mirror that so
        # cancellation and concurrency behave realistically in tests.
        await asyncio.sleep(0)
        return self.invoke(messages, options)
