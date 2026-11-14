"""Provider-agnostic chat model interface.

``ChatModel`` is the single seam between Jev/Main Brain and any LLM provider.
Adapters ship for OpenAI (``llm/openai.py``) and tests use ``FakeChatModel``
(``llm/fake.py``). The generative Main Brain and any LLM-backed Jev both go
through this protocol and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = ["ChatModel", "ChatResponse", "Message", "InvokeOptions"]

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True)
class InvokeOptions:
    """Per-call generation settings; adapters map these to provider params."""

    temperature: float = 0.7
    max_tokens: int = 1024
    stop: tuple[str, ...] | None = None
    # Provider-specific structured-output spec (e.g. OpenAI json_schema dict);
    # validated downstream by the Jev schema layer.
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatResponse:
    """Normalized completion result."""

    text: str
    model: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Anything that can render one chat completion, sync and async."""

    model_name: str

    def invoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse: ...

    async def ainvoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse: ...


def flatten_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Convert Message dataclasses to plain role/content dicts for providers."""
    return [{"role": m.role, "content": m.content} for m in messages]
