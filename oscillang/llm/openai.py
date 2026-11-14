"""OpenAI adapter for the ChatModel protocol.

The ``openai`` package is imported lazily so it stays an optional dependency;
tests inject a stub client instead of hitting the network.
"""

from __future__ import annotations

from typing import Any

from oscillang.llm.base import ChatResponse, InvokeOptions, Message, flatten_messages

__all__ = ["OpenAIChat"]


class OpenAIChat:
    """ChatModel backed by the official OpenAI client (sync + async)."""

    model_name: str

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.model_name = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "OpenAIChat requires the openai package: pip install 'oscillang[openai]'"
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._async_client: Any | None = async_client

    @property
    def async_client(self) -> Any:
        """Async client, constructed on first use so sync-only setups need no key."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI()
        return self._async_client

    def _build_kwargs(
        self,
        messages: list[Message],
        options: InvokeOptions | None,
    ) -> dict[str, Any]:
        opts = options or InvokeOptions()
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": flatten_messages(messages),
            "temperature": opts.temperature,
            "max_tokens": opts.max_tokens,
        }
        if opts.stop is not None:
            kwargs["stop"] = list(opts.stop)
        if opts.response_format is not None:
            kwargs["response_format"] = opts.response_format
        return kwargs

    def invoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse:
        completion = self._client.chat.completions.create(**self._build_kwargs(messages, options))
        return self._to_response(completion)

    async def ainvoke(
        self,
        messages: list[Message],
        options: InvokeOptions | None = None,
    ) -> ChatResponse:
        completion = await self.async_client.chat.completions.create(
            **self._build_kwargs(messages, options)
        )
        return self._to_response(completion)

    def _to_response(self, completion: Any) -> ChatResponse:
        choice = completion.choices[0]
        usage: dict[str, int] = {}
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                "total_tokens": getattr(raw_usage, "total_tokens", 0),
            }
        return ChatResponse(
            text=choice.message.content or "",
            model=self.model_name,
            stop_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )
