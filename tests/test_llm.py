"""Tests for the provider-agnostic ChatModel layer.

The OpenAI adapter is exercised through an injected stub client — no network
and no openai package required.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.llm.fake import FakeChatModel
from oscillang.llm.openai import OpenAIChat


def test_fake_is_chat_model_protocol_conformant() -> None:
    fake = FakeChatModel()
    assert isinstance(fake, ChatModel)


def test_fake_default_template_deterministic() -> None:
    msgs = [Message(role="system", content="be terse"), Message(role="user", content="hello")]
    a = FakeChatModel(model_name="fake-a")
    b = FakeChatModel(model_name="fake-a")
    out_a = a.invoke(msgs).text
    out_b = b.invoke(msgs).text
    assert out_a == out_b
    assert "hello" in out_a
    assert out_a.startswith("[fake-a]")
    assert a.calls[0] == msgs


def test_fake_custom_responder_and_option_log() -> None:
    opts = InvokeOptions(temperature=0.1, max_tokens=8)
    fake = FakeChatModel(model_name="f", responder=lambda ms: "RESP:" + ms[-1].content)
    out = fake.invoke([Message(role="user", content="x")], opts)
    assert out.text == "RESP:x"
    assert fake.options_seen[-1] is opts


def test_fake_ainvoke_matches_invoke() -> None:
    fake = FakeChatModel()
    msgs = [Message(role="user", content="async")]
    sync_out = fake.invoke(msgs).text
    async_out = asyncio.run(fake.ainvoke(msgs)).text
    assert sync_out == async_out
    assert len(fake.calls) == 2


def _stub_response(text: str, finish: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


class StubCompletions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict | None = None

    def create(self, **kwargs):  # noqa: ANN201
        self.kwargs = kwargs
        return _stub_response(self.text)


class StubClient:
    def __init__(self, text: str) -> None:
        self.chat = SimpleNamespace(completions=StubCompletions(text))


def test_openai_adapter_builds_request_and_response() -> None:
    client = StubClient("generated text")
    adapter = OpenAIChat(model="gpt-test", client=client)
    assert isinstance(adapter, ChatModel)
    out = adapter.invoke(
        [Message(role="system", content="sys"), Message(role="user", content="q")],
        InvokeOptions(temperature=0.2, max_tokens=64, stop=("END",)),
    )
    kwargs = client.chat.completions.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64
    assert kwargs["stop"] == ["END"]
    assert "response_format" not in kwargs
    assert out.text == "generated text"
    assert out.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    assert out.stop_reason == "stop"


def test_openai_adapter_passes_response_format() -> None:
    client = StubClient("{}")
    adapter = OpenAIChat(model="gpt-test", client=client)
    schema = {"type": "json_schema", "json_schema": {"name": "decision"}}
    adapter.invoke([Message(role="user", content="q")], InvokeOptions(response_format=schema))
    assert client.chat.completions.kwargs["response_format"] == schema


def test_openai_adapter_async() -> None:
    class AsyncStubCompletions:
        def __init__(self) -> None:
            self.kwargs: dict | None = None

        async def create(self, **kwargs):  # noqa: ANN201
            self.kwargs = kwargs
            return _stub_response("async text")

    async_client = SimpleNamespace(chat=SimpleNamespace(completions=AsyncStubCompletions()))
    adapter = OpenAIChat(model="gpt-test", client=StubClient("x"), async_client=async_client)
    out = asyncio.run(adapter.ainvoke([Message(role="user", content="q")]))
    assert out.text == "async text"


def test_openai_lazy_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="oscillang\\[openai\\]"):
        OpenAIChat(model="gpt-test")
