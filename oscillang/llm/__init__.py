"""LLM provider adapters behind the ChatModel protocol."""

from oscillang.llm.base import ChatModel, ChatResponse, InvokeOptions, Message
from oscillang.llm.fake import FakeChatModel

__all__ = ["ChatModel", "ChatResponse", "FakeChatModel", "InvokeOptions", "Message"]
