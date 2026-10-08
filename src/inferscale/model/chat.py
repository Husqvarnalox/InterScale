"""Chat message type and the documented fallback prompt format."""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


def format_chat_fallback(messages: list[ChatMessage]) -> str:
    """Used when the tokenizer ships no chat template.

    Format (one block per message, then an open assistant turn)::

        <|system|>
        You are helpful.
        <|user|>
        Hi
        <|assistant|>
    """
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    parts.append("<|assistant|>\n")
    return "".join(parts)
