"""Tokenizer abstraction so the engine does not depend on Hugging Face types."""

from __future__ import annotations

from typing import Any, Protocol

from inferscale.model.chat import ChatMessage, format_chat_fallback


class TokenizerLike(Protocol):
    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def format_chat(self, messages: list[ChatMessage]) -> str: ...


class HFTokenizer:
    """Thin wrapper over a ``transformers`` tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id; cannot detect end of sequence")
        self.eos_token_id: int = eos
        pad = tokenizer.pad_token_id
        self.pad_token_id: int = eos if pad is None else pad

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids, skip_special_tokens=True))

    def format_chat(self, messages: list[ChatMessage]) -> str:
        if getattr(self._tok, "chat_template", None):
            return str(
                self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        return format_chat_fallback(messages)


class ByteTokenizer:
    """Self-contained UTF-8 byte tokenizer used by the offline ``builtin:tiny`` model.

    ids 0-255 are bytes; 256 = EOS, 257 = PAD, 258 = BOS.
    """

    vocab_size = 259
    eos_token_id = 256
    pad_token_id = 257
    bos_token_id = 258

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def format_chat(self, messages: list[ChatMessage]) -> str:
        return format_chat_fallback(messages)
