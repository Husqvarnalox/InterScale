"""Per-request decoding state: detokenization, stop conditions, finish reasons."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from inferscale.engine.request import GenerationRequest
from inferscale.model.tokenizer import TokenizerLike


class IncrementalDetokenizer:
    """Decodes only a small trailing window per token (never the whole output).

    ``prefix_offset``/``read_offset`` mark a window of already-emitted context
    needed so multi-token characters (and SentencePiece spacing) decode right.
    Text ending in U+FFFD is withheld until more bytes arrive.
    """

    def __init__(self, tokenizer: TokenizerLike) -> None:
        self._tok = tokenizer
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        prefix_text = self._tok.decode(self._ids[self._prefix : self._read])
        new_text = self._tok.decode(self._ids[self._prefix :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix = self._read
            self._read = len(self._ids)
            return new_text[len(prefix_text) :]
        return ""

    def flush(self) -> str:
        prefix_text = self._tok.decode(self._ids[self._prefix : self._read])
        new_text = self._tok.decode(self._ids[self._prefix :])
        self._prefix = self._read = len(self._ids)
        return new_text[len(prefix_text) :]


@dataclass
class StepOutput:
    text: str
    finish_reason: str | None


class Sequence:
    """Engine-side decoding state for one running request."""

    def __init__(
        self,
        request: GenerationRequest,
        tokenizer: TokenizerLike,
    ) -> None:
        self.request = request
        self.max_tokens = request.params.max_tokens
        self.eos_token_id = tokenizer.eos_token_id
        self.generator: torch.Generator | None = None
        self._detok = IncrementalDetokenizer(tokenizer)
        self._stop = tuple(s for s in request.params.stop if s)
        self._held = ""  # text withheld because it might start a stop string
        self.last_token: int | None = None

    def _hold_len(self, text: str) -> int:
        """Longest suffix of ``text`` that is a proper prefix of some stop string."""
        best = 0
        for stop in self._stop:
            for n in range(min(len(stop) - 1, len(text)), best, -1):
                if text.endswith(stop[:n]):
                    best = n
                    break
        return best

    def _finish(self, reason: str, tail: str) -> StepOutput:
        return StepOutput(text=self._held + tail, finish_reason=reason)

    def append(self, token_id: int) -> StepOutput:
        """Consume a sampled token and return the text now safe to emit."""
        self.last_token = token_id
        if token_id == self.eos_token_id:
            self._held += self._detok.flush()
            return self._finish("stop", "")

        self.request.generated_token_ids.append(token_id)
        pending = self._held + self._detok.push(token_id)
        self._held = ""

        for stop in self._stop:
            idx = pending.find(stop)
            if idx != -1:
                return StepOutput(text=pending[:idx], finish_reason="stop")

        if len(self.request.generated_token_ids) >= self.max_tokens:
            self._held = pending
            return self._finish("length", self._detok.flush())

        hold = self._hold_len(pending) if self._stop else 0
        if hold:
            self._held = pending[-hold:]
            return StepOutput(text=pending[:-hold], finish_reason=None)
        return StepOutput(text=pending, finish_reason=None)
