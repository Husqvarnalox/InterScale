"""ModelRunner: the only component that calls ``model(...)``.

It exposes two explicit phases to the engine:

* ``prefill``: run the whole (left-padded) prompt batch once, building the KV cache.
* ``decode``: feed one new token per sequence, reusing and extending the cache.

The runner knows token ids and tensors only; it knows nothing about requests,
sampling, or HTTP.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from inferscale.model.tokenizer import TokenizerLike


class BatchState(Protocol):
    """Opaque per-batch KV cache owner. Freed when the engine drops the reference."""

    def keep(self, rows: list[int]) -> None:
        """Retain only the given batch rows (finished/cancelled rows are evicted)."""
        ...


class Runner(Protocol):
    tokenizer: TokenizerLike
    device: torch.device
    max_model_len: int
    name: str
    last_forward_seconds: float

    def prefill(self, prompts: list[list[int]]) -> tuple[BatchState, torch.Tensor]:
        """Return (state, next-token logits ``[batch, vocab]``)."""
        ...

    def decode(self, state: BatchState, tokens: list[int]) -> torch.Tensor:
        """Return next-token logits ``[batch, vocab]`` after consuming ``tokens``."""
        ...


@dataclass
class HFBatchState:
    """KV cache plus the bookkeeping that keeps left-padded rows correct."""

    cache: Any
    attention_mask: torch.Tensor  # [B, T] cumulative, grows by one column per step
    positions: torch.Tensor  # [B] position id of the next token for each row

    def keep(self, rows: list[int]) -> None:
        idx = torch.tensor(rows, device=self.attention_mask.device, dtype=torch.long)
        self.attention_mask = self.attention_mask.index_select(0, idx)
        self.positions = self.positions.index_select(0, idx)
        self.cache.batch_select_indices(idx)


@dataclass
class ForwardTimings:
    last_seconds: float = 0.0
    history: list[float] = field(default_factory=list)


class HFModelRunner:
    """Runs a Hugging Face causal LM with a manual prefill/decode loop."""

    def __init__(
        self,
        model: Any,
        tokenizer: TokenizerLike,
        device: torch.device,
        max_model_len: int,
        name: str,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_model_len = max_model_len
        self.name = name
        self.last_forward_seconds = 0.0
        params = inspect.signature(model.forward).parameters
        self._supports_logits_to_keep = "logits_to_keep" in params

    def _sync(self) -> None:
        # The caller needs the logits immediately after, so syncing here adds no
        # stall; it only makes the measured forward time real on async devices.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _forward(self, **kwargs: Any) -> Any:
        if self._supports_logits_to_keep:
            kwargs["logits_to_keep"] = 1  # skip the lm_head over all prompt positions
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(use_cache=True, **kwargs)
        self._sync()
        self.last_forward_seconds = time.perf_counter() - start
        return out

    def prefill(self, prompts: list[list[int]]) -> tuple[HFBatchState, torch.Tensor]:
        batch = len(prompts)
        width = max(len(p) for p in prompts)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((batch, width), pad, dtype=torch.long)
        mask = torch.zeros((batch, width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            mask[row, width - len(prompt) :] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        ids, mask, positions = (t.to(self.device) for t in (ids, mask, positions))

        out = self._forward(input_ids=ids, attention_mask=mask, position_ids=positions)
        state = HFBatchState(
            cache=out.past_key_values,
            attention_mask=mask,
            positions=mask.sum(-1),
        )
        return state, out.logits[:, -1, :].float()

    def decode(self, state: BatchState, tokens: list[int]) -> torch.Tensor:
        assert isinstance(state, HFBatchState)
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        ones = torch.ones((ids.shape[0], 1), dtype=state.attention_mask.dtype, device=self.device)
        state.attention_mask = torch.cat([state.attention_mask, ones], dim=1)
        out = self._forward(
            input_ids=ids,
            attention_mask=state.attention_mask,
            position_ids=state.positions.unsqueeze(1),
            past_key_values=state.cache,
        )
        state.cache = out.past_key_values
        state.positions = state.positions + 1
        logits: torch.Tensor = out.logits[:, -1, :].float()
        return logits
