"""Token sampling, independent of any model or runner."""

from __future__ import annotations

import torch

from inferscale.engine.request import SamplingParams


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id from a 1-D ``[vocab]`` logits tensor.

    Returns a 0-d int64 tensor on the logits' device (no host sync).
    ``temperature == 0`` is greedy; otherwise temperature, top-k, then top-p
    (nucleus) filtering are applied before multinomial sampling.
    """
    if params.temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits.float() / params.temperature
    if 0 < params.top_k < logits.shape[-1]:
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Drop a token if the mass *before* it already reaches top_p; the most
        # likely token is therefore always kept.
        remove = (torch.cumsum(probs, dim=-1) - probs) >= params.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(0)


def sample_batch(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator | None],
) -> list[int]:
    """Sample one token per row; a single device->host transfer for the batch."""
    tokens = [sample_token(logits[i], params[i], generators[i]) for i in range(len(params))]
    return [int(t) for t in torch.stack(tokens).tolist()]


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
