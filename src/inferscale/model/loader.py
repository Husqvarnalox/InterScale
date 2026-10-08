"""Model loading. Errors are wrapped in ``ModelLoadError`` with actionable text."""

from __future__ import annotations

import logging
from typing import Any

import torch

from inferscale.config import ServerConfig
from inferscale.model.device import resolve_device, resolve_dtype
from inferscale.model.runner import HFModelRunner
from inferscale.model.tokenizer import ByteTokenizer, HFTokenizer

logger = logging.getLogger("inferscale.model")

BUILTIN_TINY = "builtin:tiny"


class ModelLoadError(RuntimeError):
    pass


def _build_builtin_tiny() -> tuple[Any, ByteTokenizer]:
    """A ~100k-param randomly initialised Llama + byte tokenizer. Offline smoke tests only."""
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = ByteTokenizer()
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), tok  # type: ignore[no-untyped-call]


def build_tiny_runner(max_model_len: int = 512, device: str = "cpu") -> HFModelRunner:
    model, tok = _build_builtin_tiny()
    dev = torch.device(device)
    return HFModelRunner(model.to(dev), tok, dev, max_model_len, BUILTIN_TINY)


def load_runner(config: ServerConfig) -> HFModelRunner:
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    logger.info("loading model %s on %s (%s)", config.model, device, dtype)

    if config.model == BUILTIN_TINY:
        tiny, tiny_tok = _build_builtin_tiny()
        tiny = tiny.to(device=device, dtype=dtype)
        return HFModelRunner(tiny, tiny_tok, device, config.max_model_len, config.model)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        hf_tok = AutoTokenizer.from_pretrained(config.model)
        model: Any = AutoModelForCausalLM.from_pretrained(config.model, dtype=dtype)
        tokenizer = HFTokenizer(hf_tok)
    except Exception as exc:
        raise ModelLoadError(
            f"could not load model {config.model!r}: {exc}. Check the model id/path and "
            "network access, or use --model builtin:tiny for an offline smoke test."
        ) from exc

    model = model.to(device)
    ctx = getattr(model.config, "max_position_embeddings", None)
    max_len = min(config.max_model_len, ctx) if ctx else config.max_model_len
    return HFModelRunner(model, tokenizer, device, max_len, config.model)
