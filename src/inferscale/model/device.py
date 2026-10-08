"""Device and dtype selection with conservative, documented fallbacks."""

from __future__ import annotations

import logging

import torch

from inferscale.config import ConfigError

logger = logging.getLogger("inferscale.model")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(name: str) -> torch.device:
    """``auto`` prefers CUDA, then MPS, then CPU. Explicit requests must be available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ConfigError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("--device mps requested but MPS is not available")
    if name not in ("cuda", "mps", "cpu"):
        raise ConfigError(f"unknown device {name!r}")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """``auto``: bf16 (or fp16) on CUDA, fp16 on MPS, fp32 on CPU.

    Explicit fp16 on CPU is upgraded to fp32 (poor/slow kernel coverage) and
    explicit bf16 on MPS is downgraded to fp16 (not reliably supported).
    """
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    if name not in _DTYPES:
        raise ConfigError(f"unknown dtype {name!r}")
    dtype = _DTYPES[name]
    if device.type == "cpu" and dtype is torch.float16:
        logger.warning("float16 is not supported well on CPU; using float32 instead")
        return torch.float32
    if device.type == "mps" and dtype is torch.bfloat16:
        logger.warning("bfloat16 is not reliably supported on MPS; using float16 instead")
        return torch.float16
    return dtype
