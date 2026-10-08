"""Server configuration. Precedence: CLI flags > environment > defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "INFERSCALE_"

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
DTypeName = Literal["auto", "float32", "float16", "bfloat16"]
LogLevel = Literal["debug", "info", "warning", "error"]


class ConfigError(ValueError):
    """Raised for invalid user configuration."""


class ServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: DeviceName = "auto"
    dtype: DTypeName = "auto"
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_batch_size: int = Field(8, ge=1)
    max_queue_size: int = Field(64, ge=1)
    max_model_len: int = Field(2048, ge=2)
    log_level: LogLevel = "info"

    @classmethod
    def resolve(
        cls,
        cli: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ServerConfig:
        """Merge sources; CLI values of ``None`` mean "not given"."""
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for name in cls.model_fields:
            raw = env.get(ENV_PREFIX + name.upper())
            if raw not in (None, ""):
                values[name] = raw
        for name, value in (cli or {}).items():
            if value is not None:
                values[name] = value
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigError(f"invalid configuration: {problems}") from exc
