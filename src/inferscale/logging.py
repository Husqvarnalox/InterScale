"""Structured-ish logging: ``message key=value key=value``."""

from __future__ import annotations

import logging
from typing import Any

LOGGER_NAME = "inferscale"


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields: dict[str, Any] = getattr(record, "fields", {})
        if not fields:
            return base
        rendered = " ".join(f"{k}={_render(v)}" for k, v in fields.items())
        return f"{base} {rendered}"


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def configure_logging(level: str = "info") -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    if not any(getattr(h, "_inferscale", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._inferscale = True  # type: ignore[attr-defined]
        handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})
