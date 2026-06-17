"""Structured JSON logging configuration.

Emits one JSON object per log line so logs can be ingested by any log pipeline
(Loki, CloudWatch, Datadog) without parsing rules.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure root logging. Idempotent — safe to call multiple times."""
    root = logging.getLogger()
    # Remove any existing handlers (uvicorn installs its own; we replace them)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    if json_output:
        formatter: logging.Formatter = cast(
            "logging.Formatter",
            jsonlogger.JsonFormatter(  # type: ignore[no-untyped-call]
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
                timestamp=True,
            ),
        )
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use this everywhere instead of `logging.getLogger` directly."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
