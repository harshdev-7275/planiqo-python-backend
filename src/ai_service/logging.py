"""Structured JSON logging with per-request correlation IDs.

Production logs are line-delimited JSON (one object per line) so any pipeline
(Loki, CloudWatch, Datadog) can ingest them without parse rules. Every record
also carries:

- ``request_id`` — a correlation id set by ``RequestLoggingMiddleware`` via a
  contextvar, so all logs for a single request (including async tool calls deep
  in the agent) can be stitched together.
- static service metadata (``service``, ``version``, ``env``) — so logs from
  multiple deployments/instances stay distinguishable in one stream.

The contextvar is reset at the end of every request by the middleware, which is
what prevents one request's id from bleeding into the next on the same worker.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar, Token
from typing import cast

from pythonjsonlogger import jsonlogger

# Per-request correlation id. ``None`` outside of a request (e.g. startup logs).
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Generate a fresh correlation id (hex, no dashes)."""
    return uuid.uuid4().hex


def bind_request_id(request_id: str) -> Token[str | None]:
    """Bind a correlation id to the current context; returns a reset token."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous correlation id using the token from ``bind_request_id``."""
    _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the current request's correlation id, if any."""
    return _request_id.get()


class _ContextFilter(logging.Filter):
    """Attaches the request id and static service fields to every record."""

    def __init__(self, static_fields: dict[str, str]) -> None:
        super().__init__()
        self._static = static_fields

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = _request_id.get()
        if request_id is not None:
            record.request_id = request_id
        for key, value in self._static.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    service: str | None = None,
    version: str | None = None,
    environment: str | None = None,
) -> None:
    """Configure root logging. Idempotent — safe to call multiple times.

    Args:
        level: Root log level (e.g. "INFO").
        json_output: Emit JSON (production) or human-readable lines (dev).
        service/version/environment: Static fields stamped on every record so
            logs are self-describing in a shared aggregation backend.
    """
    root = logging.getLogger()
    # Remove any existing handlers (uvicorn installs its own; we replace them).
    for existing in list(root.handlers):
        root.removeHandler(existing)

    static_fields: dict[str, str] = {}
    if service:
        static_fields["service"] = service
    if version:
        static_fields["version"] = version
    if environment:
        static_fields["env"] = environment

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
    handler.addFilter(_ContextFilter(static_fields))
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy third-party loggers — their volume drowns real signal.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use this everywhere instead of `logging.getLogger` directly."""
    return logging.getLogger(name)


__all__ = [
    "bind_request_id",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "reset_request_id",
]
