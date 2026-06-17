"""Typed exceptions for the AI service.

Use these in route handlers and services. The exception handler in
`api/errors.py` (registered in `main.py`) converts them to structured JSON
responses with stable error codes.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected application errors.

    Subclasses set `status_code` and `error_code` so the global handler can map
    them to a uniform JSON response shape.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    error_code = "NOT_FOUND"


class ValidationFailedError(AppError):
    status_code = 422
    error_code = "VALIDATION_FAILED"


class UpstreamError(AppError):
    """An external dependency (LLM provider, internal API) failed."""

    status_code = 502
    error_code = "UPSTREAM_ERROR"


class ConfigurationError(AppError):
    """Misconfiguration detected at runtime (e.g. missing API key)."""

    status_code = 500
    error_code = "CONFIGURATION_ERROR"


__all__ = [
    "AppError",
    "ConfigurationError",
    "NotFoundError",
    "UpstreamError",
    "ValidationFailedError",
]
