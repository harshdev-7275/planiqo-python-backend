"""Core (cross-cutting) modules — errors, lifespan, etc."""

from ai_service.core.errors import (
    AppError,
    ConfigurationError,
    NotFoundError,
    UpstreamError,
    ValidationFailedError,
)
from ai_service.core.lifespan import lifespan
from ai_service.core.middleware import RequestLoggingMiddleware

__all__ = [
    "AppError",
    "ConfigurationError",
    "NotFoundError",
    "RequestLoggingMiddleware",
    "UpstreamError",
    "ValidationFailedError",
    "lifespan",
]
