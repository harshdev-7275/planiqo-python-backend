"""Core (cross-cutting) modules — errors, lifespan, etc."""

from ai_service.core.errors import (
    AppError,
    ConfigurationError,
    NotFoundError,
    UpstreamError,
    ValidationFailedError,
)
from ai_service.core.lifespan import lifespan

__all__ = [
    "AppError",
    "ConfigurationError",
    "NotFoundError",
    "UpstreamError",
    "ValidationFailedError",
    "lifespan",
]
