"""Request-logging middleware — correlation ids + structured access logs.

Wraps every request to:
1. assign (or honour an inbound) ``X-Request-ID`` and bind it for the duration
   of the request so all downstream logs share the same correlation id;
2. emit one structured ``request.completed`` log with method, path, status and
   latency (and ``request.started`` for symmetry);
3. echo the correlation id back in the ``X-Request-ID`` response header.

Health-probe paths are logged at DEBUG so they don't flood INFO in production.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ai_service.logging import (
    bind_request_id,
    get_logger,
    new_request_id,
    reset_request_id,
)

logger = get_logger("ai_service.request")

_REQUEST_ID_HEADER = "X-Request-ID"
# Probe/noise paths: still correlated, but logged at DEBUG to keep INFO clean.
_QUIET_PREFIXES = ("/health",)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Correlate and log every HTTP request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or new_request_id()
        token = bind_request_id(request_id)

        path = request.url.path
        level = logging.DEBUG if path.startswith(_QUIET_PREFIXES) else logging.INFO
        client = request.client.host if request.client else None
        start = time.perf_counter()

        logger.log(
            level,
            "request.started",
            extra={"method": request.method, "path": path, "client": client},
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            # Logged while the request id is still bound, then reset before re-raising.
            logger.exception(
                "request.failed",
                extra={"method": request.method, "path": path, "duration_ms": duration_ms},
            )
            reset_request_id(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.log(
            level,
            "request.completed",
            extra={
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers[_REQUEST_ID_HEADER] = request_id
        reset_request_id(token)
        return response


__all__ = ["RequestLoggingMiddleware"]
