from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from config.settings import settings

EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}


class InternalAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS" or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        secret = request.headers.get("X-Internal-Secret")
        if not secret or secret != settings.INTERNAL_SECRET:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        return await call_next(request)
