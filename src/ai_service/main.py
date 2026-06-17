"""FastAPI application factory + uvicorn entry point.

Run with:
    uv run ai-service
    # or:
    uv run uvicorn ai_service.main:app --reload
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_service import __version__
from ai_service.api import api_router
from ai_service.config import Settings, get_settings
from ai_service.core import AppError, lifespan
from ai_service.logging import get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional pre-built Settings (for tests). If omitted,
            `get_settings()` is called and the result is cached.

    Returns:
        A fully-wired FastAPI instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI service for the PM monorepo.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lambda fastapi_app: lifespan(fastapi_app, settings),
    )

    # --- Middleware ---
    # CORS — allow only configured origins. Use ["*"] only in local dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception handlers ---
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Map typed AppErrors to a uniform JSON shape."""
        logger.warning(
            "request.app_error",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "details": exc.details,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort handler — never leak stack traces to clients."""
        logger.exception(
            "request.unhandled_exception",
            extra={"path": request.url.path, "exception_type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                    "details": {},
                }
            },
        )

    # --- Routers ---
    app.include_router(api_router)

    return app


# Module-level app for `uvicorn ai_service.main:app` and the `ai-service` script entry.
app: FastAPI = create_app()


def run() -> None:
    """Console-script entry point: `uv run ai-service`."""
    settings = get_settings()
    uvicorn.run(
        "ai_service.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Proxy headers when running behind a reverse proxy (nginx, Caddy, k8s ingress)
        proxy_headers=True,
        forwarded_allow_ips="*",
        # Production defaults — uvicorn workers=1 by default. Scale horizontally
        # by adding processes; use --workers N for a single-process multi-worker setup.
        access_log=settings.app_env != "production",
    )


if __name__ == "__main__":
    run()


__all__ = ["__version__", "app", "create_app", "run"]
