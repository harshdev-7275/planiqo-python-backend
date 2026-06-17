"""Health endpoints — liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from ai_service.config import Settings, get_settings
from ai_service.schemas.health import LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns 200 as long as the process can answer HTTP. No dependency checks.",
)
async def liveness() -> LivenessResponse:
    """Process is alive."""
    return LivenessResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Returns 200 with service identity when ready to serve traffic.",
)
async def readiness(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Process is ready to accept requests."""
    return ReadinessResponse(
        status="ready",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


__all__ = ["router"]
