"""Pydantic schemas for health endpoints and the service root."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Response shape for `GET /health`.

    Always returned as long as the process can answer HTTP.
    """

    status: str = Field(default="ok", description="Liveness status.")


class ReadinessResponse(BaseModel):
    """Response shape for `GET /health/ready`.

    Includes service identity so a load balancer or operator can confirm
    which build is serving traffic.
    """

    status: str = Field(default="ready", description="Readiness status.")
    app: str = Field(..., description="Service name.")
    version: str = Field(..., description="Service version.")
    environment: str = Field(..., description="Deployment environment.")


class ServiceInfoResponse(BaseModel):
    """Response shape for `GET /` — service identity and discovery pointers."""

    name: str = Field(..., description="Service name.")
    version: str = Field(..., description="Service version.")
    environment: str = Field(..., description="Deployment environment.")
    status: str = Field(default="ok", description="Reachability status.")
    docs_url: str = Field(..., description="Path to the Swagger UI.")


__all__ = ["LivenessResponse", "ReadinessResponse", "ServiceInfoResponse"]
