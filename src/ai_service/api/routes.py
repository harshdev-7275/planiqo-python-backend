"""Root API router — aggregates feature routers.

Mounted by `main.create_app()`. Add new feature routers here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ai_service.api import chat, graph_routes, health, neo4j_routes, suggestions
from ai_service.config import Settings, get_settings
from ai_service.schemas import ServiceInfoResponse

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(neo4j_routes.router)
api_router.include_router(chat.router)
api_router.include_router(graph_routes.router)
api_router.include_router(suggestions.router)


@api_router.get(
    "/",
    response_model=ServiceInfoResponse,
    summary="Service info",
    description="Returns service identity and a pointer to the docs.",
)
async def root(settings: Settings = Depends(get_settings)) -> ServiceInfoResponse:
    """Service root — useful for sanity-checking the deployment."""
    return ServiceInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        status="ok",
        docs_url="/docs",
    )


__all__ = ["api_router"]
