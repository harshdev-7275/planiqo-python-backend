"""Neo4j health endpoint — separate from /health/ready so callers can probe
graph reachability without coupling the overall readiness signal to Neo4j.

A flaky Neo4j should not take the whole service out of readiness — the AI
features degrade gracefully, but the HTTP service stays up.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status

router = APIRouter(tags=["neo4j"])


@router.get(
    "/neo4j/health",
    status_code=status.HTTP_200_OK,
    summary="Neo4j health probe",
    description="Returns 200 with reachability status. Does NOT fail readiness if Neo4j is down.",
)
async def neo4j_health(request: Request) -> dict[str, Any]:
    """Check Neo4j reachability via the client stored on app.state."""
    client = getattr(request.app.state, "neo4j", None)
    if client is None:
        return {
            "configured": False,
            "healthy": False,
            "reason": "Neo4j not configured (NEO4J_URL or NEO4J_PASSWORD missing)",
        }
    healthy = await client.health_check()
    return {"configured": True, "healthy": healthy}


__all__ = ["router"]
