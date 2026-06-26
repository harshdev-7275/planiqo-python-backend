"""Graph sync and stats endpoints.

POST /v1/graph/sync                        — full org sync
POST /v1/graph/sync/project/:project_id   — incremental project sync (for node-backend webhooks)
GET  /v1/graph/stats                       — node / relationship counts

Both sync endpoints require either a service-to-service token
(Authorization: Bearer <NODE_BACKEND_SERVICE_TOKEN>) or a valid user JWT.
The stats endpoint is unauthenticated (read-only, no PII).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from ai_service.clients.node_backend import NodeBackendClient
from ai_service.graph.queries import get_graph_stats
from ai_service.graph.sync import GraphSyncService
from ai_service.logging import get_logger
from ai_service.neo4j import Neo4jClient
from ai_service.schemas.graph import (
    GraphProjectSyncRequest,
    GraphStatsResponse,
    GraphSyncRequest,
    GraphSyncResponse,
    SlackChannelSyncRequest,
    SlackSyncAck,
    SlackSyncRequest,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/graph", tags=["graph"])


def _get_neo4j(request: Request) -> Neo4jClient:
    client: Neo4jClient | None = getattr(request.app.state, "neo4j", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j is not configured. Set NEO4J_URL and NEO4J_PASSWORD in env.",
        )
    return client


def _get_backend(request: Request) -> NodeBackendClient:
    client: NodeBackendClient | None = getattr(request.app.state, "node_backend", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="node-backend client is not initialized.",
        )
    return client


@router.post(
    "/sync",
    response_model=GraphSyncResponse,
    status_code=status.HTTP_200_OK,
    summary="Full org sync",
    description=(
        "Fetches all projects, categories, sprints, issues, comments, and history "
        "for the given org from node-backend and upserts them into Neo4j. "
        "Idempotent — safe to re-run at any time."
    ),
)
async def sync_full(
    request: Request,
    body: GraphSyncRequest,
) -> GraphSyncResponse:
    neo4j = _get_neo4j(request)
    backend = _get_backend(request)
    svc = GraphSyncService(neo4j=neo4j, backend=backend)

    try:
        stats = await svc.full_sync(body.org_slug, body.org_id)
    except Exception:
        logger.exception("graph.sync.full.error", extra={"org_id": body.org_id})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Graph sync failed. Check logs for details.",
        ) from None

    return GraphSyncResponse(**stats.to_dict())


@router.post(
    "/sync/project/{project_id}",
    response_model=GraphSyncResponse,
    status_code=status.HTTP_200_OK,
    summary="Incremental project sync",
    description=(
        "Re-syncs all issues (+ comments + history) for a single project. "
        "Called by node-backend after create/update events. "
        "Skips categories/sprints/members (already seeded by a full sync)."
    ),
)
async def sync_project(
    request: Request,
    project_id: UUID,
    body: GraphProjectSyncRequest,
) -> GraphSyncResponse:
    neo4j = _get_neo4j(request)
    backend = _get_backend(request)
    svc = GraphSyncService(neo4j=neo4j, backend=backend)

    try:
        stats = await svc.sync_single_issue(
            body.org_slug, body.org_id, str(project_id)
        )
    except Exception:
        logger.exception(
            "graph.sync.project.error",
            extra={"project_id": str(project_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Incremental project sync failed. Check logs for details.",
        ) from None

    return GraphSyncResponse(**stats.to_dict())


# ---------------------------------------------------------------------------
# Slack sync stubs (Phase 0+1)
#
# Real SlackSyncService lands in Phase 2 (graph/chat_sync.py + SlackMessage
# nodes + hybrid retrieval). For now the endpoints just acknowledge the
# trigger so node-backend's kg_sync_log row closes — the same retry contract
# applies. Phase 2 will replace the body of these handlers with the actual
# sync. Both require either a service-to-service token or a user JWT, just
# like the existing sync endpoints.
# ---------------------------------------------------------------------------


@router.post(
    "/sync/slack",
    response_model=SlackSyncAck,
    status_code=status.HTTP_200_OK,
    summary="Slack full-org sync (stub)",
    description=(
        "Phase 0+1 stub — acknowledges the trigger so node-backend's "
        "kg_sync_log row closes. Real SlackSyncService arrives in Phase 2."
    ),
)
async def sync_slack_org(body: SlackSyncRequest) -> SlackSyncAck:
    logger.info("graph.sync.slack.stub", extra={"org_id": body.org_id})
    return SlackSyncAck(status="accepted", org_id=body.org_id, channel_id=None)


@router.post(
    "/sync/slack/channel/{channel_id}",
    response_model=SlackSyncAck,
    status_code=status.HTTP_200_OK,
    summary="Slack channel sync (stub)",
    description=(
        "Phase 0+1 stub for a single Slack channel. The actual sync that "
        "pulls messages from node-backend, embeds them, and MERGE-writes "
        "SlackMessage nodes is Phase 2."
    ),
)
async def sync_slack_channel(channel_id: UUID, body: SlackChannelSyncRequest) -> SlackSyncAck:
    logger.info(
        "graph.sync.slack.channel.stub",
        extra={"org_id": body.org_id, "channel_id": str(channel_id)},
    )
    return SlackSyncAck(status="accepted", org_id=body.org_id, channel_id=str(channel_id))


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Graph stats",
    description="Returns node and relationship counts by label/type. No auth required.",
)
async def graph_stats(request: Request) -> GraphStatsResponse:
    neo4j: Neo4jClient | None = getattr(request.app.state, "neo4j", None)
    if neo4j is None:
        return GraphStatsResponse(nodes={}, relationships={}, configured=False)
    try:
        data: dict[str, Any] = await get_graph_stats(neo4j)
        return GraphStatsResponse(
            nodes=data.get("nodes", {}),
            relationships=data.get("relationships", {}),
        )
    except Exception:
        logger.exception("graph.stats.error")
        return GraphStatsResponse(nodes={}, relationships={}, configured=True)


__all__ = ["router"]
