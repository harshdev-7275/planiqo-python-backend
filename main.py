import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from agents import supervisor
from chains.render_contract import attach_render_blocks
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from config.settings import settings
from graph.events import apply_event
from graph.schema import apply_constraints
from graph.sync import full_sync
from metering.metrics import latency_stats
from metering.usage import usage_store
from middleware.auth import InternalAuthMiddleware


class ChatRequest(BaseModel):
    message: str
    user_id: str
    org_slug: str
    project_id: str | None = None


class SyncRequest(BaseModel):
    org_slug: str


class GraphEvent(BaseModel):
    """One entity-change pushed by the node-backend on a direct REST write.

    ``entity`` is one of graph.events.SUPPORTED_ENTITIES; ``data`` carries the
    fields the matching sync function needs (shaped like the bot-route response
    for that entity)."""

    entity: str
    data: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("AI service starting up")
    # Neo4j is optional infrastructure: the graph powers smart-assignee and
    # similarity, which already degrade gracefully per-call. If it is unreachable
    # at boot, log and carry on — core chat/health must still come up.
    try:
        neo4j_client.connect()
        await apply_constraints(neo4j_client)
        logger.info("Neo4j ready")
    except Exception as e:
        logger.warning(
            "Neo4j unavailable at startup ({}). Continuing without graph features — "
            "smart-assignee and similarity are skipped until it recovers.",
            e,
        )
    yield
    await neo4j_client.close()
    await node_api_client.close()
    logger.info("AI service shut down")


app = FastAPI(title="AI Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health")
async def health() -> dict[str, Any]:
    node_ok = await node_api_client.ping()
    neo4j_ok = await neo4j_client.ping()
    return {
        "status": "ok",
        "node_api": node_ok,
        "neo4j": neo4j_ok,
    }


@app.post("/graph/sync")
async def graph_sync(body: SyncRequest) -> dict[str, int]:
    return await full_sync(
        neo4j_client=neo4j_client,
        node_api_client=node_api_client,
        org_slug=body.org_slug,
    )


@app.post("/graph/events")
async def graph_events(body: GraphEvent) -> dict[str, Any]:
    """Apply a single incremental entity change to the graph.

    Fired by the node-backend on direct (non-bot) REST writes so the graph
    stays in step with Postgres between full syncs. Best-effort: always 200,
    with ``ok: false`` in the body if the graph is unavailable or the payload
    is unknown — the caller does not block its user-facing write on this."""
    return await apply_event(neo4j_client, body.entity, body.data)


@app.get("/admin/usage/{org_slug}")
async def admin_usage(org_slug: str) -> dict[str, Any]:
    """Per-org token usage + request count. Internal-only (the auth middleware
    blocks external callers; the frontend has no need to see this)."""
    return {
        "org_slug": org_slug,
        "tokens_used": usage_store.get(org_slug),
        "request_count": usage_store.get_request_count(org_slug),
        "quota": settings.ORG_TOKEN_QUOTA,
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Ops snapshot: per-org token/request usage + /chat latency aggregates.
    Internal-only (behind the auth middleware) — for dashboards/scrapers, not
    the frontend."""
    return {
        "usage": usage_store.snapshot(),
        "latency": latency_stats.snapshot(),
    }


@app.post("/chat")
async def chat(body: ChatRequest) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        response = await supervisor.run(
            message=body.message,
            user_id=body.user_id,
            org_slug=body.org_slug,
            project_id=body.project_id,
        )
        # Service edge: guarantee the reply carries validated, camelCase
        # render blocks (at least a prose block) for the frontend renderer.
        return attach_render_blocks(response)
    finally:
        latency_stats.record((time.perf_counter() - start) * 1000)
