"""Application lifespan — startup and shutdown hooks.

The ai-service does NOT connect to Postgres directly. The only external
services it talks to are:
- node-backend (via NodeBackendClient — httpx)
- Neo4j (via Neo4jClient — graph queries, post-backfill)
- LLM provider (Groq or MiniMax — handled inside the agent)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from ai_service.config import Settings
from ai_service.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from ai_service.clients.node_backend import NodeBackendClient
    from ai_service.neo4j import Neo4jClient

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI, settings: Settings) -> AsyncIterator[None]:
    """Manage application lifecycle.

    Order:
    1. Configure logging first so subsequent lifecycle logs are structured.
    2. Open node-backend HTTP client (required for any data access).
    3. Open Neo4j driver (optional — only if configured).
    """
    configure_logging(level=settings.log_level, json_output=settings.app_env != "development")

    logger.info(
        "service.starting",
        extra={
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "app_env": settings.app_env,
        },
    )

    # --- node-backend HTTP client ---
    node_backend: NodeBackendClient | None = None
    if settings.node_backend_configured:
        # Lazy import to avoid circular dependency: clients.node_backend
        # imports core.errors, which triggers core.__init__, which triggers
        # this lifespan module. The import must happen at call time, not
        # module load time.
        from ai_service.clients.node_backend import NodeBackendClient as _NodeBackendClient

        node_backend = _NodeBackendClient(settings)
        try:
            await node_backend.connect()
            healthy = await node_backend.health_check()
            logger.info(
                "node_backend.connected",
                extra={"healthy": healthy, "url": settings.node_backend_url},
            )
        except Exception:
            logger.exception("node_backend.startup_failed")
    else:
        logger.warning(
            "node_backend.skipped",
            extra={"reason": "NODE_BACKEND_URL or NODE_BACKEND_SERVICE_TOKEN unset"},
        )
    app.state.node_backend = node_backend

    # Shared secret the BFF (node-backend) must present on inbound requests.
    # When empty (local dev), service-token auth is disabled — see chat.get_agent_deps.
    app.state.service_token = settings.node_backend_service_token

    # --- Neo4j (optional) ---
    neo4j_client: Neo4jClient | None = None
    if settings.neo4j_configured:
        # Lazy import to avoid a circular dependency (deps.py also references Neo4jClient).
        from ai_service.neo4j import Neo4jClient as _Neo4jClient

        neo4j_client = _Neo4jClient(settings)
        try:
            await neo4j_client.connect()
            healthy = await neo4j_client.health_check()
            logger.info(
                "neo4j.startup",
                extra={"healthy": healthy, "database": settings.neo4j_database},
            )
            # Apply uniqueness constraints once at startup (idempotent).
            from ai_service.graph.schema import bootstrap_schema as _bootstrap

            await _bootstrap(neo4j_client)
        except Exception:
            logger.exception("neo4j.startup_failed")
        app.state.neo4j = neo4j_client
    else:
        logger.info("neo4j.skipped", extra={"reason": "NEO4J_URL or NEO4J_PASSWORD unset"})
        app.state.neo4j = None

    yield

    # --- Shutdown ---
    if neo4j_client is not None:
        await neo4j_client.close()
    if node_backend is not None:
        await node_backend.close()
        logger.info("node_backend.closed")
    logger.info("service.stopping", extra={"app_name": settings.app_name})


__all__ = ["lifespan"]
