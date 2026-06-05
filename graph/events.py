"""Apply a single entity-change event to the knowledge graph.

The node-backend fires one of these (fire-and-forget, ``X-Internal-Secret``) on
every direct REST write — issue created/updated, sprint created, etc. — so
changes made *outside* the AI-bot path still reach Neo4j between full syncs.
Without this, the graph only saw bot writes and drifted from Postgres until the
next ``/graph/sync``.

Graph is optional infrastructure (AIService.md): if Neo4j is unreachable the
event is dropped with a warning, never raised. The node-backend treats the call
as best-effort and does not block the user's write on it, so returning a soft
``{"ok": False}`` (rather than a 500) keeps both sides resilient.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from clients.neo4j_client import Neo4jClient
from graph import sync


async def _issue(neo4j_client: Neo4jClient, data: dict[str, Any]) -> None:
    await sync.upsert_issue(neo4j_client, data)


async def _project(neo4j_client: Neo4jClient, data: dict[str, Any]) -> None:
    await sync.upsert_project(neo4j_client, data)


async def _sprint(neo4j_client: Neo4jClient, data: dict[str, Any]) -> None:
    await sync.upsert_sprint(neo4j_client, data)


async def _user(neo4j_client: Neo4jClient, data: dict[str, Any]) -> None:
    await sync.upsert_user(neo4j_client, data)


async def _member(neo4j_client: Neo4jClient, data: dict[str, Any]) -> None:
    await sync.upsert_member(
        neo4j_client, data["userId"], data["projectId"], data["role"]
    )


# Entity name → sync handler. The node-backend sends the entity name and a data
# payload shaped like the corresponding bot-route response, so the same sync
# functions used by full_sync apply here unchanged.
_HANDLERS: dict[str, Callable[[Neo4jClient, dict[str, Any]], Awaitable[None]]] = {
    "issue":   _issue,
    "project": _project,
    "sprint":  _sprint,
    "user":    _user,
    "member":  _member,
}

SUPPORTED_ENTITIES = frozenset(_HANDLERS)


async def apply_event(
    neo4j_client: Neo4jClient,
    entity: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch one graph event to the matching sync function.

    Returns a soft result dict rather than raising:
      * unknown entity → ``{"ok": False, "reason": "unknown_entity"}``
      * Neo4j/transport failure → ``{"ok": False, "reason": "graph_unavailable"}``
      * success → ``{"ok": True, "entity": entity}``
    """
    handler = _HANDLERS.get(entity)
    if handler is None:
        logger.warning("graph event for unknown entity {!r} ignored", entity)
        return {"ok": False, "reason": "unknown_entity", "entity": entity}

    try:
        await handler(neo4j_client, data)
        return {"ok": True, "entity": entity}
    except KeyError as e:
        # A required field was missing from the payload — a node-side bug, not a
        # graph outage. Log loudly but still don't 500 the best-effort caller.
        logger.warning("graph event {} missing field {}", entity, e)
        return {"ok": False, "reason": "bad_payload", "entity": entity}
    except Exception as e:  # noqa: BLE001 — graph is best-effort, never raise to caller
        logger.warning("graph event {} failed: {}", entity, e)
        return {"ok": False, "reason": "graph_unavailable", "entity": entity}
