"""Agent tools — call the node-backend over HTTP, and the Neo4j graph.

Tools are built per-request because they capture the org_slug from the
caller's context. Graph tools are added only when Neo4j is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_service.agents.tools.issues import make_issue_tools
from ai_service.agents.tools.projects import make_project_tools
from ai_service.agents.tools.users import make_user_tools
from ai_service.clients.node_backend import NodeBackendClient

if TYPE_CHECKING:
    from ai_service.neo4j import Neo4jClient


def build_all_tools(
    client: NodeBackendClient,
    org_slug: str,
    *,
    neo4j_client: Neo4jClient | None = None,
    org_id: str = "",
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Return all agent tools bound to the given client and org context.

    Tools are recreated per request because org_slug is request-scoped.
    Graph tools are included only when `neo4j_client` is provided.

    When `scoped_project_id` is set (the user selected a project in the chat
    UI), the issue/project tools are hard-restricted to that project and the
    org-wide graph tools are filtered to it — the agent cannot reach any other
    project's data.
    """
    tools: list[Any] = [
        *make_issue_tools(client, org_slug, scoped_project_id=scoped_project_id),
        *make_project_tools(client, org_slug, scoped_project_id=scoped_project_id),
        *make_user_tools(client, org_slug),
    ]
    if neo4j_client is not None and org_id:
        from ai_service.agents.tools.graph import make_graph_tools

        tools.extend(
            make_graph_tools(neo4j_client, org_id, scoped_project_id=scoped_project_id)
        )
    return tools


__all__ = ["build_all_tools"]
