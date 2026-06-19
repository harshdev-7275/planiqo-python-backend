"""Sprint tools — call the node-backend to fetch sprints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai_service.clients.node_backend import NodeBackendClient


def make_sprint_tools(
    client: NodeBackendClient,
    org_slug: str,
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Build sprint-related tools bound to the node-backend client.

    When ``scoped_project_id`` is set, the tool is hard-restricted to that
    project.
    """

    if scoped_project_id is not None:
        return _make_scoped_sprint_tools(client, org_slug, scoped_project_id)

    @tool
    async def list_sprints(
        project_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List all sprints in a project.

        Sprints are time-boxed iterations. Use this to answer questions like
        "what sprints do we have?", "is there an active sprint?", or to resolve
        a sprint name to its id.

        Args:
            project_id: UUID of the project.
            limit: Max sprints to return (1-200, default 200).

        Returns:
            List of sprint dicts: id, name, status, startDate, endDate, projectId.
        """
        if limit < 1 or limit > 200:
            limit = 200
        return await client.list_sprints(org_slug, UUID(project_id), limit=limit)

    return [list_sprints]


def _make_scoped_sprint_tools(
    client: NodeBackendClient,
    org_slug: str,
    project_id: str,
) -> list[Any]:
    """Sprint tools locked to a single project."""
    bound_project = UUID(project_id)

    @tool
    async def list_sprints(limit: int = 200) -> list[dict[str, Any]]:
        """List all sprints in the currently selected project.

        Sprints are time-boxed iterations. Use this to answer questions like
        "what sprints do we have?", "is there an active sprint?", or to resolve
        a sprint name to its id.

        Args:
            limit: Max sprints to return (1-200, default 200).

        Returns:
            List of sprint dicts: id, name, status, startDate, endDate, projectId.
        """
        if limit < 1 or limit > 200:
            limit = 200
        return await client.list_sprints(org_slug, bound_project, limit=limit)

    return [list_sprints]


__all__ = ["make_sprint_tools"]
