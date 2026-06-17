"""Project tools — call the node-backend to fetch projects."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai_service.clients.node_backend import NodeBackendClient


def make_project_tools(
    client: NodeBackendClient,
    org_slug: str,
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Build the project-related tools bound to the node-backend client.

    The org_slug is captured at agent-build time so the agent doesn't have
    to know it for every call (it varies per request — see how this is
    wired in api/chat.py).

    When `scoped_project_id` is set, the tools are hard-restricted to that one
    project: `list_projects` returns only it, and `get_project` always returns
    it regardless of any id the agent supplies.
    """

    if scoped_project_id is not None:
        return _make_scoped_project_tools(client, org_slug, scoped_project_id)

    @tool
    async def list_projects(limit: int = 50) -> list[dict[str, Any]]:
        """List projects in the user's organization.

        Args:
            limit: Max projects to return (1-200, default 50). Sorted by name.

        Returns:
            List of project dicts: id, key, name, description, icon, color, is_archived.
        """
        if limit < 1 or limit > 200:
            raise ValueError(f"limit must be 1-200, got {limit}")
        return await client.list_projects(org_slug, limit=limit)

    @tool
    async def get_project(project_id: str) -> dict[str, Any] | None:
        """Fetch a single project by its UUID.

        Args:
            project_id: UUID of the project.

        Returns:
            Project dict, or None if not found.
        """
        return await client.get_project(org_slug, UUID(project_id))

    return [list_projects, get_project]


def _make_scoped_project_tools(
    client: NodeBackendClient,
    org_slug: str,
    project_id: str,
) -> list[Any]:
    """Project tools locked to a single project the user selected."""
    bound_project = UUID(project_id)

    @tool
    async def list_projects(limit: int = 50) -> list[dict[str, Any]]:
        """List projects — restricted to the currently selected project only.

        Returns:
            A single-element list with the selected project, or empty if it
            could not be found.
        """
        project = await client.get_project(org_slug, bound_project)
        return [project] if project else []

    @tool
    async def get_project(project_id: str = "") -> dict[str, Any] | None:
        """Fetch the currently selected project.

        The selection is fixed by the user; any `project_id` you pass is ignored
        and the selected project is always returned.

        Returns:
            Project dict, or None if not found.
        """
        return await client.get_project(org_slug, bound_project)

    return [list_projects, get_project]


__all__ = ["make_project_tools"]
