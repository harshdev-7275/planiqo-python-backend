"""Category tools — call the node-backend to fetch categories."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai_service.clients.node_backend import NodeBackendClient


def make_category_tools(
    client: NodeBackendClient,
    org_slug: str,
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Build category-related tools bound to the node-backend client.

    When ``scoped_project_id`` is set, the tool is hard-restricted to that
    project — the agent cannot list another project's categories.
    """

    if scoped_project_id is not None:
        return _make_scoped_category_tools(client, org_slug, scoped_project_id)

    @tool
    async def list_categories(
        project_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List all categories in a project.

        Categories are the top-level grouping for issues — every issue belongs
        to exactly one category. Use this to answer questions like "what
        categories exist?", "show me the project structure", or to resolve a
        category name to its id before filtering issues.

        Args:
            project_id: UUID of the project.
            limit: Max categories to return (1-200, default 200).

        Returns:
            List of category dicts: id, name, color, description, sprintId, projectId.
        """
        if limit < 1 or limit > 200:
            limit = 200
        return await client.list_categories(org_slug, UUID(project_id), limit=limit)

    return [list_categories]


def _make_scoped_category_tools(
    client: NodeBackendClient,
    org_slug: str,
    project_id: str,
) -> list[Any]:
    """Category tools locked to a single project."""
    bound_project = UUID(project_id)

    @tool
    async def list_categories(limit: int = 200) -> list[dict[str, Any]]:
        """List all categories in the currently selected project.

        Categories are the top-level grouping for issues — every issue belongs
        to exactly one category. Use this to answer questions like "what
        categories exist?", "show me the project structure", or to resolve a
        category name to its id.

        Args:
            limit: Max categories to return (1-200, default 200).

        Returns:
            List of category dicts: id, name, color, description, sprintId, projectId.
        """
        if limit < 1 or limit > 200:
            limit = 200
        return await client.list_categories(org_slug, bound_project, limit=limit)

    return [list_categories]


__all__ = ["make_category_tools"]
