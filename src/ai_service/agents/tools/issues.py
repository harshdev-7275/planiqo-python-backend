"""Issue tools — call the node-backend to fetch issues.

The agent has no direct DB access. These tools translate LLM-friendly
parameters into node-backend HTTP calls.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai_service.clients.node_backend import NodeBackendClient


def make_issue_tools(
    client: NodeBackendClient,
    org_slug: str,
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Build the issue-related tools bound to the given node-backend client.

    When `scoped_project_id` is set, a hard-restricted variant is returned: the
    tools always operate on that one project and expose no `project_id`
    parameter, so the agent physically cannot read another project's issues.
    """

    if scoped_project_id is not None:
        return _make_scoped_issue_tools(client, org_slug, scoped_project_id)

    @tool
    async def list_issues(
        project_id: str,
        status: str | None = None,
        assignee_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List issues in a project, optionally filtered by workflow status or assignee.

        Args:
            project_id: UUID of the project (e.g. "11111111-1111-1111-1111-111111111111").
            status: Optional filter — one of "todo", "in_progress", "done", "cancelled".
            assignee_id: Optional UUID of the assignee.
            limit: Maximum issues to return (1-100, default 20). Newest first.

        Returns:
            List of issue dicts with id, number, title, type, priority, status, assignee.
        """
        if limit < 1 or limit > 100:
            raise ValueError(f"limit must be 1-100, got {limit}")
        return await client.list_issues(
            org_slug,
            UUID(project_id),
            status=status,
            assignee_id=UUID(assignee_id) if assignee_id else None,
            limit=limit,
        )

    @tool
    async def get_issue(project_id: str, issue_id: str) -> dict[str, Any] | None:
        """Fetch a single issue by its UUID.

        Args:
            project_id: UUID of the project the issue belongs to.
            issue_id: UUID of the issue.

        Returns:
            Issue dict (same shape as list_issues entries), or None if no match.
        """
        return await client.get_issue(org_slug, UUID(project_id), UUID(issue_id))

    return [list_issues, get_issue]


def _make_scoped_issue_tools(
    client: NodeBackendClient,
    org_slug: str,
    project_id: str,
) -> list[Any]:
    """Issue tools locked to a single project — no `project_id` parameter."""
    bound_project = UUID(project_id)

    @tool
    async def list_issues(
        status: str | None = None,
        assignee_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List issues in the currently selected project.

        The project is fixed by the user's selection — you cannot choose another.

        Args:
            status: Optional filter — one of "todo", "in_progress", "done", "cancelled".
            assignee_id: Optional UUID of the assignee.
            limit: Maximum issues to return (1-100, default 20). Newest first.

        Returns:
            List of issue dicts with id, number, title, type, priority, status, assignee.
        """
        if limit < 1 or limit > 100:
            raise ValueError(f"limit must be 1-100, got {limit}")
        return await client.list_issues(
            org_slug,
            bound_project,
            status=status,
            assignee_id=UUID(assignee_id) if assignee_id else None,
            limit=limit,
        )

    @tool
    async def get_issue(issue_id: str) -> dict[str, Any] | None:
        """Fetch a single issue by its UUID, within the currently selected project.

        Args:
            issue_id: UUID of the issue.

        Returns:
            Issue dict (same shape as list_issues entries), or None if no match.
        """
        return await client.get_issue(org_slug, bound_project, UUID(issue_id))

    return [list_issues, get_issue]


__all__ = ["make_issue_tools"]
