"""User tools — call the node-backend to fetch users and org members."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai_service.clients.node_backend import NodeBackendClient


def make_user_tools(client: NodeBackendClient, org_slug: str) -> list[Any]:
    """Build the user-related tools bound to the node-backend client."""

    @tool
    async def get_user(
        user_id: str | None = None,
        email: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a single user by UUID or email.

        Args:
            user_id: User UUID. Mutually exclusive with email.
            email: User email (case-insensitive). Mutually exclusive with user_id.

        Returns:
            User dict (id, name, email, job_title, timezone, avatar_url, last_active_at),
            or None if not found.
        """
        return await client.get_user(
            org_slug,
            user_id=UUID(user_id) if user_id else None,
            email=email,
        )

    @tool
    async def list_org_members(limit: int = 100) -> list[dict[str, Any]]:
        """List members of the user's organization.

        Args:
            limit: Max members to return (1-500, default 100). Sorted by name.

        Returns:
            List of user dicts.
        """
        if limit < 1 or limit > 500:
            raise ValueError(f"limit must be 1-500, got {limit}")
        return await client.list_org_members(org_slug, limit=limit)

    return [get_user, list_org_members]


__all__ = ["make_user_tools"]
