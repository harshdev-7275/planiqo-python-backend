"""HTTP client for the node-backend service.

The ai-service never connects to Postgres or Neo4j directly. All data fetches
go through this client, which calls the node-backend's REST API. The
node-backend is the only place that touches the database.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx

from ai_service.config import Settings
from ai_service.core.errors import ConfigurationError, UpstreamError


class NodeBackendClient:
    """Async HTTP client for the node-backend.

    Owns its own httpx.AsyncClient. Created at app startup, closed at shutdown.
    Every request carries a Bearer token for service-to-service auth.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.node_backend_configured:
            raise ConfigurationError(
                "node-backend is not configured. Set NODE_BACKEND_URL and "
                "NODE_BACKEND_SERVICE_TOKEN in env.",
            )
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Open the HTTP client. Idempotent — safe to call multiple times."""
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.node_backend_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._settings.node_backend_service_token}",
                "User-Agent": "ai-service/0.1.0",
            },
            timeout=self._settings.node_backend_timeout_sec,
            follow_redirects=False,
        )

    async def close(self) -> None:
        """Close the HTTP client. Idempotent."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Return True if node-backend responds to GET /health."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_my_orgs(self) -> list[dict[str, Any]]:
        """GET /orgs/me — list orgs for the authenticated user."""
        result: Any = await self._get_json("/orgs/me")
        if isinstance(result, list):
            return result
        return result.get("items", [])  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------

    async def list_projects(self, org_slug: str, limit: int = 50) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects — list projects in an org."""
        result: Any = await self._get_json(f"/orgs/{org_slug}/projects", params={"limit": limit})
        return result  # type: ignore[no-any-return]

    async def get_project(self, org_slug: str, project_id: UUID) -> dict[str, Any] | None:
        """GET /orgs/:slug/projects/:projectId — fetch one project. None if 404."""
        result: Any = await self._get_json_or_none(f"/orgs/{org_slug}/projects/{project_id}")
        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Issue endpoints
    # ------------------------------------------------------------------

    async def list_issues(
        self,
        org_slug: str,
        project_id: UUID,
        *,
        status: str | None = None,
        assignee_id: UUID | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects/:projectId/issues — list issues with optional filters."""
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if assignee_id is not None:
            params["assigneeId"] = str(assignee_id)
        data: Any = await self._get_json(
            f"/orgs/{org_slug}/projects/{project_id}/issues", params=params
        )
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def get_issue(
        self, org_slug: str, project_id: UUID, issue_id: UUID
    ) -> dict[str, Any] | None:
        """GET /orgs/:slug/projects/:projectId/issues/:issueId — fetch one issue."""
        result: Any = await self._get_json_or_none(
            f"/orgs/{org_slug}/projects/{project_id}/issues/{issue_id}"
        )
        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Org membership endpoints
    # ------------------------------------------------------------------

    async def list_org_members(self, org_slug: str, limit: int = 100) -> list[dict[str, Any]]:
        """GET /orgs/:slug/members — list members of an org."""
        data: Any = await self._get_json(f"/orgs/{org_slug}/members", params={"limit": limit})
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def list_categories(
        self, org_slug: str, project_id: UUID, limit: int = 200
    ) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects/:projectId/categories — list categories."""
        data: Any = await self._get_json(
            f"/orgs/{org_slug}/projects/{project_id}/categories",
            params={"limit": limit},
        )
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def list_sprints(
        self, org_slug: str, project_id: UUID, limit: int = 200
    ) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects/:projectId/sprints — list sprints."""
        data: Any = await self._get_json(
            f"/orgs/{org_slug}/projects/{project_id}/sprints",
            params={"limit": limit},
        )
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def list_issue_comments(
        self, org_slug: str, project_id: UUID, issue_id: UUID
    ) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects/:projectId/issues/:issueId/comments."""
        data: Any = await self._get_json(
            f"/orgs/{org_slug}/projects/{project_id}/issues/{issue_id}/comments"
        )
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def list_issue_history(
        self, org_slug: str, project_id: UUID, issue_id: UUID
    ) -> list[dict[str, Any]]:
        """GET /orgs/:slug/projects/:projectId/issues/:issueId/history."""
        data: Any = await self._get_json(
            f"/orgs/{org_slug}/projects/{project_id}/issues/{issue_id}/history"
        )
        if isinstance(data, list):
            return data
        return data.get("items", [])  # type: ignore[no-any-return]

    async def get_user(
        self,
        org_slug: str,
        *,
        user_id: UUID | None = None,
        email: str | None = None,
    ) -> dict[str, Any] | None:
        """GET /orgs/:slug/members/:userId (or ?email=) — fetch one user."""
        if (user_id is None) == (email is None):
            raise ValueError("Provide exactly one of user_id or email.")
        if user_id is not None:
            result: Any = await self._get_json_or_none(f"/orgs/{org_slug}/members/{user_id}")
            return result  # type: ignore[no-any-return]
        # Email lookup
        result = await self._get_json_or_none(f"/orgs/{org_slug}/members", params={"email": email})
        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            await self.connect()
        assert self._client is not None
        try:
            response = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"node-backend request failed: {exc}",
                details={"path": path},
            ) from exc
        if response.status_code >= 500:
            raise UpstreamError(
                f"node-backend 5xx: {response.status_code} {response.text[:200]}",
                details={"path": path, "status": response.status_code},
            )
        if response.status_code >= 400:
            raise UpstreamError(
                f"node-backend {response.status_code}: {response.text[:200]}",
                details={"path": path, "status": response.status_code},
            )
        return response.json()

    async def _get_json_or_none(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Like _get_json but returns None on 404 instead of raising."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        try:
            response = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"node-backend request failed: {exc}",
                details={"path": path},
            ) from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise UpstreamError(
                f"node-backend {response.status_code}: {response.text[:200]}",
                details={"path": path, "status": response.status_code},
            )
        return response.json()


__all__ = ["NodeBackendClient"]

