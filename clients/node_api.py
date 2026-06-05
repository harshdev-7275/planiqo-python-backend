from typing import Any, cast

import httpx
from loguru import logger

from config.settings import settings


class NodeAPIClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.NODE_API_URL,
            headers={"X-Bot-Secret": settings.BOT_SECRET},
            timeout=10.0,
        )

    def _bot_headers(
        self, user_id: str | None, idempotency_key: str | None = None
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if user_id:
            headers["X-Bot-User-Id"] = user_id
        if idempotency_key:
            # Node dedupes a create with the same key, so a retried POST after
            # a dropped response doesn't create a duplicate issue/sprint (item 17).
            headers["Idempotency-Key"] = idempotency_key
        return headers

    # --- generic HTTP verbs (JSON is an untyped boundary — callers narrow) ---

    async def get(
        self, path: str, params: dict[str, Any] | None = None, user_id: str | None = None
    ) -> Any:
        response = await self._client.get(path, params=params or {}, headers=self._bot_headers(user_id))
        response.raise_for_status()
        return response.json()

    async def post(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        user_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        response = await self._client.post(
            path, json=body or {}, headers=self._bot_headers(user_id, idempotency_key)
        )
        response.raise_for_status()
        return response.json()

    async def patch(
        self, path: str, body: dict[str, Any] | None = None, user_id: str | None = None
    ) -> Any:
        response = await self._client.patch(path, json=body or {}, headers=self._bot_headers(user_id))
        response.raise_for_status()
        return response.json()

    # --- semantic bot read helpers ---

    async def get_default_status(self, org_slug: str, project_id: str) -> dict[str, Any] | None:
        """Fetch project statuses and return the one marked isDefault, or the first if none marked."""
        statuses = cast(
            list[dict[str, Any]],
            await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/statuses"),
        )
        if not statuses:
            return None
        return next((s for s in statuses if s.get("isDefault")), statuses[0])

    async def get_issues(self, org_slug: str, project_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]],
                    await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/issues"))

    async def get_statuses(self, org_slug: str, project_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]],
                    await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/statuses"))

    async def get_projects(self, org_slug: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], await self.get(f"/bot/orgs/{org_slug}/projects"))

    async def get_project_members(self, org_slug: str, project_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]],
                    await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/members"))

    async def get_sprints(self, org_slug: str, project_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]],
                    await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/sprints"))

    async def get_active_sprint(self, org_slug: str, project_id: str) -> dict[str, Any] | None:
        """Fetch sprints and return the active one, if any."""
        sprint_list = await self.get_sprints(org_slug, project_id)
        return next((s for s in sprint_list if s.get("status") == "active"), None)

    async def add_issue_to_sprint(
        self, org_slug: str, project_id: str, sprint_id: str, issue_id: str
    ) -> dict[str, Any]:
        return cast(dict[str, Any], await self.post(
            f"/bot/orgs/{org_slug}/projects/{project_id}/sprints/{sprint_id}/issues/{issue_id}"
        ))

    async def get_members(self, org_slug: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], await self.get(f"/bot/orgs/{org_slug}/members"))

    # --- infra ---

    async def ping(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=3.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning("Node API ping failed: {}", e)
            return False

    async def close(self) -> None:
        await self._client.aclose()


node_api_client = NodeAPIClient()
