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

    def _bot_headers(self, user_id: str | None) -> dict:
        if user_id:
            return {"X-Bot-User-Id": user_id}
        return {}

    # --- generic HTTP verbs ---

    async def get(self, path: str, params: dict | None = None, user_id: str | None = None) -> dict:
        response = await self._client.get(path, params=params or {}, headers=self._bot_headers(user_id))
        response.raise_for_status()
        return response.json()

    async def post(self, path: str, body: dict | None = None, user_id: str | None = None) -> dict:
        response = await self._client.post(path, json=body or {}, headers=self._bot_headers(user_id))
        response.raise_for_status()
        return response.json()

    async def patch(self, path: str, body: dict | None = None, user_id: str | None = None) -> dict:
        response = await self._client.patch(path, json=body or {}, headers=self._bot_headers(user_id))
        response.raise_for_status()
        return response.json()

    # --- semantic bot read helpers ---

    async def get_default_status(self, org_slug: str, project_id: str) -> dict | None:
        """Fetch project statuses and return the one marked isDefault, or the first if none marked."""
        statuses: list = await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/statuses")
        if not statuses:
            return None
        return next((s for s in statuses if s.get("isDefault")), statuses[0])

    async def get_issues(self, org_slug: str, project_id: str) -> list:
        return await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/issues")

    async def get_projects(self, org_slug: str) -> list:
        return await self.get(f"/bot/orgs/{org_slug}/projects")

    async def get_project_members(self, org_slug: str, project_id: str) -> list:
        return await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/members")

    async def get_sprints(self, org_slug: str, project_id: str) -> list:
        return await self.get(f"/bot/orgs/{org_slug}/projects/{project_id}/sprints")

    async def get_active_sprint(self, org_slug: str, project_id: str) -> dict | None:
        """Fetch sprints and return the active one, if any."""
        sprint_list: list = await self.get_sprints(org_slug, project_id)
        return next((s for s in sprint_list if s.get("status") == "active"), None)

    async def add_issue_to_sprint(
        self, org_slug: str, project_id: str, sprint_id: str, issue_id: str
    ) -> dict:
        return await self.post(
            f"/bot/orgs/{org_slug}/projects/{project_id}/sprints/{sprint_id}/issues/{issue_id}"
        )

    async def get_members(self, org_slug: str) -> list:
        return await self.get(f"/bot/orgs/{org_slug}/members")

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
