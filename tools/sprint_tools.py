from typing import Any

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


class _NoInput(BaseModel):
    pass


class GetSprintsTool(BaseTool):
    name: str = "get_sprints"
    description: str = "List all sprints. Returns name, status, start/end dates. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=get_sprints org={} project={}", self.org_slug, self.project_id)
        try:
            sprint_list: list[dict[str, Any]] = await self.api.get_sprints(self.org_slug, self.project_id)
            if not sprint_list:
                return "No sprints found."
            lines = [
                f"{s['name']} — {s['status']} | {s.get('startDate') or 'no date'} → {s.get('endDate') or 'no date'}"
                for s in sprint_list
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=get_sprints failed: {}", e)
            return f"Failed to get sprints: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class GetActiveSprintTool(BaseTool):
    name: str = "get_active_sprint"
    description: str = "Get the currently active sprint. Returns details or 'No active sprint'. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=get_active_sprint org={} project={}", self.org_slug, self.project_id)
        try:
            sprint = await self.api.get_active_sprint(self.org_slug, self.project_id)
            if not sprint:
                return "No active sprint."
            return (
                f"{sprint['name']} — {sprint['status']} | "
                f"{sprint.get('startDate') or 'no date'} → {sprint.get('endDate') or 'no date'}"
            )
        except Exception as e:
            logger.error("tool=get_active_sprint failed: {}", e)
            return f"Failed to get active sprint: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class _CreateSprintInput(BaseModel):
    name: str = Field(description="Sprint name")
    goal: str | None = Field(default=None, description="Optional sprint goal")


class CreateSprintTool(BaseTool):
    name: str = "create_sprint"
    description: str = "Create a sprint. Args: name, goal?."
    args_schema: type[BaseModel] = _CreateSprintInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None

    async def _arun(self, name: str, goal: str | None = None) -> str:
        logger.info("tool=create_sprint name='{}' project={}", name, self.project_id)
        try:
            body: dict[str, Any] = {"name": name}
            if goal:
                body["goal"] = goal
            result = await self.api.post(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/sprints",
                body,
                user_id=self.user_id,
            )
            created = result.get("name", name)
            logger.info("tool=create_sprint created '{}'", created)
            return f"Created sprint '{created}'."
        except Exception as e:
            logger.error("tool=create_sprint failed: {}", e)
            return f"Failed to create sprint: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _AddIssueInput(BaseModel):
    sprint_id: str = Field(description="Sprint ID")
    issue_id: str = Field(description="Issue ID")


class AddIssueToSprintTool(BaseTool):
    name: str = "add_issue_to_sprint"
    description: str = "Add an issue to a sprint. Args: sprint_id, issue_id."
    args_schema: type[BaseModel] = _AddIssueInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self, sprint_id: str, issue_id: str) -> str:
        logger.info("tool=add_issue_to_sprint sprint={} issue={}", sprint_id, issue_id)
        try:
            await self.api.add_issue_to_sprint(self.org_slug, self.project_id, sprint_id, issue_id)
            return f"Added issue {issue_id} to sprint {sprint_id}."
        except Exception as e:
            logger.error("tool=add_issue_to_sprint failed: {}", e)
            return f"Failed to add issue to sprint: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")
