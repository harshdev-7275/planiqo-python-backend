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
            sprint_list: list = await self.api.get_sprints(self.org_slug, self.project_id)
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

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")
