from typing import Any

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


class _NoInput(BaseModel):
    pass


class GetStatusesTool(BaseTool):
    name: str = "get_statuses"
    description: str = "List workflow statuses for the project. Returns id and name of each status. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=get_statuses org={} project={}", self.org_slug, self.project_id)
        try:
            statuses: list = await self.api.get(
                f"/bot/orgs/{self.org_slug}/projects/{self.project_id}/statuses"
            )
            if not statuses:
                return "No statuses found."
            lines = [f"{s['id']} — {s['name']}{' (default)' if s.get('isDefault') else ''}" for s in statuses]
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=get_statuses failed: {}", e)
            return f"Failed to get statuses: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class GetIssuesTool(BaseTool):
    name: str = "get_issues"
    description: str = "List all issues in the current project. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=get_issues org={} project={}", self.org_slug, self.project_id)
        try:
            issues: list = await self.api.get_issues(self.org_slug, self.project_id)
            if not issues:
                return "No issues found."
            lines = [
                f"#{i['number']} [{i['priority']}] {i['title']} — {i.get('status', {}).get('name', 'No status')}"
                for i in issues
            ]
            logger.info("tool=get_issues returned {} issues", len(issues))
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=get_issues failed: {}", e)
            return f"Failed to get issues: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class _CreateIssueInput(BaseModel):
    title: str = Field(description="Issue title")
    type: str = Field(default="task", description="task|bug|story|epic")
    priority: str = Field(default="medium", description="low|medium|high|critical")
    assignee_id: str | None = Field(default=None, description="Assignee user ID")
    status_id: str | None = Field(default=None, description="Status ID — omit to use project default")


class CreateIssueTool(BaseTool):
    name: str = "create_issue"
    description: str = "Create an issue. Args: title, type(task/bug/story/epic), priority(low/medium/high/critical), assignee_id?, status_id?."
    args_schema: type[BaseModel] = _CreateIssueInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None

    async def _arun(
        self,
        title: str,
        type: str = "task",
        priority: str = "medium",
        assignee_id: str | None = None,
        status_id: str | None = None,
    ) -> str:
        logger.info("tool=create_issue title='{}' type={} priority={}", title, type, priority)
        try:
            resolved_status_id = status_id
            if not resolved_status_id:
                default = await self.api.get_default_status(self.org_slug, self.project_id)
                if not default:
                    return "Failed to create issue: no statuses configured for this project"
                resolved_status_id = default["id"]
                logger.debug("tool=create_issue default status='{}' id={}", default["name"], resolved_status_id)

            body: dict = {
                "title": title,
                "type": type,
                "priority": priority,
                "statusId": resolved_status_id,
            }
            if assignee_id:
                body["assigneeId"] = assignee_id

            result = await self.api.post(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/issues",
                body,
                user_id=self.user_id,
            )
            logger.info("tool=create_issue created #{}", result.get("number"))
            return f"Created #{result['number']}: {result['title']}"
        except Exception as e:
            logger.error("tool=create_issue failed: {}", e)
            return f"Failed to create issue: {e}"

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")


class _UpdateIssueStatusInput(BaseModel):
    issue_id: str = Field(description="Issue ID")
    status_id: str = Field(description="New status ID")


class UpdateIssueStatusTool(BaseTool):
    name: str = "update_issue_status"
    description: str = "Update an issue's status. Args: issue_id, status_id."
    args_schema: type[BaseModel] = _UpdateIssueStatusInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None

    async def _arun(self, issue_id: str, status_id: str) -> str:
        logger.info("tool=update_issue_status issue_id={} status_id={}", issue_id, status_id)
        try:
            await self.api.patch(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/issues/{issue_id}/status",
                {"statusId": status_id},
                user_id=self.user_id,
            )
            logger.info("tool=update_issue_status updated {}", issue_id)
            return f"Updated issue {issue_id} status."
        except Exception as e:
            logger.error("tool=update_issue_status failed: {}", e)
            return f"Failed to update issue: {e}"

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")
