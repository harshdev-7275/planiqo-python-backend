from typing import Any

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from graph.queries import find_similar_issues, suggest_assignee


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
    neo4j_client: Any | None = None

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
            response = f"Created #{result['number']}: {result['title']}"

            if not self.neo4j_client:
                logger.debug("tool=create_issue suggestion skipped: no neo4j_client injected")
            elif assignee_id:
                logger.debug("tool=create_issue suggestion skipped: explicit assignee_id={}", assignee_id)
            else:
                try:
                    suggestions = await suggest_assignee(
                        self.neo4j_client, self.project_id, title, type
                    )
                    logger.debug(
                        "tool=create_issue suggest_assignee returned {} candidates",
                        len(suggestions),
                    )
                    if not suggestions:
                        logger.info(
                            "tool=create_issue no suggestion: empty result — "
                            "no project members in graph, run POST /graph/sync"
                        )
                    else:
                        top = suggestions[0]
                        reason = (
                            f"based on past {type} issues"
                            if top["score"] > 0
                            else "most available team member"
                        )
                        response += (
                            f"\n\U0001f4a1 Suggested assignee: {top['userName']} ({reason})"
                        )
                        logger.info(
                            "tool=create_issue suggested user='{}' score={:.1f} reason='{}'",
                            top["userName"], top["score"], reason,
                        )
                except Exception as e:
                    logger.error(
                        "tool=create_issue suggestion failed (neo4j error): {} — "
                        "issue was still created successfully",
                        e, exc_info=True,
                    )

            return response
        except Exception as e:
            logger.error("tool=create_issue failed: {}", e)
            return f"Failed to create issue: {e}"

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")


class _SuggestAssigneeInput(BaseModel):
    title: str = Field(description="Issue title")
    type: str = Field(default="task", description="task|bug|story|epic")


class SuggestAssigneeTool(BaseTool):
    name: str = "suggest_assignee"
    description: str = "Suggest the best person to assign a new issue. Args: title, type(task/bug/story/epic)."
    args_schema: type[BaseModel] = _SuggestAssigneeInput

    neo4j_client: Any
    org_slug: str
    project_id: str

    async def _arun(self, title: str, type: str = "task") -> str:
        logger.info("tool=suggest_assignee title='{}' type={}", title, type)
        try:
            suggestions = await suggest_assignee(self.neo4j_client, self.project_id, title, type)
            if not suggestions:
                return "No suggestion available — no project members found in graph (run /graph/sync)."
            top = suggestions[0]
            if top["score"] > 0:
                return (
                    f"Suggested assignee: {top['userName']}"
                    f" (score: {top['score']:.0f}, based on past {type} issues)"
                )
            return (
                f"Suggested assignee: {top['userName']}"
                f" (most available team member)"
            )
        except Exception as e:
            logger.error("tool=suggest_assignee failed: {}", e)
            return f"Failed to suggest assignee: {e}"

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")


class _FindSimilarInput(BaseModel):
    title: str = Field(description="Issue title to search for similar issues")


class FindSimilarIssuesTool(BaseTool):
    name: str = "find_similar_issues"
    description: str = "Find issues similar to a given title. Args: title."
    args_schema: type[BaseModel] = _FindSimilarInput

    neo4j_client: Any
    org_slug: str
    project_id: str

    async def _arun(self, title: str) -> str:
        logger.info("tool=find_similar_issues title='{}'", title)
        try:
            similar = await find_similar_issues(self.neo4j_client, self.project_id, title)
            if not similar:
                return "No similar issues found."
            lines = [f"#{s['number']}: {s['title']}" for s in similar]
            return "Similar issues found:\n" + "\n".join(lines)
        except Exception as e:
            logger.error("tool=find_similar_issues failed: {}", e)
            return f"Failed to find similar issues: {e}"

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
