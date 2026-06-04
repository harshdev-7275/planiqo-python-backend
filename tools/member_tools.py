"""Read-only tools for team-member queries (QUERY_MEMBER intent).

Deliberately graph-free: member resolution and assigned-issue filtering both
go through the Node API, so these keep working when Neo4j is down (the graph
powers smart-assignee/similarity, which degrade independently).
"""

from typing import Any

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


class _NoInput(BaseModel):
    pass


def _member_matches(member: dict[str, Any], query: str) -> bool:
    """Case-insensitive substring match on name OR email — 'Alice' matches
    'Alice Smith', 'bob@acme.com' matches by email."""
    q = query.strip().lower()
    name = (member.get("name") or "").lower()
    email = (member.get("email") or "").lower()
    return bool(q) and (q in name or q in email)


def _display_name(member: dict[str, Any]) -> str:
    return str(member.get("name") or member.get("userId") or "?")


class ListMembersTool(BaseTool):
    name: str = "list_members"
    description: str = "List the people on this project with their role. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=list_members org={} project={}", self.org_slug, self.project_id)
        try:
            members: list[dict[str, Any]] = await self.api.get_project_members(self.org_slug, self.project_id)
            if not members:
                return "No members found in this project."
            lines = [
                f"{m.get('name') or m.get('userId')} — {m.get('role', 'member')}"
                f" ({m.get('email', 'no email')})"
                for m in members
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=list_members failed: {}", e)
            return f"Failed to list members: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class _MemberIssuesInput(BaseModel):
    name: str = Field(description="Team member's name or email, e.g. 'Alice'")


class MemberIssuesTool(BaseTool):
    """List the issues currently assigned to one team member.

    Resolves the name to a member (by name/email), then filters the project's
    issues by that member's userId == issue.assigneeId — no issue UUID or
    graph lookup needed.
    """

    name: str = "member_issues"
    description: str = (
        "List the issues assigned to a team member. Args: name (e.g. 'Alice')."
    )
    args_schema: type[BaseModel] = _MemberIssuesInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self, name: str) -> str:
        logger.info("tool=member_issues name='{}' project={}", name, self.project_id)
        try:
            members: list[dict[str, Any]] = await self.api.get_project_members(self.org_slug, self.project_id)
            matches = [m for m in members if _member_matches(m, name)]

            if not matches:
                available = ", ".join(_display_name(m) for m in members)
                msg = f"No team member named '{name}' found in this project."
                if available:
                    msg += f" Team members: {available}."
                return msg

            if len(matches) > 1:
                names = ", ".join(_display_name(m) for m in matches)
                return (
                    f"Multiple members match '{name}': {names}. "
                    "Please be more specific (full name or email)."
                )

            member = matches[0]
            uid = member.get("userId")
            display = member.get("name") or uid

            issues: list[dict[str, Any]] = await self.api.get_issues(self.org_slug, self.project_id)
            theirs = [i for i in issues if i.get("assigneeId") == uid]
            if not theirs:
                return f"{display} has no assigned issues."

            lines = [
                f"#{i['number']} [{i.get('priority', '?')}] {i['title']}"
                f" — {i.get('status', {}).get('name', 'No status')}"
                for i in theirs
            ]
            return f"{display} is assigned {len(theirs)} issue(s):\n" + "\n".join(lines)
        except Exception as e:
            logger.error("tool=member_issues failed: {}", e)
            return f"Failed to get member issues: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")
