import pytest
from unittest.mock import AsyncMock, patch

from models.intents import Intent, IntentResult

BASE_ARGS = {"user_id": "u1", "org_slug": "acme", "project_id": None}


def _intent(intent: Intent) -> IntentResult:
    return IntentResult(intent=intent, confidence=0.95, entities={})


_AGENT_STUB = AsyncMock(return_value={"result": {"message": "stub"}})


async def _run(message: str, intent: Intent) -> dict:
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(intent))),
        patch("agents.issue_agent.run",  new=_AGENT_STUB),
        patch("agents.sprint_agent.run", new=_AGENT_STUB),
    ):
        from agents.supervisor import run
        return await run(message=message, **BASE_ARGS)


@pytest.mark.asyncio
async def test_create_issue_routes_to_issue_agent():
    result = await _run("create a bug for login", Intent.CREATE_ISSUE)
    assert result["intent"] == "CREATE_ISSUE"
    assert result["result"] is not None


@pytest.mark.asyncio
async def test_query_issues_routes_to_issue_agent():
    result = await _run("show me open issues", Intent.QUERY_ISSUES)
    assert result["intent"] == "QUERY_ISSUES"


@pytest.mark.asyncio
async def test_update_issue_routes_to_issue_agent():
    result = await _run("mark issue #42 as done", Intent.UPDATE_ISSUE)
    assert result["intent"] == "UPDATE_ISSUE"


@pytest.mark.asyncio
async def test_query_sprint_routes_to_sprint_agent():
    result = await _run("show me sprint status", Intent.QUERY_SPRINT)
    assert result["intent"] == "QUERY_SPRINT"


@pytest.mark.asyncio
async def test_create_sprint_routes_to_sprint_agent():
    result = await _run("create a new sprint", Intent.CREATE_SPRINT)
    assert result["intent"] == "CREATE_SPRINT"


@pytest.mark.asyncio
async def test_query_member_routes_to_member_agent():
    result = await _run("what is john working on", Intent.QUERY_MEMBER)
    assert result["intent"] == "QUERY_MEMBER"


@pytest.mark.asyncio
async def test_summarize_routes_to_summarize_agent():
    result = await _run("summarize this week", Intent.SUMMARIZE)
    assert result["intent"] == "SUMMARIZE"


@pytest.mark.asyncio
async def test_teams_context_routes_to_teams_agent():
    result = await _run("who is on the platform team", Intent.TEAMS_CONTEXT)
    assert result["intent"] == "TEAMS_CONTEXT"


@pytest.mark.asyncio
async def test_unknown_returns_helpful_message():
    result = await _run("", Intent.UNKNOWN)
    assert result["intent"] == "UNKNOWN"
    assert result["result"] is not None
    assert "message" in result["result"]
