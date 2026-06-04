"""Tests for FindUserByNameTool — the bridge between the model's
"assign to Alice" and the user_id that create_issue requires."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.issue_tools import FindUserByNameTool

CTX = {"org_slug": "acme", "project_id": "proj-1"}


def _mock_neo(*call_returns):
    m = MagicMock()
    m.run = AsyncMock(side_effect=list(call_returns))
    return m


# --- happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_formatted_id_name_email_list() -> None:
    neo = _mock_neo([
        {"id": "u1", "name": "Harsh Singh", "email": "harsh@acme.com"},
    ])
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="Harsh")
    assert "id=u1" in result
    assert "name=Harsh Singh" in result
    assert "email=harsh@acme.com" in result


@pytest.mark.asyncio
async def test_lists_multiple_matches() -> None:
    """When the substring matches several members, all are returned so the
    model can disambiguate (or ask the user)."""
    neo = _mock_neo([
        {"id": "u1", "name": "Alice Smith",  "email": "alice@acme.com"},
        {"id": "u2", "name": "Alicia Jones", "email": "alicia@acme.com"},
    ])
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="ali")
    assert "Alice Smith" in result
    assert "Alicia Jones" in result


# --- empty / no-match --------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_string_returns_no_match_message() -> None:
    neo = _mock_neo([])
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="")
    assert "no project member" in result.lower()


@pytest.mark.asyncio
async def test_whitespace_query_returns_no_match() -> None:
    neo = _mock_neo([])
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="   ")
    assert "no project member" in result.lower()


@pytest.mark.asyncio
async def test_neo4j_returns_empty_list_returns_no_match_message() -> None:
    neo = _mock_neo([])
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="Zzznobody")
    assert "no project member matching 'zzznobody'" in result.lower()


# --- error path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_neo4j_failure_returns_error_string() -> None:
    neo = MagicMock()
    neo.run = AsyncMock(side_effect=Exception("neo4j down"))
    tool = FindUserByNameTool(neo4j_client=neo, **CTX)
    result = await tool._arun(name="Alice")
    assert "Failed" in result


# --- agent wiring ------------------------------------------------------------


def test_find_user_by_name_tool_is_in_issue_agent_tool_list() -> None:
    """Regression guard: the tool must be registered or the model has no way
    to resolve names. We check the source string rather than instantiating
    the agent graph (which would build real LangChain objects)."""
    import inspect
    from agents import issue_agent
    src = inspect.getsource(issue_agent._build_agent)
    assert "FindUserByNameTool" in src
    assert "FindUserByNameTool(**graph_ctx)" in src


def test_issue_agent_system_prompt_instructs_name_resolution() -> None:
    """The system prompt must tell the model to resolve names BEFORE calling
    create_issue — otherwise the LLM may pass the literal name."""
    import inspect
    from agents import issue_agent
    src = inspect.getsource(issue_agent)
    # _SYSTEM constant must mention find_user_by_name AND forbid raw names.
    assert "find_user_by_name" in src.lower()
    assert "assignee_id" in src or "assignee" in src
    # Specifically: a negative instruction about names-as-ids.
    assert ("never pass" in src.lower() or "do not pass" in src.lower())
