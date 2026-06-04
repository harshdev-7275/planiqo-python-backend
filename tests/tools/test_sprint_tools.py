import pytest
from unittest.mock import AsyncMock, MagicMock

from tools.sprint_tools import (
    AddIssueToSprintTool,
    CreateSprintTool,
    GetActiveSprintTool,
    GetSprintsTool,
)

CTX = {"org_slug": "acme", "project_id": "proj-1"}


def _mock_api(**kwargs):
    api = MagicMock()
    for method, mock in kwargs.items():
        setattr(api, method, mock)
    return api


_SPRINTS = [
    {"id": "s1", "name": "Sprint 1", "status": "completed", "startDate": "2026-05-01T00:00:00Z", "endDate": "2026-05-14T00:00:00Z"},
    {"id": "s2", "name": "Sprint 2", "status": "active",    "startDate": "2026-05-15T00:00:00Z", "endDate": "2026-05-28T00:00:00Z"},
    {"id": "s3", "name": "Sprint 3", "status": "planned",   "startDate": None,                    "endDate": None},
]


# --- GetSprintsTool ---

@pytest.mark.asyncio
async def test_get_sprints_returns_formatted_list():
    api = _mock_api(get_sprints=AsyncMock(return_value=_SPRINTS))
    tool = GetSprintsTool(api=api, **CTX)
    result = await tool._arun()

    assert "Sprint 1" in result
    assert "completed" in result
    assert "Sprint 2" in result
    assert "active" in result
    assert "Sprint 3" in result
    assert "planned" in result


@pytest.mark.asyncio
async def test_get_sprints_empty_returns_message():
    api = _mock_api(get_sprints=AsyncMock(return_value=[]))
    tool = GetSprintsTool(api=api, **CTX)
    result = await tool._arun()
    assert result == "No sprints found."


@pytest.mark.asyncio
async def test_get_sprints_calls_semantic_method():
    api = _mock_api(get_sprints=AsyncMock(return_value=[]))
    tool = GetSprintsTool(api=api, org_slug="acme", project_id="proj-99")
    await tool._arun()
    api.get_sprints.assert_called_once_with("acme", "proj-99")


@pytest.mark.asyncio
async def test_get_sprints_returns_error_string_on_failure():
    api = _mock_api(get_sprints=AsyncMock(side_effect=Exception("timeout")))
    tool = GetSprintsTool(api=api, **CTX)
    result = await tool._arun()
    assert "Failed" in result


# --- GetActiveSprintTool ---

@pytest.mark.asyncio
async def test_get_active_sprint_returns_formatted_sprint():
    active = {"id": "s2", "name": "Sprint 2", "status": "active", "startDate": "2026-05-15T00:00:00Z", "endDate": "2026-05-28T00:00:00Z"}
    api = _mock_api(get_active_sprint=AsyncMock(return_value=active))
    tool = GetActiveSprintTool(api=api, **CTX)
    result = await tool._arun()

    assert "Sprint 2" in result
    assert "active" in result
    assert "2026-05-15" in result


@pytest.mark.asyncio
async def test_get_active_sprint_returns_no_active_message():
    api = _mock_api(get_active_sprint=AsyncMock(return_value=None))
    tool = GetActiveSprintTool(api=api, **CTX)
    result = await tool._arun()
    assert "No active sprint" in result


@pytest.mark.asyncio
async def test_get_active_sprint_calls_api_method():
    api = _mock_api(get_active_sprint=AsyncMock(return_value=None))
    tool = GetActiveSprintTool(api=api, org_slug="acme", project_id="proj-42")
    await tool._arun()
    api.get_active_sprint.assert_called_once_with("acme", "proj-42")


@pytest.mark.asyncio
async def test_get_active_sprint_returns_error_string_on_failure():
    api = _mock_api(get_active_sprint=AsyncMock(side_effect=Exception("503")))
    tool = GetActiveSprintTool(api=api, **CTX)
    result = await tool._arun()
    assert "Failed" in result


# --- AddIssueToSprintTool ---

@pytest.mark.asyncio
async def test_add_issue_to_sprint_returns_confirmation():
    api = _mock_api(add_issue_to_sprint=AsyncMock(return_value={}))
    tool = AddIssueToSprintTool(api=api, **CTX)
    result = await tool._arun(sprint_id="s2", issue_id="issue-7")
    assert "issue-7" in result
    assert "s2" in result


@pytest.mark.asyncio
async def test_add_issue_to_sprint_calls_correct_method():
    api = _mock_api(add_issue_to_sprint=AsyncMock(return_value={}))
    tool = AddIssueToSprintTool(api=api, **CTX)
    await tool._arun(sprint_id="s2", issue_id="issue-7")
    api.add_issue_to_sprint.assert_called_once_with("acme", "proj-1", "s2", "issue-7")


@pytest.mark.asyncio
async def test_add_issue_to_sprint_returns_error_string_on_failure():
    api = _mock_api(add_issue_to_sprint=AsyncMock(side_effect=Exception("404")))
    tool = AddIssueToSprintTool(api=api, **CTX)
    result = await tool._arun(sprint_id="s2", issue_id="issue-7")
    assert "Failed" in result


# --- CreateSprintTool ---

@pytest.mark.asyncio
async def test_create_sprint_returns_confirmation():
    api = _mock_api(post=AsyncMock(return_value={"id": "s9", "name": "Sprint 7"}))
    tool = CreateSprintTool(api=api, **CTX)
    result = await tool._arun(name="Sprint 7")
    assert "Sprint 7" in result


@pytest.mark.asyncio
async def test_create_sprint_posts_to_correct_path_with_acting_user():
    api = _mock_api(post=AsyncMock(return_value={"id": "s9", "name": "Sprint 7"}))
    tool = CreateSprintTool(api=api, org_slug="acme", project_id="proj-1", user_id="u-42")
    await tool._arun(name="Sprint 7")
    api.post.assert_called_once_with(
        "/orgs/acme/projects/proj-1/sprints",
        {"name": "Sprint 7"},
        user_id="u-42",
    )


@pytest.mark.asyncio
async def test_create_sprint_includes_goal_when_provided():
    api = _mock_api(post=AsyncMock(return_value={"id": "s9", "name": "Sprint 7"}))
    tool = CreateSprintTool(api=api, **CTX)
    await tool._arun(name="Sprint 7", goal="Ship checkout")
    _, body = api.post.call_args[0]
    assert body["goal"] == "Ship checkout"


@pytest.mark.asyncio
async def test_create_sprint_omits_goal_when_absent():
    api = _mock_api(post=AsyncMock(return_value={"id": "s9", "name": "Sprint 7"}))
    tool = CreateSprintTool(api=api, **CTX)
    await tool._arun(name="Sprint 7")
    _, body = api.post.call_args[0]
    assert "goal" not in body


@pytest.mark.asyncio
async def test_create_sprint_returns_error_string_on_failure():
    api = _mock_api(post=AsyncMock(side_effect=Exception("403")))
    tool = CreateSprintTool(api=api, **CTX)
    result = await tool._arun(name="Sprint 7")
    assert "Failed" in result
