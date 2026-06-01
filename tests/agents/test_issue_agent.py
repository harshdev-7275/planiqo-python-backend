import pytest
from unittest.mock import AsyncMock, MagicMock

from tools.issue_tools import CreateIssueTool, GetIssuesTool, UpdateIssueStatusTool

CTX = {"org_slug": "acme", "project_id": "proj-1"}


def _mock_api(**kwargs):
    api = MagicMock()
    for method, mock in kwargs.items():
        setattr(api, method, mock)
    return api


# --- GetIssuesTool ---

@pytest.mark.asyncio
async def test_get_issues_returns_formatted_list():
    api = _mock_api(get_issues=AsyncMock(return_value=[
        {"number": 1, "title": "Fix login bug", "priority": "high", "status": {"name": "In Progress"}},
        {"number": 2, "title": "Add dark mode", "priority": "low", "status": {"name": "Todo"}},
    ]))
    tool = GetIssuesTool(api=api, **CTX)
    result = await tool._arun()

    assert "#1" in result
    assert "Fix login bug" in result
    assert "high" in result
    assert "#2" in result
    assert "Add dark mode" in result


@pytest.mark.asyncio
async def test_get_issues_empty_returns_message():
    api = _mock_api(get_issues=AsyncMock(return_value=[]))
    tool = GetIssuesTool(api=api, **CTX)
    result = await tool._arun()
    assert result == "No issues found."


@pytest.mark.asyncio
async def test_get_issues_calls_semantic_method():
    api = _mock_api(get_issues=AsyncMock(return_value=[]))
    tool = GetIssuesTool(api=api, org_slug="acme", project_id="proj-42")
    await tool._arun()
    api.get_issues.assert_called_once_with("acme", "proj-42")


@pytest.mark.asyncio
async def test_get_issues_returns_error_string_on_failure():
    api = _mock_api(get_issues=AsyncMock(side_effect=Exception("timeout")))
    tool = GetIssuesTool(api=api, **CTX)
    result = await tool._arun()
    assert "Failed" in result


# --- CreateIssueTool ---

@pytest.mark.asyncio
async def test_create_issue_returns_confirmation():
    api = _mock_api(
        get_default_status=AsyncMock(return_value={"id": "status-1", "name": "Todo", "isDefault": True}),
        post=AsyncMock(return_value={"number": 42, "title": "Fix login bug"}),
    )
    tool = CreateIssueTool(api=api, **CTX)
    result = await tool._arun(title="Fix login bug", type="bug", priority="high")
    assert "42" in result
    assert "Fix login bug" in result


@pytest.mark.asyncio
async def test_create_issue_uses_default_status_when_none_provided():
    api = _mock_api(
        get_default_status=AsyncMock(return_value={"id": "status-1", "name": "Todo", "isDefault": True}),
        post=AsyncMock(return_value={"number": 1, "title": "Test"}),
    )
    tool = CreateIssueTool(api=api, **CTX)
    await tool._arun(title="Test", type="task", priority="medium")
    api.get_default_status.assert_called_once_with("acme", "proj-1")
    api.post.assert_called_once_with(
        "/orgs/acme/projects/proj-1/issues",
        {"title": "Test", "type": "task", "priority": "medium", "statusId": "status-1"},
        user_id=None,
    )


@pytest.mark.asyncio
async def test_create_issue_skips_default_lookup_when_status_provided():
    api = _mock_api(
        post=AsyncMock(return_value={"number": 1, "title": "Test"}),
    )
    tool = CreateIssueTool(api=api, **CTX)
    await tool._arun(title="Test", status_id="explicit-status")
    api.post.assert_called_once()
    assert "get_default_status" not in dir(api) or not api.get_default_status.called


@pytest.mark.asyncio
async def test_create_issue_includes_optional_fields():
    api = _mock_api(
        get_default_status=AsyncMock(return_value={"id": "status-1", "name": "Todo", "isDefault": True}),
        post=AsyncMock(return_value={"number": 5, "title": "Test"}),
    )
    tool = CreateIssueTool(api=api, **CTX)
    await tool._arun(title="Test", assignee_id="user-99")
    _, call_body = api.post.call_args[0]
    assert call_body["assigneeId"] == "user-99"
    assert call_body["statusId"] == "status-1"


@pytest.mark.asyncio
async def test_create_issue_returns_error_string_on_failure():
    api = _mock_api(
        get_default_status=AsyncMock(return_value={"id": "status-1", "name": "Todo"}),
        post=AsyncMock(side_effect=Exception("500")),
    )
    tool = CreateIssueTool(api=api, **CTX)
    result = await tool._arun(title="Test")
    assert "Failed" in result


@pytest.mark.asyncio
async def test_create_issue_handles_no_statuses():
    api = _mock_api(get_default_status=AsyncMock(return_value=None))
    tool = CreateIssueTool(api=api, **CTX)
    result = await tool._arun(title="Test")
    assert "no statuses" in result


# --- UpdateIssueStatusTool ---

@pytest.mark.asyncio
async def test_update_issue_status_returns_confirmation():
    api = _mock_api(patch=AsyncMock(return_value={}))
    tool = UpdateIssueStatusTool(api=api, **CTX)
    result = await tool._arun(issue_id="issue-7", status_id="status-done")
    assert "issue-7" in result


@pytest.mark.asyncio
async def test_update_issue_status_calls_correct_path():
    api = _mock_api(patch=AsyncMock(return_value={}))
    tool = UpdateIssueStatusTool(api=api, **CTX)
    await tool._arun(issue_id="issue-7", status_id="status-done")
    api.patch.assert_called_once_with(
        "/orgs/acme/projects/proj-1/issues/issue-7/status",
        {"statusId": "status-done"},
        user_id=None,
    )


@pytest.mark.asyncio
async def test_update_issue_status_returns_error_string_on_failure():
    api = _mock_api(patch=AsyncMock(side_effect=Exception("404")))
    tool = UpdateIssueStatusTool(api=api, **CTX)
    result = await tool._arun(issue_id="issue-7", status_id="status-done")
    assert "Failed" in result
