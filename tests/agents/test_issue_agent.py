import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.issue_tools import (
    CreateIssueTool,
    FindSimilarIssuesTool,
    GetIssuesTool,
    SuggestAssigneeTool,
    UpdateIssueStatusTool,
)

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


# --- CreateIssueTool — suggest_assignee integration ---

_SUGGESTION = [{"userId": "u1", "userName": "Harsh Singh", "score": 9.0}]
_NO_SUGGESTION = []
_ZERO_SCORE = [{"userId": "u1", "userName": "Harsh Singh", "score": 0.0}]
_NEGATIVE_SCORE = [{"userId": "u1", "userName": "Harsh Singh", "score": -1.0}]

def _DEFAULT_API():
    return _mock_api(
        get_default_status=AsyncMock(return_value={"id": "s1", "name": "Todo"}),
        post=AsyncMock(return_value={"number": 7, "title": "Login crash"}),
    )


@pytest.mark.asyncio
async def test_create_issue_appends_suggestion_when_score_positive():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug", priority="high")
    assert "Harsh Singh" in result
    assert "7" in result


@pytest.mark.asyncio
async def test_create_issue_suggestion_only_when_no_explicit_assignee():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)) as mock_fn:
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        await tool._arun(title="Login crash", type="bug", assignee_id="u99")
    mock_fn.assert_not_called()


@pytest.mark.asyncio
async def test_create_issue_no_suggestion_appended_when_score_zero():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_ZERO_SCORE)):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug")
    assert "Harsh Singh" not in result


@pytest.mark.asyncio
async def test_create_issue_no_suggestion_when_no_neo4j_client():
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)) as mock_fn:
        tool = CreateIssueTool(api=_DEFAULT_API(), **CTX)   # no neo4j_client
        result = await tool._arun(title="Login crash", type="bug")
    mock_fn.assert_not_called()
    assert "7" in result   # issue still created fine


@pytest.mark.asyncio
async def test_create_issue_suggestion_failure_does_not_crash_tool():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug")
    assert "7" in result   # issue created despite suggestion failure


# --- SuggestAssigneeTool ---

@pytest.mark.asyncio
async def test_suggest_assignee_tool_returns_formatted_suggestion():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)):
        tool = SuggestAssigneeTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="fix login", type="bug")
    assert "Harsh Singh" in result


@pytest.mark.asyncio
async def test_suggest_assignee_tool_no_data_when_empty():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_NO_SUGGESTION)):
        tool = SuggestAssigneeTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="fix login", type="bug")
    assert "not enough" in result.lower() or "no suggestion" in result.lower()


@pytest.mark.asyncio
async def test_suggest_assignee_tool_no_data_when_score_not_positive():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_NEGATIVE_SCORE)):
        tool = SuggestAssigneeTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="fix login", type="bug")
    assert "Harsh Singh" not in result


@pytest.mark.asyncio
async def test_suggest_assignee_tool_passes_project_id_and_type():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)) as mock_fn:
        tool = SuggestAssigneeTool(neo4j_client=neo, org_slug="acme", project_id="proj-42")
        await tool._arun(title="fix login", type="story")
    _, call_kwargs = mock_fn.call_args
    assert mock_fn.call_args[0][1] == "proj-42"
    assert mock_fn.call_args[0][3] == "story"


@pytest.mark.asyncio
async def test_suggest_assignee_tool_returns_error_string_on_failure():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = SuggestAssigneeTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="fix login", type="bug")
    assert "Failed" in result


# --- FindSimilarIssuesTool ---

_SIMILAR = [
    {"id": "i1", "number": 2, "title": "Login page broken",  "match_count": 2},
    {"id": "i2", "number": 5, "title": "Auth crash on login", "match_count": 1},
]


@pytest.mark.asyncio
async def test_find_similar_tool_returns_formatted_list():
    neo = MagicMock()
    with patch("tools.issue_tools.find_similar_issues", new=AsyncMock(return_value=_SIMILAR)):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "#2" in result
    assert "Login page broken" in result
    assert "#5" in result


@pytest.mark.asyncio
async def test_find_similar_tool_returns_no_similar_when_empty():
    neo = MagicMock()
    with patch("tools.issue_tools.find_similar_issues", new=AsyncMock(return_value=[])):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "no similar" in result.lower()


@pytest.mark.asyncio
async def test_find_similar_tool_passes_title_and_project():
    neo = MagicMock()
    with patch("tools.issue_tools.find_similar_issues", new=AsyncMock(return_value=[])) as mock_fn:
        tool = FindSimilarIssuesTool(neo4j_client=neo, org_slug="acme", project_id="proj-7")
        await tool._arun(title="login page crash")
    assert mock_fn.call_args[0][1] == "proj-7"
    assert mock_fn.call_args[0][2] == "login page crash"


@pytest.mark.asyncio
async def test_find_similar_tool_returns_error_string_on_failure():
    neo = MagicMock()
    with patch("tools.issue_tools.find_similar_issues", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "Failed" in result
