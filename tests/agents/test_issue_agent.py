import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.issue_tools import (
    CreateIssueTool,
    FindSimilarIssuesTool,
    GetIssuesTool,
    SuggestAssigneeTool,
    UpdateIssueStatusTool,
    UpdateIssueTool,
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


# --- UpdateIssueTool (priority / title / reassign by issue NUMBER) ---

_UPD_ISSUES = [
    {"number": 5, "id": "issue-uuid-5", "title": "Login broken", "priority": "low"},
    {"number": 6, "id": "issue-uuid-6", "title": "Dark mode", "priority": "medium"},
]


@pytest.mark.asyncio
async def test_update_issue_resolves_number_then_patches_priority():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(return_value={"number": 5, "title": "Login broken"}),
    )
    tool = UpdateIssueTool(api=api, org_slug="acme", project_id="proj-1", user_id="u-1")
    await tool._arun(issue_number=5, priority="high")
    api.patch.assert_called_once_with(
        "/orgs/acme/projects/proj-1/issues/issue-uuid-5",
        {"priority": "high"},
        user_id="u-1",
    )


@pytest.mark.asyncio
async def test_update_issue_sets_title_and_assignee():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(return_value={"number": 5, "title": "New title"}),
    )
    tool = UpdateIssueTool(api=api, **CTX)
    await tool._arun(issue_number=5, title="New title", assignee_id="u-9")
    _, body = api.patch.call_args[0]
    assert body["title"] == "New title"
    assert body["assigneeId"] == "u-9"  # camelCase for the backend schema


@pytest.mark.asyncio
async def test_update_issue_returns_confirmation_with_number():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(return_value={"number": 5, "title": "Login broken"}),
    )
    tool = UpdateIssueTool(api=api, **CTX)
    result = await tool._arun(issue_number=5, priority="high")
    assert "#5" in result


@pytest.mark.asyncio
async def test_update_issue_number_not_found_does_not_patch():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(return_value={}),
    )
    tool = UpdateIssueTool(api=api, **CTX)
    result = await tool._arun(issue_number=99, priority="high")
    assert "#99" in result and "not found" in result.lower()
    api.patch.assert_not_called()


@pytest.mark.asyncio
async def test_update_issue_with_no_fields_does_not_patch():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(return_value={}),
    )
    tool = UpdateIssueTool(api=api, **CTX)
    result = await tool._arun(issue_number=5)
    assert "nothing" in result.lower()
    api.patch.assert_not_called()


@pytest.mark.asyncio
async def test_update_issue_returns_error_string_on_failure():
    api = _mock_api(
        get_issues=AsyncMock(return_value=_UPD_ISSUES),
        patch=AsyncMock(side_effect=Exception("403")),
    )
    tool = UpdateIssueTool(api=api, **CTX)
    result = await tool._arun(issue_number=5, priority="high")
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
async def test_create_issue_uses_availability_message_when_score_zero():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_ZERO_SCORE)):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug")
    assert "Harsh Singh" in result
    assert "available" in result.lower()


@pytest.mark.asyncio
async def test_create_issue_uses_history_message_when_score_positive():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_SUGGESTION)):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug")
    assert "Harsh Singh" in result
    assert "based on past bug issues" in result


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


# --- incremental Neo4j sync (Sprint 2.1) ------------------------------------
#
# Before this change, a bot-created issue lived in Postgres but was
# invisible to find_similar_issues and suggest_assignee until the next
# nightly full_sync. The tool now upserts into Neo4j in the same request
# so smart features are immediately consistent with the source of truth.


@pytest.mark.asyncio
async def test_create_issue_upserts_to_neo4j_after_post() -> None:
    """After a successful POST, the tool must upsert the new issue to
    Neo4j so find_similar_issues can see it immediately."""
    from tools.issue_tools import CreateIssueTool

    neo = MagicMock()
    with patch("tools.issue_tools.upsert_issue", new=AsyncMock()) as mock_upsert:
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        await tool._arun(title="Login crash", type="bug", priority="high")
    mock_upsert.assert_awaited_once()
    args = mock_upsert.await_args.args
    # First arg is the neo4j client, second is the merged issue dict
    # (body + Node response so the upsert has all the fields).
    assert args[0] is neo
    issue = args[1]
    assert issue["title"] == "Login crash"
    assert issue["type"] == "bug"
    assert issue["priority"] == "high"
    assert issue["number"] == 7


@pytest.mark.asyncio
async def test_create_issue_skips_upsert_when_no_neo4j_client() -> None:
    """Without a neo4j_client injection, the tool must not try to call
    graph.sync.upsert_issue — the bot is already running in fail-soft
    mode (main.py) and the issue was still created in Postgres."""
    from tools.issue_tools import CreateIssueTool

    with patch("tools.issue_tools.upsert_issue", new=AsyncMock()) as mock_upsert:
        tool = CreateIssueTool(api=_DEFAULT_API(), **CTX)  # no neo4j_client
        result = await tool._arun(title="t")
    mock_upsert.assert_not_called()
    assert "7" in result   # issue was still created (default API returns #7)


@pytest.mark.asyncio
async def test_create_issue_upsert_failure_does_not_break_user_response() -> None:
    """A Neo4j outage during incremental sync must NOT break the chat —
    log and carry on. The issue was successfully created in Postgres;
    the smart features can wait for the next full_sync."""
    from tools.issue_tools import CreateIssueTool

    neo = MagicMock()
    with patch("tools.issue_tools.upsert_issue", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = CreateIssueTool(api=_DEFAULT_API(), neo4j_client=neo, **CTX)
        result = await tool._arun(title="Login crash", type="bug")
    assert "7" in result  # user-facing message still says "Created #7"


@pytest.mark.asyncio
async def test_update_issue_upserts_to_neo4j_after_patch() -> None:
    """After a successful PATCH, the tool must upsert the updated issue
    so the graph reflects the new priority / title / assignee."""
    from tools.issue_tools import UpdateIssueTool

    neo = MagicMock()
    with patch("tools.issue_tools.upsert_issue", new=AsyncMock()) as mock_upsert:
        tool = UpdateIssueTool(
            api=_mock_api(
                get_issues=AsyncMock(return_value=_UPD_ISSUES),
                patch=AsyncMock(return_value={"number": 5, "title": "Login broken"}),
            ),
            neo4j_client=neo, **CTX,
        )
        await tool._arun(issue_number=5, priority="high")
    mock_upsert.assert_awaited_once()
    args = mock_upsert.await_args.args
    assert args[0] is neo
    # Pre-update row merged with the patch body — the new priority
    # must be on the upsert, not the old "low".
    assert args[1]["priority"] == "high"
    assert args[1]["number"] == 5
    assert args[1]["title"] == "Login broken"


@pytest.mark.asyncio
async def test_update_issue_upsert_failure_does_not_break_user_response() -> None:
    """A Neo4j outage during an update's incremental sync must NOT break
    the chat — same fail-soft rule as create."""
    from tools.issue_tools import UpdateIssueTool

    neo = MagicMock()
    with patch("tools.issue_tools.upsert_issue", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = UpdateIssueTool(
            api=_mock_api(
                get_issues=AsyncMock(return_value=_UPD_ISSUES),
                patch=AsyncMock(return_value={"number": 5}),
            ),
            neo4j_client=neo, **CTX,
        )
        result = await tool._arun(issue_number=5, priority="high")
    assert "#5" in result  # user-facing message still says "Updated #5"


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
async def test_suggest_assignee_tool_uses_availability_message_when_no_history():
    neo = MagicMock()
    with patch("tools.issue_tools.suggest_assignee", new=AsyncMock(return_value=_NEGATIVE_SCORE)):
        tool = SuggestAssigneeTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="fix login", type="bug")
    assert "Harsh Singh" in result
    assert "available" in result.lower()


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
    # Sprint 2.2: the tool now routes through hybrid_similar_issues (the
    # default noop embedding client yields a non-None zero vector).
    neo = MagicMock()
    with patch("tools.issue_tools.hybrid_similar_issues", new=AsyncMock(return_value=_SIMILAR)):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "#2" in result
    assert "Login page broken" in result
    assert "#5" in result


@pytest.mark.asyncio
async def test_find_similar_tool_returns_no_similar_when_empty():
    neo = MagicMock()
    with patch("tools.issue_tools.hybrid_similar_issues", new=AsyncMock(return_value=[])):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "no similar" in result.lower()


@pytest.mark.asyncio
async def test_find_similar_tool_passes_title_and_project():
    neo = MagicMock()
    with patch("tools.issue_tools.hybrid_similar_issues", new=AsyncMock(return_value=[])) as mock_fn:
        tool = FindSimilarIssuesTool(neo4j_client=neo, org_slug="acme", project_id="proj-7")
        await tool._arun(title="login page crash")
    # hybrid_similar_issues(neo4j_client, project_id, title, embedding, k=5)
    assert mock_fn.call_args[0][1] == "proj-7"
    assert mock_fn.call_args[0][2] == "login page crash"


@pytest.mark.asyncio
async def test_find_similar_tool_returns_error_string_on_failure():
    neo = MagicMock()
    with patch("tools.issue_tools.find_similar_issues", new=AsyncMock(side_effect=Exception("neo4j down"))):
        tool = FindSimilarIssuesTool(neo4j_client=neo, **CTX)
        result = await tool._arun(title="login page crash")
    assert "Failed" in result
