import pytest
from unittest.mock import AsyncMock, MagicMock

from graph.sync import (
    full_sync,
    upsert_change_activity,
    upsert_comment_activity,
    upsert_issue,
    upsert_member,
    upsert_project,
    upsert_user,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neo4j():
    m = MagicMock()
    m.run = AsyncMock(return_value=[])
    return m


def _api(**kwargs):
    m = MagicMock()
    for method, value in kwargs.items():
        setattr(m, method, AsyncMock(return_value=value))
    return m


_USER = {"userId": "u1", "name": "Harsh", "email": "harsh@acme.com", "avatarUrl": "https://example.com/a.png", "role": "member"}
_PROJECT = {"id": "proj-1", "name": "Alpha", "key": "ALP"}
_ISSUE_FULL = {
    "id": "issue-1", "projectId": "proj-1", "number": 42, "title": "Fix login",
    "type": "bug", "priority": "high", "createdAt": "2026-06-01T00:00:00Z",
    "completedAt": "2026-06-02T00:00:00Z",
    "assigneeId": "u1", "reporterId": "u2", "sprintId": "sprint-1",
}
_ISSUE_MINIMAL = {
    "id": "issue-2", "projectId": "proj-1", "number": 43, "title": "Add dark mode",
    "type": "task", "priority": "medium", "createdAt": "2026-06-01T00:00:00Z",
    "completedAt": None,
    "assigneeId": None, "reporterId": "u2", "sprintId": None,
}


# ---------------------------------------------------------------------------
# upsert_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_user_runs_merge():
    neo = _neo4j()
    await upsert_user(neo, _USER)
    neo.run.assert_called_once()
    query, params = neo.run.call_args[0]
    assert "MERGE" in query and "User" in query
    assert params["id"] == "u1"
    assert params["name"] == "Harsh"
    assert params["email"] == "harsh@acme.com"


@pytest.mark.asyncio
async def test_upsert_user_uses_parameterised_query():
    neo = _neo4j()
    await upsert_user(neo, _USER)
    query, _ = neo.run.call_args[0]
    assert "$id" in query
    assert "$name" in query
    assert "$email" in query
    assert "harsh@acme.com" not in query  # never inline values


# ---------------------------------------------------------------------------
# upsert_project
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_project_runs_merge():
    neo = _neo4j()
    await upsert_project(neo, _PROJECT)
    neo.run.assert_called_once()
    query, params = neo.run.call_args[0]
    assert "MERGE" in query and "Project" in query
    assert params["id"] == "proj-1"
    assert params["name"] == "Alpha"
    assert params["key"] == "ALP"


@pytest.mark.asyncio
async def test_upsert_project_uses_parameterised_query():
    neo = _neo4j()
    await upsert_project(neo, _PROJECT)
    query, _ = neo.run.call_args[0]
    assert "$id" in query
    assert "Alpha" not in query


# ---------------------------------------------------------------------------
# upsert_issue — node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_issue_runs_merge_for_issue_node():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    calls = neo.run.call_args_list
    issue_call = next((c for c in calls if "Issue" in c[0][0] and "MERGE" in c[0][0]), None)
    assert issue_call is not None
    params = issue_call[0][1]
    assert params["id"] == "issue-1"
    assert params["number"] == 42
    assert params["title"] == "Fix login"
    assert params["type"] == "bug"
    assert params["priority"] == "high"
    assert params["completedAt"] == "2026-06-02T00:00:00Z"


@pytest.mark.asyncio
async def test_upsert_issue_creates_in_project_relationship():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("IN_PROJECT" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_creates_reported_relationship():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("REPORTED" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_creates_assigned_to_when_assignee_present():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("ASSIGNED_TO" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_skips_assigned_to_when_no_assignee():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_MINIMAL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert not any("ASSIGNED_TO" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_creates_in_sprint_when_sprint_present():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("IN_SPRINT" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_skips_in_sprint_when_no_sprint():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_MINIMAL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert not any("IN_SPRINT" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_uses_only_parameterised_queries():
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_FULL)
    for c in neo.run.call_args_list:
        query = c[0][0]
        assert "Fix login" not in query
        assert "issue-1" not in query


# ---------------------------------------------------------------------------
# upsert_member
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_member_creates_member_of_relationship():
    neo = _neo4j()
    await upsert_member(neo, user_id="u1", project_id="proj-1", role="member")
    neo.run.assert_called_once()
    query, params = neo.run.call_args[0]
    assert "MEMBER_OF" in query
    assert params["user_id"] == "u1"
    assert params["project_id"] == "proj-1"
    assert params["role"] == "member"


@pytest.mark.asyncio
async def test_upsert_member_uses_merge_not_create():
    neo = _neo4j()
    await upsert_member(neo, "u1", "proj-1", "member")
    query, _ = neo.run.call_args[0]
    assert "MERGE" in query
    assert "CREATE" not in query.replace("ON CREATE", "")


# ---------------------------------------------------------------------------
# upsert_comment_activity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_comment_activity_creates_commented_on_relationship():
    neo = _neo4j()
    await upsert_comment_activity(neo, user_id="u1", issue_id="issue-1")
    neo.run.assert_called_once()
    query, params = neo.run.call_args[0]
    assert "COMMENTED_ON" in query
    assert params["user_id"] == "u1"
    assert params["issue_id"] == "issue-1"


@pytest.mark.asyncio
async def test_upsert_comment_activity_increments_count():
    neo = _neo4j()
    await upsert_comment_activity(neo, "u1", "issue-1")
    query, _ = neo.run.call_args[0]
    assert "r.count" in query
    assert "ON CREATE" in query
    assert "ON MATCH" in query


# ---------------------------------------------------------------------------
# upsert_change_activity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_change_activity_creates_changed_relationship():
    neo = _neo4j()
    await upsert_change_activity(neo, user_id="u1", issue_id="issue-1", field="status")
    neo.run.assert_called_once()
    query, params = neo.run.call_args[0]
    assert "CHANGED" in query
    assert params["user_id"] == "u1"
    assert params["issue_id"] == "issue-1"
    assert params["field"] == "status"


@pytest.mark.asyncio
async def test_upsert_change_activity_increments_count_per_field():
    neo = _neo4j()
    await upsert_change_activity(neo, "u1", "issue-1", "priority")
    query, _ = neo.run.call_args[0]
    assert "r.count" in query
    assert "ON CREATE" in query
    assert "ON MATCH" in query


# ---------------------------------------------------------------------------
# full_sync
# ---------------------------------------------------------------------------

_PROJECTS = [{"id": "proj-1", "name": "Alpha", "key": "ALP"}]
_MEMBERS  = [{"userId": "u1", "name": "Harsh", "email": "h@acme.com", "avatarUrl": None, "role": "admin"}]
_ISSUES   = [_ISSUE_FULL, _ISSUE_MINIMAL]


@pytest.mark.asyncio
async def test_full_sync_returns_node_and_rel_counts():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=_MEMBERS, get_issues=_ISSUES)
    result = await full_sync(neo, api, "acme")
    assert "nodes_created" in result
    assert "relationships_created" in result
    assert result["nodes_created"] > 0
    assert result["relationships_created"] > 0


@pytest.mark.asyncio
async def test_full_sync_upserts_each_project():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=[], get_issues=[])
    await full_sync(neo, api, "acme")
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("Project" in q and "MERGE" in q for q in queries)


@pytest.mark.asyncio
async def test_full_sync_upserts_each_member():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=_MEMBERS, get_issues=[])
    await full_sync(neo, api, "acme")
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("User" in q and "MERGE" in q for q in queries)
    assert any("MEMBER_OF" in q for q in queries)


@pytest.mark.asyncio
async def test_full_sync_upserts_each_issue():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=[], get_issues=_ISSUES)
    await full_sync(neo, api, "acme")
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("Issue" in q and "MERGE" in q for q in queries)


@pytest.mark.asyncio
async def test_full_sync_fetches_members_per_project():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=_MEMBERS, get_issues=[])
    await full_sync(neo, api, "acme")
    api.get_project_members.assert_called_once_with("acme", "proj-1")


@pytest.mark.asyncio
async def test_full_sync_fetches_issues_per_project():
    neo = _neo4j()
    api = _api(get_projects=_PROJECTS, get_project_members=[], get_issues=_ISSUES)
    await full_sync(neo, api, "acme")
    api.get_issues.assert_called_once_with("acme", "proj-1")


@pytest.mark.asyncio
async def test_full_sync_counts_include_optional_relationships():
    neo = _neo4j()
    # _ISSUE_FULL has assigneeId + sprintId; _ISSUE_MINIMAL has neither
    api = _api(get_projects=_PROJECTS, get_project_members=[], get_issues=_ISSUES)
    result = await full_sync(neo, api, "acme")
    # _ISSUE_FULL: IN_PROJECT + REPORTED + ASSIGNED_TO + IN_SPRINT = 4
    # _ISSUE_MINIMAL: IN_PROJECT + REPORTED = 2
    assert result["relationships_created"] >= 6
