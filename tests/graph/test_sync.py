import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.sync import (
    full_sync,
    tombstone_missing_members,
    upsert_change_activity,
    upsert_comment_activity,
    upsert_issue,
    upsert_label,
    upsert_member,
    upsert_project,
    upsert_sprint,
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
    # full_sync now also fetches sprints (item 15) — default to empty so tests
    # that don't care about sprints don't have to mock the method.
    kwargs.setdefault("get_sprints", [])
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
    assert "tombstones_created" in result
    assert result["nodes_created"] > 0
    assert result["relationships_created"] > 0
    assert result["tombstones_created"] == 0  # first sync: no prior state to tombstone


# ---------------------------------------------------------------------------
# upsert_member — tombstone reset on rejoin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_member_resets_removed_at_on_existing_relationship():
    """A re-join must clear the tombstone. We verify by checking the cypher
    sets ``removed_at = null`` on the relationship — applies to both new
    and existing edges (we use a single SET, not ON CREATE/ON MATCH)."""
    neo = _neo4j()
    await upsert_member(neo, user_id="u1", project_id="proj-1", role="member")
    query, _ = neo.run.call_args[0]
    assert "removed_at" in query
    assert "null" in query
    # The SET clause must cover BOTH the role and the tombstone reset.
    assert "SET r.role" in query
    assert "r.removed_at = null" in query


# ---------------------------------------------------------------------------
# tombstone_missing_members
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tombstone_marks_only_missing_active_users():
    """Users in the graph's MEMBER_OF who are NOT in active_user_ids get
    removed_at set. Users still in the set are untouched."""
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[{"tombstoned": 2}])
    result = await tombstone_missing_members(neo, "proj-1", {"u1", "u2"})

    assert result == 2
    query, params = neo.run.call_args[0]
    # Must filter on removed_at IS NULL (idempotent — already-tombstoned rows skipped).
    assert "removed_at IS NULL" in query
    # Must use a NOT IN clause with the active set.
    assert "NOT u.id IN" in query
    assert "active_user_ids" in params
    assert sorted(params["active_user_ids"]) == ["u1", "u2"]


@pytest.mark.asyncio
async def test_tombstone_with_empty_active_set_marks_everything():
    """If the project's entire roster was wiped, every still-active edge
    must be tombstoned — no NOT IN clause needed because there are no
    active IDs to keep."""
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[{"tombstoned": 5}])
    result = await tombstone_missing_members(neo, "proj-1", set())

    assert result == 5
    query, params = neo.run.call_args[0]
    assert "NOT u.id IN" not in query  # special-cased empty set
    assert "active_user_ids" not in params
    assert "removed_at IS NULL" in query


@pytest.mark.asyncio
async def test_tombstone_uses_parameterised_cypher():
    """No literal project_id in the query string."""
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[{"tombstoned": 0}])
    await tombstone_missing_members(neo, "proj-42", {"u1"})
    query, _ = neo.run.call_args[0]
    assert "proj-42" not in query
    assert "$project_id" in query


@pytest.mark.asyncio
async def test_tombstone_returns_zero_on_empty_neo4j_result():
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[])
    assert await tombstone_missing_members(neo, "proj-1", {"u1"}) == 0


# ---------------------------------------------------------------------------
# upsert_label / upsert_sprint (item 15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_label_merges_node():
    neo = _neo4j()
    await upsert_label(neo, {"id": "lbl-1", "name": "backend"})
    query, params = neo.run.call_args[0]
    assert "MERGE" in query and "Label" in query
    assert params["id"] == "lbl-1"
    assert params["name"] == "backend"


@pytest.mark.asyncio
async def test_upsert_sprint_sets_metadata():
    neo = _neo4j()
    await upsert_sprint(neo, {
        "id": "sprint-1", "name": "Q3 Hardening", "status": "active",
        "startDate": "2026-06-01", "endDate": "2026-06-14",
    })
    query, params = neo.run.call_args[0]
    assert "MERGE" in query and "Sprint" in query
    assert params["name"] == "Q3 Hardening"
    assert params["status"] == "active"
    assert params["startDate"] == "2026-06-01"


# ---------------------------------------------------------------------------
# upsert_issue — labels + BLOCKS (items 14/15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_issue_creates_has_label_edges():
    neo = _neo4j()
    issue = {**_ISSUE_MINIMAL, "labels": [{"id": "lbl-1", "name": "backend"}]}
    await upsert_issue(neo, issue)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("HAS_LABEL" in q for q in queries)
    assert any("Label" in q and "MERGE" in q for q in queries)


@pytest.mark.asyncio
async def test_upsert_issue_creates_blocks_edges():
    neo = _neo4j()
    issue = {**_ISSUE_MINIMAL, "blocks": ["issue-9"]}
    await upsert_issue(neo, issue)
    blocks_calls = [c for c in neo.run.call_args_list if "BLOCKS" in c[0][0]]
    assert blocks_calls, "expected a BLOCKS edge"
    # The edge direction is (this)-[:BLOCKS]->(blocked).
    params = blocks_calls[0][0][1]
    assert params["issue_id"] == "issue-2"
    assert params["blocked_id"] == "issue-9"


@pytest.mark.asyncio
async def test_upsert_issue_creates_reverse_blocks_for_blockedby():
    neo = _neo4j()
    issue = {**_ISSUE_MINIMAL, "blockedBy": ["issue-7"]}
    await upsert_issue(neo, issue)
    blocks_calls = [c for c in neo.run.call_args_list if "BLOCKS" in c[0][0]]
    assert blocks_calls
    params = blocks_calls[0][0][1]
    # blockedBy => (blocker)-[:BLOCKS]->(this)
    assert params["blocker_id"] == "issue-7"


@pytest.mark.asyncio
async def test_upsert_issue_without_labels_or_blocks_adds_no_extra_edges():
    """Backward compat: an issue with neither key must not emit label/blocks
    cypher (so the full_sync path is unchanged)."""
    neo = _neo4j()
    await upsert_issue(neo, _ISSUE_MINIMAL)
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert not any("HAS_LABEL" in q for q in queries)
    assert not any("BLOCKS" in q for q in queries)


@pytest.mark.asyncio
async def test_full_sync_upserts_sprints():
    neo = _neo4j()
    api = _api(
        get_projects=_PROJECTS, get_project_members=[], get_issues=[],
        get_sprints=[{"id": "sprint-1", "name": "Q3", "status": "active"}],
    )
    await full_sync(neo, api, "acme")
    queries = [c[0][0] for c in neo.run.call_args_list]
    assert any("Sprint" in q and "MERGE" in q and "status" in q for q in queries)


# ---------------------------------------------------------------------------
# full_sync — tombstone integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_sync_calls_tombstone_with_active_member_set():
    """The orchestrator must feed the active-member set to tombstone_missing_members
    so the graph reflects Node's current view of the project."""
    neo = _neo4j()
    api = _api(
        get_projects=_PROJECTS,
        get_project_members=_MEMBERS,  # contains "u1"
        get_issues=[],
    )
    with patch(
        "graph.sync.tombstone_missing_members",
        new=AsyncMock(return_value=0),
    ) as tomb_fn:
        await full_sync(neo, api, "acme")

    tomb_fn.assert_awaited_once()
    args = tomb_fn.await_args.args
    # args: (neo4j_client, project_id, active_user_ids)
    assert args[1] == "proj-1"
    assert args[2] == {"u1"}


@pytest.mark.asyncio
async def test_full_sync_tombstone_count_in_return_value():
    """The summary dict must surface how many tombstones were created so ops
    can spot mass-removal events (a project whose team all left)."""
    neo = _neo4j()
    api = _api(
        get_projects=_PROJECTS,
        get_project_members=_MEMBERS,
        get_issues=[],
    )
    with patch(
        "graph.sync.tombstone_missing_members",
        new=AsyncMock(return_value=3),
    ):
        result = await full_sync(neo, api, "acme")
    assert result["tombstones_created"] == 3


@pytest.mark.asyncio
async def test_full_sync_reactivates_rejoined_member():
    """If a previously-tombstoned user is back in Node's response, the
    upsert_member path clears removed_at — verified by the cypher itself."""
    neo = _neo4j()
    api = _api(
        get_projects=_PROJECTS,
        get_project_members=_MEMBERS,  # u1 is in here
        get_issues=[],
    )
    await full_sync(neo, api, "acme")

    # The second-to-last run call (before the tombstone count) is the
    # upsert_member for "u1". Its cypher must set removed_at = null.
    upsert_calls = [
        c for c in neo.run.call_args_list
        if "MEMBER_OF" in c[0][0] and "MERGE" in c[0][0]
    ]
    assert upsert_calls, "expected at least one upsert_member cypher"
    cypher = upsert_calls[0][0][0]
    assert "removed_at" in cypher
    assert "null" in cypher


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
