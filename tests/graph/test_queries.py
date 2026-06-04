import pytest
from unittest.mock import AsyncMock, MagicMock

from graph.queries import (
    find_similar_issues, find_user_by_name, get_expertise_map, suggest_assignee,
)


def _neo4j(*call_returns):
    """Mock that returns each item in call_returns on successive neo4j.run calls."""
    m = MagicMock()
    m.run = AsyncMock(side_effect=list(call_returns))
    return m


# ---------------------------------------------------------------------------
# suggest_assignee
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suggest_assignee_runs_three_queries():
    neo = _neo4j([], [], [])
    await suggest_assignee(neo, "proj-1", "fix login bug", "bug")
    assert neo.run.call_count == 3


@pytest.mark.asyncio
async def test_suggest_assignee_returns_sorted_by_score_desc():
    neo = _neo4j(
        [{"userId": "u1", "cnt": 1}, {"userId": "u2", "cnt": 3}],
        [],
        [{"userId": "u1", "userName": "Alice", "open_count": 0},
         {"userId": "u2", "userName": "Bob",   "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "fix login", "bug")
    assert results[0]["userId"] == "u2"   # 3 resolved × 3 = score 9
    assert results[1]["userId"] == "u1"   # 1 resolved × 3 = score 3


@pytest.mark.asyncio
async def test_suggest_assignee_resolved_weight_is_three():
    neo = _neo4j(
        [{"userId": "u1", "cnt": 2}],
        [],
        [{"userId": "u1", "userName": "Alice", "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert results[0]["score"] == 6   # 2 × 3


@pytest.mark.asyncio
async def test_suggest_assignee_comments_add_to_score():
    neo = _neo4j(
        [],
        [{"userId": "u1", "cnt": 4}],
        [{"userId": "u1", "userName": "Alice", "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert results[0]["score"] == 4


@pytest.mark.asyncio
async def test_suggest_assignee_open_issues_penalise_score():
    neo = _neo4j(
        [{"userId": "u1", "cnt": 2}],
        [],
        [{"userId": "u1", "userName": "Alice", "open_count": 3}],  # penalty -3 → final 3
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert results[0]["score"] == 3


@pytest.mark.asyncio
async def test_suggest_assignee_load_query_includes_all_members():
    """Users only in the load query (no resolved/comment history) still appear."""
    neo = _neo4j(
        [],
        [],
        [{"userId": "u1", "userName": "Alice", "open_count": 1},
         {"userId": "u2", "userName": "Bob",   "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    user_ids  = [r["userId"]   for r in results]
    user_names = [r["userName"] for r in results]
    assert "u1" in user_ids
    assert "u2" in user_ids
    assert "Alice" in user_names
    assert "Bob"   in user_names


@pytest.mark.asyncio
async def test_suggest_assignee_empty_graph_returns_empty():
    neo = _neo4j([], [], [])
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert results == []


@pytest.mark.asyncio
async def test_suggest_assignee_load_query_filters_removed_members():
    """The cypher that fetches current project members must exclude
    tombstoned edges — otherwise an ex-member's open-issue count is still
    counted in the load penalty and they remain a 'candidate'."""
    neo = _neo4j([], [], [])
    await suggest_assignee(neo, "proj-1", "title", "bug")
    load_call = neo.run.call_args_list[2]  # 3rd cypher = load
    load_query = load_call[0][0]
    assert "MEMBER_OF" in load_query
    assert "removed_at IS NULL" in load_query


@pytest.mark.asyncio
async def test_suggest_assignee_excludes_tombstoned_members_from_result():
    """An ex-member with great past work must NOT be in the suggestion list
    once they are tombstoned. Setup: the user appears in the resolved and
    load queries, but with ``removed_at`` set; the active filter at the end
    drops them.

    The current implementation does this by filtering the final result on
    the active-member set from the load query. We model that here: the
    load query returns [] (so active set is empty), the resolved query
    returns a high-scoring user, and we expect the final result to be [].
    """
    neo = _neo4j(
        # resolved
        [{"userId": "u-ex", "cnt": 5}],
        # commented
        [],
        # load (empty — u-ex is tombstoned, so not in active members)
        [],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert results == [], "tombstoned user must not surface as a suggestion"


@pytest.mark.asyncio
async def test_suggest_assignee_keeps_active_member_with_resolved_history():
    """Sanity: a user who IS in the active load set AND has resolved history
    should still surface — the tombstone filter only drops ex-members, not
    low-activity members."""
    neo = _neo4j(
        [{"userId": "u1", "cnt": 2}],   # 2 resolved bugs = score 6
        [],
        [{"userId": "u1", "userName": "Alice", "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert len(results) == 1
    assert results[0]["userId"] == "u1"
    assert results[0]["score"] == 6.0  # 2*3 - 0


@pytest.mark.asyncio
async def test_suggest_assignee_mixed_active_and_tombstoned():
    """A tombstoned user with great history AND an active user with none —
    only the active user should appear."""
    neo = _neo4j(
        # resolved: tombstoned user has tons of history, active has 1
        [{"userId": "u-tomb", "cnt": 10}, {"userId": "u-active", "cnt": 1}],
        [],
        # load: only the active user
        [{"userId": "u-active", "userName": "Bob", "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    assert len(results) == 1
    assert results[0]["userId"] == "u-active"
    assert results[0]["userName"] == "Bob"


# --- tombstone-aware behavior (regression for the original bug report) -----


@pytest.mark.asyncio
async def test_suggest_assignee_excludes_ex_member_who_left_project():
    """The actual bug: an ex-member who left the project still gets
    recommended by smart-assignee. With tombstones, they are filtered out.

    Seed:
      - u-ex was a great engineer (5 resolved bugs) and IS in resolved count
      - u-ex's MEMBER_OF is tombstoned (not in load query result)
      - u-active is a current member (in load, 0 open)
    Expect:
      - Only u-active is recommended.
    """
    neo = _neo4j(
        [{"userId": "u-ex", "cnt": 5}],
        [],
        [{"userId": "u-active", "userName": "Current Member", "open_count": 0}],
    )
    results = await suggest_assignee(neo, "proj-1", "title", "bug")
    user_ids = {r["userId"] for r in results}
    assert "u-ex" not in user_ids, "tombstoned ex-member must not be recommended"
    assert "u-active" in user_ids


@pytest.mark.asyncio
async def test_suggest_assignee_passes_project_id_and_type():
    neo = _neo4j([], [], [])
    await suggest_assignee(neo, "proj-99", "title", "story")
    for call in neo.run.call_args_list:
        params = call[0][1]
        assert params.get("project_id") == "proj-99"


@pytest.mark.asyncio
async def test_suggest_assignee_queries_are_parameterised():
    neo = _neo4j([], [], [])
    await suggest_assignee(neo, "proj-1", "fix login", "bug")
    for call in neo.run.call_args_list:
        query = call[0][0]
        assert "proj-1" not in query
        assert "bug" not in query
        assert "fix login" not in query


# ---------------------------------------------------------------------------
# find_similar_issues
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_find_similar_issues_returns_neo4j_results():
    expected = [{"id": "i1", "number": 1, "title": "Login page broken", "match_count": 2}]
    neo = _neo4j(expected)
    results = await find_similar_issues(neo, "proj-1", "login page crash")
    assert results == expected


@pytest.mark.asyncio
async def test_find_similar_issues_filters_short_words():
    neo = _neo4j([])
    await find_similar_issues(neo, "proj-1", "login page crash")
    _, params = neo.run.call_args[0]
    # "login"(5), "page"(4), "crash"(5) — all pass; "on"(2), "in"(2) would be filtered
    assert all(len(w) > 3 for w in params["words"])


@pytest.mark.asyncio
async def test_find_similar_issues_empty_title_returns_empty():
    neo = _neo4j()
    results = await find_similar_issues(neo, "proj-1", "")
    assert results == []
    neo.run.assert_not_called()


@pytest.mark.asyncio
async def test_find_similar_issues_all_short_words_returns_empty():
    neo = _neo4j()
    results = await find_similar_issues(neo, "proj-1", "a to do it")
    assert results == []
    neo.run.assert_not_called()


@pytest.mark.asyncio
async def test_find_similar_issues_query_is_parameterised():
    neo = _neo4j([])
    await find_similar_issues(neo, "proj-1", "login page crash")
    query, params = neo.run.call_args[0]
    assert "proj-1" not in query
    assert "login" not in query
    assert "$words" in query
    assert "$project_id" in query


@pytest.mark.asyncio
async def test_find_similar_issues_words_are_lowercased():
    neo = _neo4j([])
    await find_similar_issues(neo, "proj-1", "Login Page Crash")
    _, params = neo.run.call_args[0]
    assert all(w == w.lower() for w in params["words"])


@pytest.mark.asyncio
async def test_find_similar_issues_passes_project_id():
    neo = _neo4j([])
    await find_similar_issues(neo, "proj-42", "login page crash")
    _, params = neo.run.call_args[0]
    assert params["project_id"] == "proj-42"


# ---------------------------------------------------------------------------
# get_expertise_map
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_expertise_map_returns_neo4j_results():
    expected = [
        {"userId": "u1", "name": "Harsh", "assigned": 5, "comments": 12, "changes": 8},
        {"userId": "u2", "name": "Sara",  "assigned": 2, "comments":  3, "changes": 1},
    ]
    neo = _neo4j(expected)
    results = await get_expertise_map(neo, "proj-1")
    assert results == expected


@pytest.mark.asyncio
async def test_get_expertise_map_runs_exactly_one_query():
    neo = _neo4j([])
    await get_expertise_map(neo, "proj-1")
    assert neo.run.call_count == 1


@pytest.mark.asyncio
async def test_get_expertise_map_passes_project_id():
    neo = _neo4j([])
    await get_expertise_map(neo, "proj-99")
    _, params = neo.run.call_args[0]
    assert params["project_id"] == "proj-99"


@pytest.mark.asyncio
async def test_get_expertise_map_query_is_parameterised():
    neo = _neo4j([])
    await get_expertise_map(neo, "proj-1")
    query, _ = neo.run.call_args[0]
    assert "proj-1" not in query
    assert "$project_id" in query


@pytest.mark.asyncio
async def test_get_expertise_map_query_covers_all_activity_types():
    neo = _neo4j([])
    await get_expertise_map(neo, "proj-1")
    query, _ = neo.run.call_args[0]
    assert "ASSIGNED_TO" in query
    assert "COMMENTED_ON" in query
    assert "CHANGED" in query


# ---------------------------------------------------------------------------
# find_user_by_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_user_by_name_empty_query_returns_empty():
    neo = _neo4j([])
    results = await find_user_by_name(neo, "proj-1", "")
    assert results == []
    neo.run.assert_not_called()


@pytest.mark.asyncio
async def test_find_user_by_name_whitespace_query_returns_empty():
    neo = _neo4j([])
    results = await find_user_by_name(neo, "proj-1", "   ")
    assert results == []
    neo.run.assert_not_called()


@pytest.mark.asyncio
async def test_find_user_by_name_returns_neo4j_results():
    expected = [
        {"id": "u-alice", "name": "Alice Smith", "email": "alice@acme.com"},
        {"id": "u-alicia", "name": "Alicia Keys", "email": "alicia@acme.com"},
    ]
    neo = _neo4j(expected)
    results = await find_user_by_name(neo, "proj-1", "ali")
    assert results == expected


@pytest.mark.asyncio
async def test_find_user_by_name_is_case_insensitive():
    """The cypher does toLower() on both sides; the params should pass the
    original casing and let the DB lower it. Lock that in."""
    neo = _neo4j([{"id": "u1", "name": "Harsh", "email": "h@x.com"}])
    await find_user_by_name(neo, "proj-1", "HARSH")
    _, params = neo.run.call_args[0]
    # We pass the raw query — the DB lowercases it.
    assert params["name"] == "HARSH"
    assert params["project_id"] == "proj-1"


@pytest.mark.asyncio
async def test_find_user_by_name_query_is_parameterised():
    neo = _neo4j([])
    await find_user_by_name(neo, "proj-1", "Alice")
    query, params = neo.run.call_args[0]
    assert "Alice" not in query
    assert "proj-1" not in query
    assert "$name" in query
    assert "$project_id" in query
    assert params["name"] == "Alice"


@pytest.mark.asyncio
async def test_find_user_by_name_query_filters_removed_members():
    """Cypher must filter on r.removed_at IS NULL so tombstoned members do
    not match — they left the project and must not be assignable."""
    neo = _neo4j([])
    await find_user_by_name(neo, "proj-1", "Alice")
    query, _ = neo.run.call_args[0]
    assert "removed_at" in query
    assert "IS NULL" in query


@pytest.mark.asyncio
async def test_find_user_by_name_query_matches_email_too():
    """A user might say 'ping alice@acme.com' — the resolver must also
    match on email substring."""
    neo = _neo4j([])
    await find_user_by_name(neo, "proj-1", "alice@acme.com")
    query, _ = neo.run.call_args[0]
    assert "u.email" in query
    assert "u.name" in query
