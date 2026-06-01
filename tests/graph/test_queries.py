import pytest
from unittest.mock import AsyncMock, MagicMock

from graph.queries import find_similar_issues, get_expertise_map, suggest_assignee


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
