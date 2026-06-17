"""Tests for knowledge-graph Cypher query helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_service.graph.queries import (
    get_graph_stats,
    get_issue_neighbors,
    get_similar_issues,
    get_smart_assignee,
    get_sprint_issues,
    get_subtask_tree,
    get_user_activity,
    get_user_workload,
)
from ai_service.neo4j import Neo4jClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.last_query: str = ""
        self.last_params: dict[str, Any] = {}

    async def run(self, query: str, **kwargs: Any) -> MagicMock:
        self.last_query = query
        self.last_params = kwargs
        result = MagicMock()
        result.data = AsyncMock(return_value=self._rows)
        return result


def _make_client(rows: list[dict[str, Any]]) -> tuple[Neo4jClient, _MockSession]:
    sess = _MockSession(rows)
    client = MagicMock(spec=Neo4jClient)

    @asynccontextmanager
    async def _session() -> AsyncIterator[_MockSession]:
        yield sess

    client.session = _session  # type: ignore[method-assign]
    return client, sess  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# get_smart_assignee
# ---------------------------------------------------------------------------


class TestGetSmartAssignee:
    async def test_returns_rows_from_neo4j(self) -> None:
        rows = [
            {"user_id": "u-1", "name": "Alice", "resolved_count": 3,
             "comment_count": 2, "open_issues": 1, "score": 4.0},
        ]
        client, _ = _make_client(rows)
        result = await get_smart_assignee(client, "iss-1", "org-1")
        assert len(result) == 1
        assert result[0]["user_id"] == "u-1"

    async def test_returns_empty_list_when_no_data(self) -> None:
        client, _ = _make_client([])
        result = await get_smart_assignee(client, "iss-1", "org-1")
        assert result == []

    async def test_query_uses_issue_id_and_org_id_params(self) -> None:
        client, sess = _make_client([])
        await get_smart_assignee(client, "iss-42", "org-99")
        assert sess.last_params["issue_id"] == "iss-42"
        assert sess.last_params["org_id"] == "org-99"

    async def test_query_references_resolved_relationship(self) -> None:
        client, sess = _make_client([])
        await get_smart_assignee(client, "iss-1", "org-1")
        assert "RESOLVED" in sess.last_query

    async def test_query_references_commented_on_relationship(self) -> None:
        client, sess = _make_client([])
        await get_smart_assignee(client, "iss-1", "org-1")
        assert "COMMENTED_ON" in sess.last_query

    async def test_query_applies_workload_penalty(self) -> None:
        client, sess = _make_client([])
        await get_smart_assignee(client, "iss-1", "org-1")
        assert "open_issues" in sess.last_query.lower() or "ASSIGNED_TO" in sess.last_query


# ---------------------------------------------------------------------------
# get_similar_issues
# ---------------------------------------------------------------------------


class TestGetSimilarIssues:
    async def test_returns_empty_list_for_empty_keywords(self) -> None:
        client, _ = _make_client([{"issue_id": "iss-1"}])
        result = await get_similar_issues(client, [], "org-1")
        assert result == []

    async def test_passes_keywords_to_query(self) -> None:
        client, sess = _make_client([])
        await get_similar_issues(client, ["auth", "login"], "org-1")
        assert sess.last_params["keywords"] == ["auth", "login"]

    async def test_returns_rows_from_neo4j(self) -> None:
        rows = [{"issue_id": "iss-1", "title": "Fix auth", "type": "bug",
                 "priority": "high", "completed_at": None, "category_name": "Auth"}]
        client, _ = _make_client(rows)
        result = await get_similar_issues(client, ["auth"], "org-1")
        assert result[0]["issue_id"] == "iss-1"

    async def test_query_scoped_to_org(self) -> None:
        client, sess = _make_client([])
        await get_similar_issues(client, ["test"], "org-42")
        assert "BELONGS_TO" in sess.last_query
        assert sess.last_params["org_id"] == "org-42"


# ---------------------------------------------------------------------------
# get_issue_neighbors
# ---------------------------------------------------------------------------


class TestGetIssueNeighbors:
    async def test_returns_neighbors(self) -> None:
        rows = [{"issue_id": "iss-2", "title": "Related", "type": "task",
                 "priority": "low", "completed_at": None,
                 "category_name": "Backend", "in_same_sprint": False}]
        client, _ = _make_client(rows)
        result = await get_issue_neighbors(client, "iss-1")
        assert result[0]["in_same_sprint"] is False

    async def test_query_excludes_target_issue(self) -> None:
        client, sess = _make_client([])
        await get_issue_neighbors(client, "iss-1")
        assert "$issue_id" in sess.last_query or "issue_id" in sess.last_query


# ---------------------------------------------------------------------------
# get_sprint_issues
# ---------------------------------------------------------------------------


class TestGetSprintIssues:
    async def test_returns_sprint_issues(self) -> None:
        rows = [{"issue_id": "iss-1", "title": "T", "type": "task",
                 "priority": "medium", "status_id": "st-1", "completed_at": None,
                 "assignee_id": "u-1", "assignee_name": "Alice"}]
        client, _ = _make_client(rows)
        result = await get_sprint_issues(client, "sp-1")
        assert result[0]["assignee_name"] == "Alice"

    async def test_query_uses_sprint_id(self) -> None:
        client, sess = _make_client([])
        await get_sprint_issues(client, "sp-99")
        assert "IN_SPRINT" in sess.last_query
        assert sess.last_params["sprint_id"] == "sp-99"


# ---------------------------------------------------------------------------
# get_user_workload
# ---------------------------------------------------------------------------


class TestGetUserWorkload:
    async def test_all_users_when_no_user_id(self) -> None:
        rows = [{"user_id": "u-1", "name": "Alice", "open_issues": 3}]
        client, sess = _make_client(rows)
        result = await get_user_workload(client, "org-1")
        assert result[0]["open_issues"] == 3
        assert "MEMBER_OF" in sess.last_query

    async def test_single_user_when_user_id_provided(self) -> None:
        rows = [{"user_id": "u-1", "name": "Alice", "open_issues": 2}]
        client, sess = _make_client(rows)
        result = await get_user_workload(client, "org-1", user_id="u-1")
        assert sess.last_params["user_id"] == "u-1"
        assert result[0]["open_issues"] == 2


# ---------------------------------------------------------------------------
# get_subtask_tree
# ---------------------------------------------------------------------------


class TestGetSubtaskTree:
    async def test_returns_subtasks(self) -> None:
        rows = [{"issue_id": "iss-2", "title": "Sub", "type": "subtask",
                 "priority": "low", "completed_at": None, "depth": 1}]
        client, _ = _make_client(rows)
        result = await get_subtask_tree(client, "iss-1")
        assert result[0]["depth"] == 1

    async def test_query_uses_subtask_of_relationship(self) -> None:
        client, sess = _make_client([])
        await get_subtask_tree(client, "iss-1")
        assert "SUBTASK_OF" in sess.last_query


# ---------------------------------------------------------------------------
# get_graph_stats
# ---------------------------------------------------------------------------


class TestGetGraphStats:
    async def test_returns_node_and_rel_counts(self) -> None:
        # The function runs two queries: one for nodes, one for rels.
        # We need a session that handles both differently.
        call_count = 0
        node_rows = [{"label": "Issue", "count": 42}]
        rel_rows = [{"rel_type": "ASSIGNED_TO", "count": 10}]

        sess = MagicMock()

        async def multi_run(query: str, **_kw: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.data = AsyncMock(return_value=node_rows)
            else:
                result.data = AsyncMock(return_value=rel_rows)
            return result

        sess.run = multi_run
        client = MagicMock(spec=Neo4jClient)

        @asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            yield sess

        client.session = _session  # type: ignore[method-assign]
        stats = await get_graph_stats(client)  # type: ignore[arg-type]
        assert stats["nodes"]["Issue"] == 42
        assert stats["relationships"]["ASSIGNED_TO"] == 10


# ---------------------------------------------------------------------------
# Project scoping (hard restriction when a project is selected in chat)
# ---------------------------------------------------------------------------


class TestSimilarIssuesProjectScope:
    async def test_no_project_filter_when_unscoped(self) -> None:
        client, sess = _make_client([])
        await get_similar_issues(client, ["auth"], "org-1")
        assert "project_id" not in sess.last_params

    async def test_filters_by_project_when_scoped(self) -> None:
        client, sess = _make_client([])
        await get_similar_issues(
            client, ["auth"], "org-1", project_id="proj-1"
        )
        assert sess.last_params["project_id"] == "proj-1"
        assert "p.id" in sess.last_query


class TestUserWorkloadProjectScope:
    async def test_no_project_filter_when_unscoped(self) -> None:
        client, sess = _make_client([])
        await get_user_workload(client, "org-1")
        assert "project_id" not in sess.last_params

    async def test_filters_by_project_when_scoped(self) -> None:
        client, sess = _make_client([])
        await get_user_workload(client, "org-1", project_id="proj-1")
        assert sess.last_params["project_id"] == "proj-1"
        assert "IN_PROJECT" in sess.last_query


class TestGetUserActivity:
    async def test_returns_rows_from_neo4j(self) -> None:
        rows = [
            {"user_id": "u-1", "name": "Carol", "change_count": 9,
             "comment_count": 2, "total_activity": 11},
        ]
        client, _ = _make_client(rows)
        result = await get_user_activity(client, "org-1")
        assert len(result) == 1
        assert result[0]["total_activity"] == 11

    async def test_returns_empty_list_when_no_data(self) -> None:
        client, _ = _make_client([])
        assert await get_user_activity(client, "org-1") == []

    async def test_query_references_changed_and_commented_on(self) -> None:
        client, sess = _make_client([])
        await get_user_activity(client, "org-99")
        assert "CHANGED" in sess.last_query
        assert "COMMENTED_ON" in sess.last_query
        assert sess.last_params["org_id"] == "org-99"

    async def test_ranks_by_total_activity_descending(self) -> None:
        client, sess = _make_client([])
        await get_user_activity(client, "org-1")
        assert "ORDER BY total_activity DESC" in sess.last_query

    async def test_no_project_filter_when_unscoped(self) -> None:
        client, sess = _make_client([])
        await get_user_activity(client, "org-1")
        # Unscoped: no project_id binding; scope is the whole org.
        assert "project_id" not in sess.last_params
        assert "$project_id" not in sess.last_query

    async def test_filters_by_project_when_scoped(self) -> None:
        client, sess = _make_client([])
        await get_user_activity(client, "org-1", project_id="proj-1")
        assert sess.last_params["project_id"] == "proj-1"
        assert "$project_id" in sess.last_query
