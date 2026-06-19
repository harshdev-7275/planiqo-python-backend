"""Tests for knowledge-graph agent tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ai_service.agents.tools.graph import make_graph_tools
from ai_service.neo4j import Neo4jClient


def _make_neo4j() -> Neo4jClient:
    client = MagicMock(spec=Neo4jClient)
    return client  # type: ignore[return-value]


class TestMakeGraphTools:
    def test_returns_eight_tools(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        assert len(tools) == 8

    def test_tool_names_include_suggest_assignee(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        names = [t.name for t in tools]
        assert "graph_suggest_assignee" in names

    def test_tool_names_include_similar_issues(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        names = [t.name for t in tools]
        assert "graph_similar_issues" in names

    def test_all_tools_have_description(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        for t in tools:
            assert t.description, f"Tool {t.name} is missing a description"


class TestGraphSuggestAssigneeTool:
    async def test_calls_get_smart_assignee(self) -> None:
        expected = [{"user_id": "u-1", "name": "Alice", "score": 3.0}]
        with patch(
            "ai_service.graph.queries.get_smart_assignee",
            new=AsyncMock(return_value=expected),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_suggest_assignee")
            result = await tool_fn.ainvoke(
                {"issue_id": "iss-1", "issue_title": "Fix login"}
            )
        mock_fn.assert_awaited_once()
        assert result == expected

    async def test_returns_empty_list_when_no_graph_data(self) -> None:
        with patch(
            "ai_service.graph.queries.get_smart_assignee",
            new=AsyncMock(return_value=[]),
        ):
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_suggest_assignee")
            result = await tool_fn.ainvoke(
                {"issue_id": "iss-999", "issue_title": "Ghost issue"}
            )
        assert result == []


class TestGraphSimilarIssuesTool:
    async def test_calls_get_similar_issues(self) -> None:
        expected = [{"issue_id": "iss-2", "title": "Similar"}]
        with patch(
            "ai_service.graph.queries.get_similar_issues",
            new=AsyncMock(return_value=expected),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_similar_issues")
            result = await tool_fn.ainvoke({"keywords": ["auth"]})
        mock_fn.assert_awaited_once()
        assert result == expected

    async def test_clamps_limit_to_10_when_out_of_range(self) -> None:
        with patch(
            "ai_service.graph.queries.get_similar_issues",
            new=AsyncMock(return_value=[]),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_similar_issues")
            await tool_fn.ainvoke({"keywords": ["auth"], "limit": 999})
        _, call_kwargs = mock_fn.call_args
        assert call_kwargs.get("limit") == 10


class TestGraphUserWorkloadTool:
    async def test_passes_empty_string_as_none(self) -> None:
        with patch(
            "ai_service.graph.queries.get_user_workload",
            new=AsyncMock(return_value=[]),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_user_workload")
            await tool_fn.ainvoke({"user_id": ""})
        _, kwargs = mock_fn.call_args
        assert kwargs.get("user_id") is None


class TestGraphUserActivityTool:
    async def test_calls_get_user_activity(self) -> None:
        expected = [
            {"user_id": "u-1", "name": "Carol", "change_count": 9,
             "comment_count": 2, "total_activity": 11},
        ]
        with patch(
            "ai_service.graph.queries.get_user_activity",
            new=AsyncMock(return_value=expected),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_user_activity")
            result = await tool_fn.ainvoke({})
        mock_fn.assert_awaited_once()
        assert result == expected

    async def test_clamps_limit_to_10_when_out_of_range(self) -> None:
        with patch(
            "ai_service.graph.queries.get_user_activity",
            new=AsyncMock(return_value=[]),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1")
            tool_fn = next(t for t in tools if t.name == "graph_user_activity")
            await tool_fn.ainvoke({"limit": 999})
        _, kwargs = mock_fn.call_args
        assert kwargs.get("limit") == 10

    async def test_passes_scoped_project_id(self) -> None:
        with patch(
            "ai_service.graph.queries.get_user_activity",
            new=AsyncMock(return_value=[]),
        ) as mock_fn:
            tools = make_graph_tools(_make_neo4j(), "org-1", scoped_project_id="proj-1")
            tool_fn = next(t for t in tools if t.name == "graph_user_activity")
            await tool_fn.ainvoke({})
        _, kwargs = mock_fn.call_args
        assert kwargs.get("project_id") == "proj-1"
