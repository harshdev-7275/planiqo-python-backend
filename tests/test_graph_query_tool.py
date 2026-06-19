"""Tests for the graph_query escape-hatch tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_service.agents.tools.graph import make_graph_tools, _WRITE_PATTERN
from ai_service.neo4j import Neo4jClient


def _make_neo4j() -> Neo4jClient:
    return MagicMock(spec=Neo4jClient)  # type: ignore[return-value]


def _get_graph_query(tools: list) -> object:
    return next(t for t in tools if t.name == "graph_query")


class TestWritePatternRejection:
    """The regex must catch all write keywords."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Org {id: '1'})",
            "MERGE (n:Org {id: '1'})",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) REMOVE n.x",
            "DROP CONSTRAINT foo",
            "CALL { CREATE (n:Org) }",
        ],
    )
    def test_rejects_write_operations(self, cypher: str) -> None:
        assert _WRITE_PATTERN.search(cypher) is not None

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:Org) RETURN n",
            "MATCH (n)-[r]->(m) RETURN n, type(r), m LIMIT 10",
            "MATCH (c:Category) RETURN c.name, count(*) ORDER BY count(*) DESC",
        ],
    )
    def test_allows_read_operations(self, cypher: str) -> None:
        assert _WRITE_PATTERN.search(cypher) is None


class TestGraphQueryTool:
    def test_tool_is_included(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        names = [t.name for t in tools]
        assert "graph_query" in names

    @pytest.mark.asyncio
    async def test_rejects_write_and_returns_error(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        tool_fn = _get_graph_query(tools)
        result = await tool_fn.ainvoke({"cypher": "CREATE (n:Org {id: '1'})"})
        assert len(result) == 1
        assert "error" in result[0]
        assert "Write operations" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_successful_read_query(self) -> None:
        expected = [{"name": "Backend"}, {"name": "Frontend"}]
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=expected)

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        neo4j = _make_neo4j()
        neo4j.session = MagicMock(return_value=mock_session)

        tools = make_graph_tools(neo4j, "org-1")
        tool_fn = _get_graph_query(tools)
        result = await tool_fn.ainvoke(
            {"cypher": "MATCH (c:Category) RETURN c.name AS name LIMIT 10"}
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_returns_error_on_query_failure(self) -> None:
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=RuntimeError("syntax error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        neo4j = _make_neo4j()
        neo4j.session = MagicMock(return_value=mock_session)

        tools = make_graph_tools(neo4j, "org-1")
        tool_fn = _get_graph_query(tools)
        result = await tool_fn.ainvoke(
            {"cypher": "MATCH (n:BadLabel) RETURN n LIMIT 5"}
        )
        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_truncates_to_max_rows(self) -> None:
        big_result = [{"i": i} for i in range(100)]
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=big_result)

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        neo4j = _make_neo4j()
        neo4j.session = MagicMock(return_value=mock_session)

        tools = make_graph_tools(neo4j, "org-1")
        tool_fn = _get_graph_query(tools)
        result = await tool_fn.ainvoke(
            {"cypher": "MATCH (n) RETURN n LIMIT 100"}
        )
        assert len(result) == 50


class TestGraphToolCount:
    def test_returns_eight_tools(self) -> None:
        tools = make_graph_tools(_make_neo4j(), "org-1")
        assert len(tools) == 8
