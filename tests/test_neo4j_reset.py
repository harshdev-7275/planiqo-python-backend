"""Tests for the destructive Neo4j reset_graph safety guards.

The reset function is irreversible — these tests are the contract that
prevents accidental wipes. Every guard is a unit test against a mocked driver
so we never touch a real database.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import ServiceUnavailable

from ai_service.config import Settings
from ai_service.core.errors import ConfigurationError
from ai_service.neo4j import Neo4jClient, ResetResult


class _SessionMock:
    """Async context-manager mock for a Neo4j session.

    Responds to common query types (counts, SHOW CONSTRAINTS, SHOW INDEXES,
    MATCH (n) DETACH DELETE n, DROP statements) with appropriate values.
    Records every run() call so tests can assert on the Cypher issued.
    Decrements node/rel counts on DETACH DELETE so "after" reflects the wipe.
    """

    def __init__(
        self,
        *,
        node_count: int = 0,
        rel_count: int = 0,
        constraint_names: list[str] | None = None,
        index_rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self._node_count = node_count
        self._rel_count = rel_count
        self._constraint_names = constraint_names or []
        self._index_rows = index_rows or []
        self.run_calls: list[str] = []

    async def __aenter__(self) -> _SessionMock:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def run(self, query: str, *_args: object, **_kwargs: object) -> MagicMock:
        self.run_calls.append(query)
        result = MagicMock()

        if "MATCH (n) RETURN count(n)" in query:
            result.single = AsyncMock(return_value={"n": self._node_count})
        elif "MATCH ()-[r]->() RETURN count(r)" in query:
            result.single = AsyncMock(return_value={"r": self._rel_count})
        elif "SHOW CONSTRAINTS" in query:
            result.values = AsyncMock(return_value=[(name,) for name in self._constraint_names])
        elif "SHOW INDEXES" in query:
            result.values = AsyncMock(return_value=self._index_rows)
        elif "DETACH DELETE" in query:
            self._node_count = 0
            self._rel_count = 0
            result.single = AsyncMock(return_value=None)
        else:
            result.single = AsyncMock(return_value=None)

        return result


def _make_client(
    *,
    env: str = "development",
    allow_destructive: bool = True,
    url: str = "neo4j+s://test.databases.neo4j.io",
    password: str = "test-password",
    session: _SessionMock | None = None,
) -> Neo4jClient:
    """Build a Neo4jClient whose driver is fully mocked — no network I/O."""
    settings = Settings(
        app_env=env,
        neo4j_allow_destructive=allow_destructive,
        neo4j_url=url,
        neo4j_password=password,
    )
    client = Neo4jClient(settings)
    if session is None:
        session = _SessionMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value = session
    mock_driver.close = AsyncMock()
    # Inject the mock — bypasses the real driver's network I/O. mypy sees _driver
    # as AsyncDriver | None, but the cast below is a no-op at runtime.
    client._driver = cast("Any", mock_driver)
    return client


# =====================================================================
# Safety guards — these are the most important tests in the file
# =====================================================================


class TestResetGraphSafetyGuards:
    """The reset function is the most dangerous operation in the service.
    These tests enforce the three required safety conditions."""

    def test_refuses_when_confirm_is_false(self) -> None:
        session = _SessionMock()
        client = _make_client(session=session)
        with pytest.raises(ConfigurationError, match="confirm=True"):
            asyncio.run(client.reset_graph())
        # The driver must not be touched — no Cypher should have been issued.
        assert session.run_calls == []

    def test_refuses_when_allow_destructive_is_false(self) -> None:
        client = _make_client(allow_destructive=False)
        with pytest.raises(ConfigurationError, match="NEO4J_ALLOW_DESTRUCTIVE"):
            asyncio.run(client.reset_graph(confirm=True))

    def test_refuses_in_production_environment(self) -> None:
        client = _make_client(env="production", allow_destructive=True)
        with pytest.raises(ConfigurationError, match="production"):
            asyncio.run(client.reset_graph(confirm=True))

    def test_refuses_in_staging_environment(self) -> None:
        """Production AND staging are no-go zones. Only development allows destructive ops."""
        client = _make_client(env="staging", allow_destructive=True)
        with pytest.raises(ConfigurationError, match="staging"):
            asyncio.run(client.reset_graph(confirm=True))

    def test_succeeds_in_development_with_all_confirmations(self) -> None:
        session = _SessionMock(node_count=5, rel_count=3, constraint_names=["c1"], index_rows=[])
        client = _make_client(session=session)
        result = asyncio.run(client.reset_graph(confirm=True))
        assert isinstance(result, ResetResult)
        assert result.nodes_before == 5
        assert result.relationships_before == 3
        assert result.nodes_after == 0
        assert result.relationships_after == 0
        assert result.constraints_dropped == 1
        assert result.indexes_dropped == 0


# =====================================================================
# Execution — when guards pass, the right Cypher must run
# =====================================================================


class TestResetGraphExecution:
    def test_runs_detach_delete_cypher(self) -> None:
        session = _SessionMock()
        client = _make_client(session=session)
        asyncio.run(client.reset_graph(confirm=True))
        assert any("MATCH (n)" in c and "DETACH DELETE" in c for c in session.run_calls), (
            f"Expected MATCH (n) DETACH DELETE in Cypher, got: {session.run_calls}"
        )

    def test_drops_all_constraints(self) -> None:
        session = _SessionMock(constraint_names=["c_unique_email", "c_node_key"])
        client = _make_client(session=session)
        result = asyncio.run(client.reset_graph(confirm=True))
        assert result.constraints_dropped == 2
        drop_calls = [c for c in session.run_calls if "DROP CONSTRAINT" in c]
        assert len(drop_calls) == 2

    def test_drops_user_indexes_but_keeps_system_indexes(self) -> None:
        index_rows = [
            ("system_lookup_idx", "LOOKUP", "NODE", None, None),  # system — skip
            ("user_btree_idx", "BTREE", "NODE", ["Issue"], ["status"]),  # user — drop
        ]
        session = _SessionMock(index_rows=index_rows)
        client = _make_client(session=session)
        result = asyncio.run(client.reset_graph(confirm=True))
        assert result.indexes_dropped == 1
        drop_calls = [c for c in session.run_calls if "DROP INDEX" in c]
        assert len(drop_calls) == 1
        assert "user_btree_idx" in drop_calls[0]
        assert "system_lookup_idx" not in " ".join(drop_calls)

    def test_reports_node_counts_before_and_after(self) -> None:
        session = _SessionMock(node_count=42, rel_count=17)
        client = _make_client(session=session)
        result = asyncio.run(client.reset_graph(confirm=True))
        assert result.nodes_before == 42
        assert result.relationships_before == 17
        assert result.nodes_after == 0
        assert result.relationships_after == 0
        count_queries = [
            c
            for c in session.run_calls
            if "MATCH (n) RETURN count(n)" in c or "MATCH ()-[r]->() RETURN count(r)" in c
        ]
        assert len(count_queries) == 4  # nodes-before, rels-before, nodes-after, rels-after


# =====================================================================
# Lifecycle
# =====================================================================


class TestClientLifecycle:
    def test_client_requires_url_and_password(self) -> None:
        settings = Settings(neo4j_url="", neo4j_password="")
        with pytest.raises(ConfigurationError, match="NEO4J_URL"):
            Neo4jClient(settings)

    async def test_close_is_idempotent(self) -> None:
        client = _make_client()
        await client.close()
        await client.close()
        assert client._driver is None


# =====================================================================
# Health check
# =====================================================================


class TestHealthCheck:
    async def test_health_check_returns_true_when_neo4j_responds(self) -> None:
        session = _SessionMock()
        original_run = session.run

        async def run_with_return(query: str, *args: object, **kwargs: object) -> MagicMock:
            if "RETURN 1" in query:
                result = MagicMock()
                result.single = AsyncMock(return_value={"ok": 1})
                return result
            return await original_run(query, *args, **kwargs)

        session.run = run_with_return  # type: ignore[method-assign]
        client = _make_client(session=session)
        assert await client.health_check() is True

    async def test_health_check_returns_false_on_service_unavailable(self) -> None:
        client = _make_client()
        driver = cast("Any", client._driver)
        sess = driver.session.return_value

        async def raise_unavailable(*_a: object, **_k: object) -> object:
            raise ServiceUnavailable("neo4j is down")  # type: ignore[no-untyped-call]

        sess.run = raise_unavailable
        assert await client.health_check() is False
