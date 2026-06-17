"""Tests for graph schema bootstrap."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from ai_service.graph.schema import (
    _CONSTRAINTS,
    bootstrap_schema,
)
from ai_service.neo4j import Neo4jClient


def _make_neo4j_client(run_calls: list[str]) -> Neo4jClient:
    """Build a Neo4jClient whose session records run() calls."""
    sess = MagicMock()

    async def record_run(query: str, **_kw: object) -> MagicMock:
        run_calls.append(query)
        return MagicMock()

    sess.run = record_run
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=None)

    driver = MagicMock()
    driver.session.return_value = sess

    client = MagicMock(spec=Neo4jClient)
    client._driver = cast("Any", driver)
    client.session.return_value = sess
    return client  # type: ignore[return-value]


class TestBootstrapSchema:
    async def test_runs_one_query_per_constraint(self) -> None:
        calls: list[str] = []
        client = _make_neo4j_client(calls)
        await bootstrap_schema(client)
        assert len(calls) == len(_CONSTRAINTS)

    async def test_every_query_uses_if_not_exists(self) -> None:
        calls: list[str] = []
        client = _make_neo4j_client(calls)
        await bootstrap_schema(client)
        for cypher in calls:
            assert "IF NOT EXISTS" in cypher, f"Missing IF NOT EXISTS in: {cypher}"

    async def test_creates_constraint_for_each_node_label(self) -> None:
        calls: list[str] = []
        client = _make_neo4j_client(calls)
        await bootstrap_schema(client)
        combined = " ".join(calls)
        for label in ("Org", "Project", "Category", "Sprint", "Issue", "User"):
            assert f":{label}" in combined or f"(n:{label})" in combined, (
                f"No constraint for label {label}"
            )

    async def test_all_constraints_require_id_uniqueness(self) -> None:
        calls: list[str] = []
        client = _make_neo4j_client(calls)
        await bootstrap_schema(client)
        for cypher in calls:
            assert "n.id IS UNIQUE" in cypher, f"Expected n.id IS UNIQUE in: {cypher}"

    async def test_bootstrap_is_idempotent(self) -> None:
        """Running bootstrap twice should produce the same queries — no side-effects."""
        calls: list[str] = []
        client = _make_neo4j_client(calls)
        await bootstrap_schema(client)
        first_run = list(calls)
        calls.clear()
        await bootstrap_schema(client)
        assert calls == first_run
