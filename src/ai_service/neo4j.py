"""Neo4j client and destructive operations.

The reset_graph() method is the only built-in destructive operation. It is
guarded by three independent conditions that ALL must hold:

    1. caller passes confirm=True
    2. NEO4J_ALLOW_DESTRUCTIVE=true is set in env
    3. APP_ENV is "development"

If any one is missing, the call raises ConfigurationError. The driver is
never opened against a real database in tests (mocks only).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from ai_service.config import Settings
from ai_service.core.errors import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResetResult:
    """Result of a reset_graph() call.

    All counts are integers. Counts may be None if Neo4j returned a non-numeric
    or missing value (defensive: never crash on a wipe).
    """

    nodes_before: int
    relationships_before: int
    nodes_after: int
    relationships_after: int
    constraints_dropped: int
    indexes_dropped: int

    def to_dict(self) -> dict[str, int]:
        return {
            "nodes_before": self.nodes_before,
            "relationships_before": self.relationships_before,
            "nodes_after": self.nodes_after,
            "relationships_after": self.relationships_after,
            "constraints_dropped": self.constraints_dropped,
            "indexes_dropped": self.indexes_dropped,
        }


class Neo4jClient:
    """Async Neo4j driver wrapper with safety guards for destructive ops.

    The driver is created lazily in `connect()`. Tests inject a mock via
    `client._driver = mock` to bypass network I/O.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.neo4j_configured:
            raise ConfigurationError(
                "Neo4j is not configured. Set NEO4J_URL and NEO4J_PASSWORD in env.",
            )
        self._settings = settings
        self._driver: AsyncDriver | None = None
        # Do NOT log the password or full URL (URLs can contain creds).

    async def connect(self) -> None:
        """Open the driver. Safe to call multiple times — no-op if already open."""
        if self._driver is not None:
            return
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_url,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            max_connection_pool_size=self._settings.neo4j_max_connection_pool_size,
            connection_timeout=self._settings.neo4j_connection_timeout_sec,
        )
        logger.info(
            "neo4j.connected",
            extra={
                "database": self._settings.neo4j_database,
                "user": self._settings.neo4j_user,
                "max_pool_size": self._settings.neo4j_max_connection_pool_size,
            },
        )

    async def close(self) -> None:
        """Close the driver. Idempotent."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j.closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session against the configured database."""
        if self._driver is None:
            await self.connect()
        assert self._driver is not None
        async with self._driver.session(database=self._settings.neo4j_database) as sess:
            yield sess

    async def health_check(self) -> bool:
        """Return True if Neo4j responds to a trivial query."""
        try:
            async with self.session() as sess:
                result = await sess.run("RETURN 1 AS ok")
                record = await result.single()
                return record is not None and record.get("ok") == 1
        except ServiceUnavailable:
            logger.warning("neo4j.health.unavailable")
            return False
        except Exception:
            logger.exception("neo4j.health.error")
            return False

    # ------------------------------------------------------------------
    # Destructive operations
    # ------------------------------------------------------------------

    def _assert_destructive_allowed(self, *, confirm: bool) -> None:
        """Enforce the three required safety guards. Raise if any fail."""
        if not confirm:
            raise ConfigurationError(
                "Destructive Neo4j operations require explicit confirmation. "
                "Pass confirm=True to the call.",
            )
        if not self._settings.neo4j_allow_destructive:
            raise ConfigurationError(
                "Destructive Neo4j operations are disabled. "
                "Set NEO4J_ALLOW_DESTRUCTIVE=true in env to enable.",
            )
        if self._settings.app_env != "development":
            raise ConfigurationError(
                f"Destructive Neo4j operations are forbidden in "
                f"{self._settings.app_env!r}. They are only allowed in development.",
            )

    async def reset_graph(self, *, confirm: bool = False) -> ResetResult:
        """Wipe the entire graph: all nodes, all relationships, all constraints,
        all indexes. IRREVERSIBLE.

        Safety guards (all required):
            1. `confirm=True` must be passed.
            2. NEO4J_ALLOW_DESTRUCTIVE=true must be set in env.
            3. APP_ENV must be "development".

        Returns a ResetResult with before/after counts.
        """
        self._assert_destructive_allowed(confirm=confirm)

        if self._driver is None:
            await self.connect()
        assert self._driver is not None

        logger.warning("neo4j.reset.starting", extra={"database": self._settings.neo4j_database})

        async with self.session() as sess:
            # 1. Count nodes + relationships BEFORE
            nodes_before, rels_before = await self._counts(sess)

            # 2. Drop all constraints
            constraints_dropped = await self._drop_all_constraints(sess)

            # 3. Drop all indexes
            indexes_dropped = await self._drop_all_indexes(sess)

            # 4. Wipe all nodes and relationships
            await sess.run("MATCH (n) DETACH DELETE n")

            # 5. Count nodes + relationships AFTER
            nodes_after, rels_after = await self._counts(sess)

        result = ResetResult(
            nodes_before=nodes_before,
            relationships_before=rels_before,
            nodes_after=nodes_after,
            relationships_after=rels_after,
            constraints_dropped=constraints_dropped,
            indexes_dropped=indexes_dropped,
        )
        logger.warning("neo4j.reset.complete", extra=result.to_dict())
        return result

    @staticmethod
    async def _counts(sess: AsyncSession) -> tuple[int, int]:
        """Return (node_count, relationship_count) as integers."""
        nodes = await Neo4jClient._single_int(sess, "MATCH (n) RETURN count(n) AS n", "n")
        rels = await Neo4jClient._single_int(sess, "MATCH ()-[r]->() RETURN count(r) AS r", "r")
        return nodes, rels

    @staticmethod
    async def _single_int(sess: AsyncSession, query: str, key: str) -> int:
        """Run a query that returns a single integer field. Returns 0 on no result."""
        result = await sess.run(query)
        record = await result.single()
        if record is None:
            return 0
        return int(record[key] or 0)

    @staticmethod
    async def _drop_all_constraints(sess: AsyncSession) -> int:
        """Drop every constraint in the database. Returns the count dropped."""
        records = await (await sess.run("SHOW CONSTRAINTS")).values()
        names: list[Any] = [r[0] for r in records if r and r[0]]
        for name in names:
            # Names can be quoted with backticks if they contain special chars.
            await sess.run(f"DROP CONSTRAINT `{name}` IF EXISTS")
        return len(names)

    @staticmethod
    async def _drop_all_indexes(sess: AsyncSession) -> int:
        """Drop every index in the database, except the system ones that
        Neo4j manages automatically (token indexes, lookup indexes, etc.).
        Returns the count dropped.
        """
        records = await (await sess.run("SHOW INDEXES")).values()
        # SHOW INDEXES columns: name, type, entityType, labelsOrTypes, properties, ...
        # We drop everything except the system ones to avoid breaking the DB.
        dropped = 0
        for row in records:
            name = row[0]
            index_type = row[1] if len(row) > 1 else ""
            # System indexes have type "LOOKUP" or "TEXT" and no labels — skip them.
            if index_type in {"LOOKUP", "TEXT"} and not (row[3] if len(row) > 3 else None):
                continue
            if not name:
                continue
            await sess.run(f"DROP INDEX `{name}` IF EXISTS")
            dropped += 1
        return dropped


__all__ = ["Neo4jClient", "ResetResult"]
