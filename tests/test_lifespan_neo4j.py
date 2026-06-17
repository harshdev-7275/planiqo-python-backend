"""Tests for the Neo4j startup/shutdown wiring in core/lifespan.py.

The lifespan is hard to test in isolation because it's a context manager
that mutates app.state. We invoke it directly and assert on side effects.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from ai_service.config import Settings
from ai_service.core.lifespan import lifespan


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "app_env": "development",
        "neo4j_url": "",
        "neo4j_password": "",
        "neo4j_database": "neo4j",
        "neo4j_user": "neo4j",
        "neo4j_allow_destructive": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestLifespanNeo4jWiring:
    @pytest.mark.asyncio
    async def test_lifespan_skips_neo4j_when_not_configured(self) -> None:
        settings = _settings()  # no url, no password
        app = FastAPI()
        async with lifespan(app, settings):
            assert app.state.neo4j is None

    @pytest.mark.asyncio
    async def test_lifespan_starts_neo4j_when_configured(self) -> None:
        settings = _settings(
            neo4j_url="neo4j+s://x.databases.neo4j.io",
            neo4j_password="secret",
        )
        app = FastAPI()

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_client.close = AsyncMock()

        # Patch the source module — the lifespan does a lazy import inside the function.
        with patch("ai_service.neo4j.Neo4jClient", return_value=mock_client):
            async with lifespan(app, settings):
                mock_client.connect.assert_awaited_once()
                mock_client.health_check.assert_awaited_once()
                assert app.state.neo4j is mock_client
            mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_survives_neo4j_startup_failure(self) -> None:
        """If Neo4j is down at startup, the service must still start (degraded mode)."""
        settings = _settings(
            neo4j_url="neo4j+s://x.databases.neo4j.io",
            neo4j_password="secret",
        )
        app = FastAPI()

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=ConnectionError("can't reach neo4j"))
        mock_client.close = AsyncMock()

        with patch("ai_service.neo4j.Neo4jClient", return_value=mock_client):
            async with lifespan(app, settings):
                assert app.state.neo4j is mock_client
            mock_client.close.assert_awaited_once()
