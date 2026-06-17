"""Tests for the /neo4j/health HTTP endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.api.neo4j_routes import router as neo4j_router


def _make_app_with_state(state_value: Any) -> FastAPI:
    """Build a minimal FastAPI app with a /neo4j/health route and an
    arbitrary value stashed on app.state.neo4j."""
    app = FastAPI()
    app.include_router(neo4j_router)
    app.state.neo4j = state_value
    return app


class TestNeo4jHealthEndpoint:
    def test_returns_200_when_neo4j_not_configured(self) -> None:
        app = _make_app_with_state(None)
        with TestClient(app) as client:
            response = client.get("/neo4j/health")
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "configured": False,
            "healthy": False,
            "reason": "Neo4j not configured (NEO4J_URL or NEO4J_PASSWORD missing)",
        }

    def test_returns_healthy_true_when_neo4j_responds(self) -> None:
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=True)
        app = _make_app_with_state(mock_client)
        with TestClient(app) as client:
            response = client.get("/neo4j/health")
        assert response.status_code == 200
        assert response.json() == {"configured": True, "healthy": True}

    def test_returns_healthy_false_when_neo4j_unreachable(self) -> None:
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=False)
        app = _make_app_with_state(mock_client)
        with TestClient(app) as client:
            response = client.get("/neo4j/health")
        assert response.status_code == 200
        body = response.json()
        assert body == {"configured": True, "healthy": False}

    def test_calls_client_health_check(self) -> None:
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=True)
        app = _make_app_with_state(mock_client)
        with TestClient(app) as client:
            client.get("/neo4j/health")
        mock_client.health_check.assert_awaited_once()
