"""Tests for graph HTTP endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.api.graph_routes import router
from ai_service.graph.sync import SyncStats


def _make_app(*, neo4j: Any = None, node_backend: Any = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j = neo4j
    app.state.node_backend = node_backend
    return app


def _make_neo4j() -> MagicMock:
    return MagicMock()


def _make_backend() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# POST /v1/graph/sync
# ---------------------------------------------------------------------------


class TestGraphSyncEndpoint:
    def test_returns_503_when_neo4j_not_configured(self) -> None:
        app = _make_app(neo4j=None, node_backend=_make_backend())
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync",
                json={"org_slug": "acme", "org_id": "org-1"},
            )
        assert response.status_code == 503
        assert "Neo4j" in response.json()["detail"]

    def test_returns_503_when_node_backend_not_configured(self) -> None:
        app = _make_app(neo4j=_make_neo4j(), node_backend=None)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync",
                json={"org_slug": "acme", "org_id": "org-1"},
            )
        assert response.status_code == 503
        assert "node-backend" in response.json()["detail"]

    def test_returns_sync_stats_on_success(self) -> None:
        stats = SyncStats(projects=2, categories=5, sprints=3, issues=20, users=4, relationships=40)
        with patch(
            "ai_service.api.graph_routes.GraphSyncService.full_sync",
            new=AsyncMock(return_value=stats),
        ):
            app = _make_app(neo4j=_make_neo4j(), node_backend=_make_backend())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/graph/sync",
                    json={"org_slug": "acme", "org_id": "org-1"},
                )
        assert response.status_code == 200
        body = response.json()
        assert body["projects"] == 2
        assert body["issues"] == 20

    def test_returns_422_for_missing_body_fields(self) -> None:
        app = _make_app(neo4j=_make_neo4j(), node_backend=_make_backend())
        with TestClient(app) as client:
            response = client.post("/v1/graph/sync", json={})
        assert response.status_code == 422

    def test_returns_502_when_sync_raises(self) -> None:
        with patch(
            "ai_service.api.graph_routes.GraphSyncService.full_sync",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            app = _make_app(neo4j=_make_neo4j(), node_backend=_make_backend())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/graph/sync",
                    json={"org_slug": "acme", "org_id": "org-1"},
                )
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# POST /v1/graph/sync/project/:project_id
# ---------------------------------------------------------------------------


class TestGraphProjectSyncEndpoint:
    def test_returns_sync_stats_on_success(self) -> None:
        stats = SyncStats(issues=10, relationships=15)
        with patch(
            "ai_service.api.graph_routes.GraphSyncService.sync_single_issue",
            new=AsyncMock(return_value=stats),
        ):
            app = _make_app(neo4j=_make_neo4j(), node_backend=_make_backend())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/graph/sync/project/11111111-1111-1111-1111-111111111111",
                    json={
                        "org_slug": "acme",
                        "org_id": "org-1",
                        "project_id": "11111111-1111-1111-1111-111111111111",
                    },
                )
        assert response.status_code == 200
        assert response.json()["issues"] == 10

    def test_returns_404_for_invalid_uuid(self) -> None:
        app = _make_app(neo4j=_make_neo4j(), node_backend=_make_backend())
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync/project/not-a-uuid",
                json={"org_slug": "acme", "org_id": "org-1",
                      "project_id": "not-a-uuid"},
            )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/graph/stats
# ---------------------------------------------------------------------------


class TestGraphStatsEndpoint:
    def test_returns_not_configured_when_neo4j_absent(self) -> None:
        app = _make_app(neo4j=None)
        with TestClient(app) as client:
            response = client.get("/v1/graph/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is False
        assert body["nodes"] == {}

    def test_returns_stats_when_neo4j_configured(self) -> None:
        graph_data = {
            "nodes": {"Issue": 42, "User": 5},
            "relationships": {"ASSIGNED_TO": 10},
        }
        with patch(
            "ai_service.api.graph_routes.get_graph_stats",
            new=AsyncMock(return_value=graph_data),
        ):
            app = _make_app(neo4j=_make_neo4j())
            with TestClient(app) as client:
                response = client.get("/v1/graph/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is True
        assert body["nodes"]["Issue"] == 42

    def test_returns_empty_stats_on_error(self) -> None:
        with patch(
            "ai_service.api.graph_routes.get_graph_stats",
            new=AsyncMock(side_effect=RuntimeError("neo4j down")),
        ):
            app = _make_app(neo4j=_make_neo4j())
            with TestClient(app) as client:
                response = client.get("/v1/graph/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is True
        assert body["nodes"] == {}
