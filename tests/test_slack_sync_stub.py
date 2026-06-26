"""Tests for the Phase 0+1 Slack sync stub endpoints.

The real SlackSyncService lands in Phase 2. These tests lock the wire
contract so node-backend's kg_sync_log row closes cleanly today.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.api.graph_routes import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


class TestSlackSyncStubEndpoints:
    def test_full_org_sync_returns_200_with_ack(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync/slack",
                json={"org_slug": "acme", "org_id": "00000000-0000-4000-8000-000000000001"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "accepted"
        assert body["org_id"] == "00000000-0000-4000-8000-000000000001"
        assert body["channel_id"] is None

    def test_full_org_sync_rejects_missing_fields(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/v1/graph/sync/slack", json={})
        assert response.status_code == 422

    def test_channel_sync_returns_200_with_ack(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync/slack/channel/00000000-0000-4000-8000-000000000042",
                json={
                    "org_slug": "acme",
                    "org_id": "00000000-0000-4000-8000-000000000001",
                    "channel_id": "00000000-0000-4000-8000-000000000042",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "accepted"
        assert body["channel_id"] == "00000000-0000-4000-8000-000000000042"

    def test_channel_sync_rejects_bad_channel_uuid(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/graph/sync/slack/channel/not-a-uuid",
                json={
                    "org_slug": "acme",
                    "org_id":   "00000000-0000-4000-8000-000000000001",
                    "channel_id": "00000000-0000-4000-8000-000000000042",
                },
            )
        assert response.status_code == 422