"""Health endpoint tests — liveness and readiness probes."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestLiveness:
    """GET /health — process is up. Always 200 unless the interpreter is dead."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body == {"status": "ok"}


class TestReadiness:
    """GET /health/ready — ready to accept traffic."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_returns_status_ready(self, client: TestClient) -> None:
        body = client.get("/health/ready").json()
        assert body["status"] == "ready"
        assert "app" in body
        assert "version" in body
        assert "environment" in body
