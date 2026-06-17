"""Root endpoint tests — service info."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestRoot:
    """GET / — service identity."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_service_info(self, client: TestClient) -> None:
        body = client.get("/").json()
        assert "name" in body
        assert "version" in body
        assert "environment" in body
        assert "docs_url" in body


class TestDocsAvailable:
    """OpenAPI / Swagger UI must be reachable in dev-friendly setups."""

    def test_swagger_ui_loads(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_openapi_schema_loads(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"]
        assert "paths" in schema


class TestExceptionHandler:
    """Uncaught exceptions must return a structured JSON 500, not HTML."""

    def test_404_returns_json(self, client: TestClient) -> None:
        response = client.get("/this-does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert "detail" in body
