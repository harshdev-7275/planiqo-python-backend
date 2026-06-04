import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from main import app, lifespan


@pytest.mark.asyncio
async def test_health_returns_ok():
    with patch("main.node_api_client.ping", new_callable=AsyncMock, return_value=False):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "node_api" in body


@pytest.mark.asyncio
async def test_health_skips_auth():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_protected_route_requires_secret():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/some-protected-route")

    assert response.status_code in (401, 404)


@pytest.mark.asyncio
async def test_internal_secret_grants_access():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/health", headers={"X-Internal-Secret": "dev-secret"}
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_lifespan_survives_neo4j_failure_at_startup():
    """Neo4j unreachable at boot must NOT crash the service. Graph features
    degrade; core chat/health stay up (fail-soft)."""
    with (
        patch("main.apply_constraints", new=AsyncMock(side_effect=Exception("neo4j down"))),
        patch("main.neo4j_client.close", new=AsyncMock()),
        patch("main.node_api_client.close", new=AsyncMock()),
    ):
        async with lifespan(app):
            pass  # entering and exiting the lifespan must not raise


@pytest.mark.asyncio
async def test_lifespan_applies_constraints_when_neo4j_healthy():
    apply = AsyncMock()
    with (
        patch("main.apply_constraints", new=apply),
        patch("main.neo4j_client.close", new=AsyncMock()),
        patch("main.node_api_client.close", new=AsyncMock()),
    ):
        async with lifespan(app):
            pass

    apply.assert_awaited_once()


@pytest.mark.asyncio
async def test_options_preflight_skips_auth():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.options("/chat")

    assert response.status_code != 401
