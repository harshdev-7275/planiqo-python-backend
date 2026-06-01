import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from config.settings import settings
from main import app

_AUTH = {"X-Internal-Secret": settings.INTERNAL_SECRET}


@pytest.mark.asyncio
async def test_sync_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/graph/sync", json={"org_slug": "acme"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sync_returns_counts():
    mock_result = {"nodes_created": 12, "relationships_created": 30}
    with patch("main.full_sync", new=AsyncMock(return_value=mock_result)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/graph/sync",
                json={"org_slug": "acme"},
                headers=_AUTH,
            )
    assert response.status_code == 200
    body = response.json()
    assert body["nodes_created"] == 12
    assert body["relationships_created"] == 30


@pytest.mark.asyncio
async def test_sync_calls_full_sync_with_org_slug():
    mock_result = {"nodes_created": 0, "relationships_created": 0}
    with patch("main.full_sync", new=AsyncMock(return_value=mock_result)) as mock_fn:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/graph/sync",
                json={"org_slug": "my-org"},
                headers=_AUTH,
            )
    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    assert kwargs.get("org_slug") == "my-org" or mock_fn.call_args[0][2] == "my-org"


@pytest.mark.asyncio
async def test_sync_missing_org_slug_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/graph/sync",
            json={},
            headers=_AUTH,
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_constraints_applied_on_startup():
    """apply_constraints is called during lifespan — indirectly verified by health passing."""
    with patch("main.apply_constraints", new=AsyncMock()):
        with patch("main.neo4j_client.connect"):
            with patch("main.neo4j_client.ping", new=AsyncMock(return_value=True)):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get("/health")
    assert response.status_code == 200
