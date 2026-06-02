from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from middleware.auth import InternalAuthMiddleware

VALID_SECRET = "test-internal-secret"


def _make_app() -> FastAPI:
    """Minimal app: middleware + one exempt /health + one protected /protected."""
    app = FastAPI()
    app.add_middleware(InternalAuthMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/protected")
    async def protected():
        return {"data": "ok"}

    return app


@pytest.fixture(autouse=True)
def patch_secret():
    mock = MagicMock()
    mock.INTERNAL_SECRET = VALID_SECRET
    with patch("middleware.auth.settings", mock):
        yield


async def test_valid_secret_passes():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected", headers={"X-Internal-Secret": VALID_SECRET})
    assert response.status_code == 200


async def test_missing_header_returns_401():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


async def test_wrong_secret_returns_401():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected", headers={"X-Internal-Secret": "wrong-secret"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


async def test_health_exempt_without_secret():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200


async def test_empty_string_secret_returns_401():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected", headers={"X-Internal-Secret": ""})
    assert response.status_code == 401


async def test_options_preflight_exempt():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.options("/protected")
    assert response.status_code != 401
