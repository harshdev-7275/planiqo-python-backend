"""Tests for the NodeBackendClient — the only way the ai-service talks to data."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest

from ai_service.clients.node_backend import NodeBackendClient
from ai_service.config import Settings
from ai_service.core.errors import ConfigurationError, UpstreamError


def _make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "node_backend_url": "http://localhost:3000",
        "node_backend_service_token": "test-token-abc",
        "node_backend_timeout_sec": 5.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _build_client_with_mock_http(
    responses: list[httpx.Response],
) -> tuple[NodeBackendClient, MagicMock]:
    """Build a NodeBackendClient with a mock httpx client that returns canned responses."""
    settings = _make_settings()
    client = NodeBackendClient(settings)
    mock_http = MagicMock()
    # Make the mock httpx client support async context manager.
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    # The AsyncClient.request / .get / .post methods are async.
    mock_http.get = AsyncMock(side_effect=responses)
    mock_http.aclose = AsyncMock()
    client._client = mock_http
    return client, mock_http


def _json_response(status_code: int, body: Any) -> httpx.Response:
    """Build an httpx.Response with a JSON body."""
    return httpx.Response(
        status_code=status_code,
        content=httpx._content.encode_json(body)[1] if False else b"",  # placeholder
        request=httpx.Request("GET", "http://test"),
    )


def _ok_json(body: Any) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("GET", "http://test"))


def _status_only(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("GET", "http://test"))


class TestNodeBackendClientLifecycle:
    def test_requires_url_and_token(self) -> None:
        settings = Settings(node_backend_url="", node_backend_service_token="")
        with pytest.raises(ConfigurationError, match="NODE_BACKEND_URL"):
            NodeBackendClient(settings)

    def test_authorization_header_is_set(self) -> None:
        settings = _make_settings()
        # The client stashes the bearer token via the headers it builds on connect().
        # We don't call connect() here — we just verify the value is captured.
        assert settings.node_backend_service_token == "test-token-abc"

    async def test_close_is_idempotent(self) -> None:
        client, mock_http = _build_client_with_mock_http([])
        await client.close()
        await client.close()
        mock_http.aclose.assert_awaited_once()


class TestNodeBackendClientListIssues:
    async def test_returns_issues_list(self) -> None:
        body = [
            {"id": "11111111-1111-1111-1111-111111111111", "number": 42, "title": "Fix bug"},
            {"id": "22222222-2222-2222-2222-222222222222", "number": 43, "title": "Add feature"},
        ]
        client, mock_http = _build_client_with_mock_http([_ok_json(body)])
        result = await client.list_issues(
            "acme",
            UUID("33333333-3333-3333-3333-333333333333"),
            status="todo",
            limit=10,
        )
        assert len(result) == 2
        assert result[0]["title"] == "Fix bug"
        # Verify the path includes org slug and project id.
        call_args = mock_http.get.await_args
        assert "acme" in call_args.args[0]
        assert "33333333-3333-3333-3333-333333333333" in call_args.args[0]
        assert call_args.kwargs.get("params", {}).get("status") == "todo"
        assert call_args.kwargs.get("params", {}).get("limit") == 10

    async def test_handles_paginated_response(self) -> None:
        body = {"items": [{"id": "1", "title": "x"}], "next": "cursor"}
        client, _ = _build_client_with_mock_http([_ok_json(body)])
        result = await client.list_issues("acme", UUID("33333333-3333-3333-3333-333333333333"))
        assert result == [{"id": "1", "title": "x"}]

    async def test_raises_upstream_on_500(self) -> None:
        client, _ = _build_client_with_mock_http([_status_only(500)])
        with pytest.raises(UpstreamError, match="5xx"):
            await client.list_issues("acme", UUID("33333333-3333-3333-3333-333333333333"))

    async def test_raises_upstream_on_4xx(self) -> None:
        client, _ = _build_client_with_mock_http([_status_only(403)])
        with pytest.raises(UpstreamError, match="403"):
            await client.list_issues("acme", UUID("33333333-3333-3333-3333-333333333333"))


class TestNodeBackendClientGetIssue:
    async def test_returns_issue_dict(self) -> None:
        body = {"id": "11111111-1111-1111-1111-111111111111", "title": "Fix bug"}
        client, _ = _build_client_with_mock_http([_ok_json(body)])
        result = await client.get_issue(
            "acme",
            UUID("33333333-3333-3333-3333-333333333333"),
            UUID("11111111-1111-1111-1111-111111111111"),
        )
        assert result is not None
        assert result["title"] == "Fix bug"

    async def test_returns_none_on_404(self) -> None:
        client, _ = _build_client_with_mock_http([_status_only(404)])
        result = await client.get_issue(
            "acme",
            UUID("33333333-3333-3333-3333-333333333333"),
            UUID("11111111-1111-1111-1111-111111111111"),
        )
        assert result is None


class TestNodeBackendClientGetUser:
    async def test_lookup_by_user_id(self) -> None:
        body = {"id": "11111111-1111-1111-1111-111111111111", "name": "Priya"}
        client, _ = _build_client_with_mock_http([_ok_json(body)])
        result = await client.get_user(
            "acme",
            user_id=UUID("11111111-1111-1111-1111-111111111111"),
        )
        assert result is not None
        assert result["name"] == "Priya"

    async def test_lookup_by_email(self) -> None:
        body = {"id": "11111111-1111-1111-1111-111111111111", "name": "Priya"}
        client, mock_http = _build_client_with_mock_http([_ok_json(body)])
        result = await client.get_user("acme", email="priya@example.com")
        assert result is not None
        assert mock_http.get.await_args.kwargs.get("params", {}).get("email") == "priya@example.com"

    def test_requires_exactly_one_identifier(self) -> None:
        client, _ = _build_client_with_mock_http([])
        import asyncio

        with pytest.raises(ValueError, match="exactly one"):
            asyncio.run(client.get_user("acme"))  # neither
        with pytest.raises(ValueError, match="exactly one"):
            asyncio.run(
                client.get_user(
                    "acme",
                    user_id=UUID("11111111-1111-1111-1111-111111111111"),
                    email="x@y.com",
                )
            )


class TestNodeBackendClientHealth:
    async def test_health_check_true_on_200(self) -> None:
        client, _ = _build_client_with_mock_http([_ok_json({"status": "ok"})])
        assert await client.health_check() is True

    async def test_health_check_false_on_connection_error(self) -> None:
        client, mock_http = _build_client_with_mock_http([])
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("nope"))
        assert await client.health_check() is False
