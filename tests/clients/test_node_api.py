"""Tests for NodeAPIClient header construction (item 17)."""

from __future__ import annotations

from clients.node_api import node_api_client


def test_bot_headers_includes_user_id() -> None:
    headers = node_api_client._bot_headers("u-123")
    assert headers["X-Bot-User-Id"] == "u-123"
    assert "Idempotency-Key" not in headers


def test_bot_headers_includes_idempotency_key_when_given() -> None:
    headers = node_api_client._bot_headers("u-123", idempotency_key="abc123")
    assert headers["X-Bot-User-Id"] == "u-123"
    assert headers["Idempotency-Key"] == "abc123"


def test_bot_headers_omits_idempotency_key_when_absent() -> None:
    assert "Idempotency-Key" not in node_api_client._bot_headers(None)
    assert node_api_client._bot_headers(None) == {}
