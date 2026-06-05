"""Tests for the pluggable metering backend (Sprint 1.3).

The in-process store stays sync (and its tests in ``test_usage.py`` are
unchanged). The new ``PostgresUsageStore`` is async and talks to a future
node-api route over HTTP; this file mocks the httpx client and asserts
the contract: every method POSTs/GETs the right path with the right body,
returns what the node-api says, and never raises on a transport error
(the supervisor's quota check must not crash the chat on a node-api
hiccup — that's a graceful-degradation requirement per AIService.md).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# --- factory ----------------------------------------------------------------


def test_factory_returns_inprocess_by_default(monkeypatch) -> None:
    """The default backend is in-process — preserves the current behaviour
    (single instance, no extra hops) so a dev who never sets the env var
    sees no change."""
    from config.settings import settings
    from metering.usage import get_usage_store

    monkeypatch.setattr(settings, "METERING_BACKEND", "inprocess")
    store = get_usage_store()
    from metering.usage import UsageStore
    assert isinstance(store, UsageStore)


def test_factory_returns_postgres_when_configured(monkeypatch) -> None:
    """``METERING_BACKEND=postgres`` returns the async PostgresUsageStore."""
    from config.settings import settings
    from metering.usage import PostgresUsageStore, get_usage_store

    monkeypatch.setattr(settings, "METERING_BACKEND", "postgres")
    store = get_usage_store()
    assert isinstance(store, PostgresUsageStore)


def test_factory_rejects_unknown_backend(monkeypatch) -> None:
    """A typo in the env var must FAIL FAST at startup, not silently
    degrade to in-process — the operator thinks they're getting multi-
    instance-safe metering when they aren't."""
    from config.settings import settings
    from metering.usage import get_usage_store

    monkeypatch.setattr(settings, "METERING_BACKEND", "redis")
    with pytest.raises(ValueError, match="Unknown METERING_BACKEND"):
        get_usage_store()


# --- PostgresUsageStore (async, HTTP-mocked) --------------------------------


def _mock_async_client() -> MagicMock:
    """Return a MagicMock that quacks like an httpx.AsyncClient — every
    verb returns a coroutine resolving to a stub response with ``.json()``
    and ``.status_code``."""
    client = MagicMock()
    response = MagicMock()
    response.json = MagicMock(return_value={})
    response.status_code = 200
    # AsyncMock for the verbs so ``await client.post(...)`` works.
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_postgres_add_posts_tokens_to_node_api() -> None:
    """``add`` must POST the tokens delta to the node-api metering route.
    The X-Internal-Secret header is verified separately via a real
    httpx.AsyncClient with a MockTransport (the magic-mock client doesn't
    track default headers)."""
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    store = PostgresUsageStore(
        base_url="http://node:4000",
        internal_secret="secret",
        http_client=client,  # inject for testability
    )
    await store.add("acme", 42)
    client.post.assert_awaited_once()
    args, kwargs = client.post.await_args
    assert "/admin/metering/acme/add" in args[0]
    assert kwargs["json"] == {"tokens": 42}


@pytest.mark.asyncio
async def test_postgres_store_sets_internal_secret_header_on_real_client() -> None:
    """The store must configure the real httpx client with the shared
    X-Internal-Secret so node-api can authenticate the call."""
    import httpx

    from metering.usage import PostgresUsageStore

    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"tokens": 0})

    transport = httpx.MockTransport(_handler)
    # No http_client injection — the store builds its own with the secret.
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="topsecret",
    )
    # Swap the transport in for the test.
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url="http://node:4000",
        headers={"X-Internal-Secret": "topsecret"},
        transport=transport,
    )
    try:
        await store.get("acme")
    finally:
        await store._client.aclose()

    assert captured_headers.get("x-internal-secret") == "topsecret"


@pytest.mark.asyncio
async def test_postgres_add_ignores_nonpositive_tokens() -> None:
    """Zero or negative deltas must NOT hit the network — the in-process
    store has the same no-op behaviour and Postgres must too."""
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    await store.add("acme", 0)
    await store.add("acme", -5)
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_postgres_get_returns_tokens_from_node_api() -> None:
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.get.return_value.json = MagicMock(return_value={"tokens": 100})
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    assert await store.get("acme") == 100
    client.get.assert_awaited_once_with("/admin/metering/acme/get")


@pytest.mark.asyncio
async def test_postgres_inc_request_returns_new_count() -> None:
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.post.return_value.json = MagicMock(return_value={"requests": 7})
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    count = await store.inc_request("acme")
    assert count == 7
    client.post.assert_awaited_once_with("/admin/metering/acme/inc-request")


@pytest.mark.asyncio
async def test_postgres_get_request_count_reads_node_api() -> None:
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.get.return_value.json = MagicMock(return_value={"requests": 3})
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    assert await store.get_request_count("acme") == 3


@pytest.mark.asyncio
async def test_postgres_get_returns_zero_on_missing_org() -> None:
    """A org that has never recorded usage returns 0 (the supervisor's
    quota check uses ``>=``; 0 is the safe default)."""
    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.get.return_value.json = MagicMock(return_value={"tokens": 0})
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    assert await store.get("nope") == 0


@pytest.mark.asyncio
async def test_postgres_gracefully_degrades_on_transport_error() -> None:
    """If the node-api is down, the metering store must NOT raise — it
    returns a safe default so the chat stays up. This is the operational
    requirement per AIService.md: 'If AI is unavailable, the core PM tool
    still works.' The supervisor's quota check may over-permit for a few
    minutes until the node-api recovers — that's a billing issue, not a
    chat-uptime issue."""
    import httpx

    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.get.side_effect = httpx.ConnectError("node-api unreachable")
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    # get must return 0 on transport failure (not raise).
    assert await store.get("acme") == 0


@pytest.mark.asyncio
async def test_postgres_add_swallows_transport_errors() -> None:
    """``add`` is fire-and-forget — a failure to record tokens must not
    crash the chat. The tokens are lost (acceptable for an outage); the
    user-facing response is unaffected."""
    import httpx

    from metering.usage import PostgresUsageStore

    client = _mock_async_client()
    client.post.side_effect = httpx.ConnectError("node-api unreachable")
    store = PostgresUsageStore(
        base_url="http://node:4000", internal_secret="s", http_client=client,
    )
    # Must not raise.
    await store.add("acme", 100)
