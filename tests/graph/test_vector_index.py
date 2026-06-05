"""Tests for the vector-index semantic dedup (Sprint 2.2).

The current ``find_similar_issues`` (in ``graph/queries.py``) matches on
shared words — "Login fails" vs "Can't sign in" gets zero overlap and
misses the duplicate. The fix is a native Neo4j vector index on
``Issue.embedding`` plus a hybrid retrieval path: word-overlap narrows
the candidate set (cheap pre-filter, uses the existing index), then a
vector kNN refines the ranking.

This file tests:
  1. The schema adds the vector index alongside the existing constraints.
  2. The embedding client interface (noop + google), with a no-op default
     so dev doesn't need an API key.
  3. The hybrid query path takes a precomputed embedding and runs a
     ``db.index.vector.queryNodes`` call against the candidates returned
     by the word-overlap pre-filter.
  4. ``upsert_issue`` accepts an optional embedding and stores it on the
     node so the vector index has data to query.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# --- schema: vector index applied on startup --------------------------------


def test_schema_includes_vector_index_for_issue_embedding() -> None:
    """The vector index DDL must run alongside the existing constraints
    so the index is present on first deploy. Apply is idempotent
    (CREATE … IF NOT EXISTS) so a restart is safe."""
    from graph.schema import VECTOR_INDEXES

    joined = "\n".join(VECTOR_INDEXES)
    assert "issue_embedding" in joined
    assert "Issue" in joined
    # Cypher 5+ native vector index syntax.
    assert "VECTOR INDEX" in joined.upper()


# --- embedding client -------------------------------------------------------


def test_factory_returns_noop_by_default(monkeypatch) -> None:
    """Dev never sees a Google dependency unless they set the API key."""
    from clients.embeddings import NoopEmbeddingClient, get_embedding_client
    from config.settings import settings

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "noop")
    client = get_embedding_client()
    assert isinstance(client, NoopEmbeddingClient)


def test_factory_rejects_unknown_provider(monkeypatch) -> None:
    from config.settings import settings
    from clients.embeddings import get_embedding_client

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        get_embedding_client()


@pytest.mark.asyncio
async def test_noop_embedding_client_returns_zero_vector_of_expected_dim() -> None:
    """The noop client must return a fixed-dim zero vector so callers
    that always call ``embed`` don't need to special-case a missing
    provider. The dim matches the schema (768)."""
    from clients.embeddings import NoopEmbeddingClient

    client = NoopEmbeddingClient(dim=768)
    vec = await client.embed("anything")
    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(v == 0.0 for v in vec)


@pytest.mark.asyncio
async def test_google_embedding_client_uses_genai_endpoint(monkeypatch) -> None:
    """The Google client must POST to the Generative Language API with
    the API key in the URL (the standard pattern for the free tier)."""
    import httpx

    from clients.embeddings import GoogleEmbeddingClient

    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"embedding": {"values": [0.1] * 768}},
        )

    transport = httpx.MockTransport(_handler)
    client = GoogleEmbeddingClient(
        api_key="test-key",
        model="text-embedding-004",
        transport=transport,
    )
    vec = await client.embed("Login page is broken")
    assert len(vec) == 768
    assert "key=test-key" in captured["url"]
    assert "text-embedding-004" in captured["url"]
    assert "Login page is broken" in captured["body"]


@pytest.mark.asyncio
async def test_google_embedding_client_swallows_transport_errors() -> None:
    """A Google outage during an upsert must NOT break the chat — the
    embedding is best-effort. The tool's call site can either skip
    the embedding or fall back to the noop client. (See AIService.md
    graceful-degradation rule.)"""
    import httpx

    from clients.embeddings import GoogleEmbeddingClient

    def _handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("google unreachable")

    transport = httpx.MockTransport(_handler)
    client = GoogleEmbeddingClient(
        api_key="k", model="m", transport=transport,
    )
    with pytest.raises(RuntimeError, match="embedding failed"):
        await client.embed("t")


# --- hybrid_similar_issues -------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_similar_uses_word_overlap_as_prefilter() -> None:
    """The hybrid query must use the existing word-overlap query to
    NARROW the candidate set before doing the vector kNN. This is the
    hybrid-retrieval pattern recommended by the GraphRAG research —
    cheap pre-filter, expensive semantic rerank."""
    from graph.queries import hybrid_similar_issues

    neo = MagicMock()
    # Pre-filter returns two candidates.
    neo.run = AsyncMock(side_effect=[
        # First call: word-overlap candidates (id, number, title)
        [
            {"id": "i1", "number": 1, "title": "Login broken"},
            {"id": "i2", "number": 2, "title": "Auth crash"},
        ],
        # Second call: vector kNN over the candidate ids.
        [
            {"id": "i1", "number": 1, "title": "Login broken", "score": 0.92},
        ],
    ])
    results = await hybrid_similar_issues(
        neo, project_id="proj-1", title="Login is broken",
        embedding=[0.1] * 768, k=5,
    )
    assert len(results) == 1
    assert results[0]["number"] == 1
    # Both queries ran (pre-filter + vector kNN).
    assert neo.run.await_count == 2


@pytest.mark.asyncio
async def test_hybrid_similar_falls_back_to_word_overlap_when_no_candidates() -> None:
    """If the pre-filter returns nothing, the hybrid path is moot —
    the vector kNN has nothing to rank. We return the empty list
    rather than running an expensive full-graph kNN. (The cheap path
    IS the right answer here.)"""
    from graph.queries import hybrid_similar_issues

    neo = MagicMock()
    neo.run = AsyncMock(return_value=[])  # no word-overlap matches
    results = await hybrid_similar_issues(
        neo, project_id="proj-1", title="obscure phrase",
        embedding=[0.0] * 768, k=5,
    )
    assert results == []
    # Only the pre-filter ran.
    assert neo.run.await_count == 1


# --- upsert_issue accepts an embedding --------------------------------------


@pytest.mark.asyncio
async def test_upsert_issue_stores_embedding_when_provided() -> None:
    """``upsert_issue`` must accept an optional embedding and write it
    to the node property so the vector index has data to query on
    subsequent dedup calls."""
    from graph.sync import upsert_issue

    neo = MagicMock()
    neo.run = AsyncMock()
    await upsert_issue(
        neo,
        {
            "id": "issue-uuid-1", "number": 42, "title": "Login broken",
            "type": "bug", "priority": "high", "createdAt": "2026-06-05",
            "completedAt": None, "projectId": "proj-1", "reporterId": "u-1",
            "assigneeId": None, "sprintId": None,
        },
        embedding=[0.1] * 768,
    )
    # The embedding must be in the Cypher params of the issue MERGE call
    # so it lands on the node.
    assert any(
        call.kwargs.get("embedding") == [0.1] * 768
        or (call.args and len(call.args) > 1 and "embedding" in (call.args[1] or {}))
        for call in neo.run.await_args_list
    )


@pytest.mark.asyncio
async def test_upsert_issue_skips_embedding_when_not_provided() -> None:
    """Backwards compat: existing callers (full_sync, etc.) that don't
    compute an embedding must still work — the embedding key is simply
    absent from the Cypher parameters."""
    from graph.sync import upsert_issue

    neo = MagicMock()
    neo.run = AsyncMock()
    await upsert_issue(
        neo,
        {
            "id": "issue-uuid-1", "number": 42, "title": "Login broken",
            "type": "bug", "priority": "high", "createdAt": "2026-06-05",
            "completedAt": None, "projectId": "proj-1", "reporterId": "u-1",
            "assigneeId": None, "sprintId": None,
        },
    )
    # No kwarg "embedding" was passed.
    for call in neo.run.await_args_list:
        kwargs = call.kwargs
        assert "embedding" not in kwargs
