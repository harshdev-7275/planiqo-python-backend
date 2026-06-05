"""Embedding client — turns issue titles/descriptions into vectors for the
Neo4j vector index (Sprint 2.2).

The default backend is a no-op that returns a fixed-dim zero vector. This
means a dev / CI / a deploy-without-GOOGLE_AI_API_KEY works out of the
box — dedup just falls back to the word-overlap path. Setting
``EMBEDDING_PROVIDER=google`` (with a free Google AI Studio key) flips
to real semantic embeddings.

Per AIService.md, transport errors are LOGGED and re-raised as a
``RuntimeError`` so the caller (the bot tool) can decide to skip the
embedding rather than crash the chat. We do NOT swallow at this layer —
the noop client already covers the "no provider" case; the Google client
should only run when explicitly configured, and a Google outage is a
real signal worth surfacing.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx
from loguru import logger


class EmbeddingClient(Protocol):
    """Contract every embedding backend must satisfy."""

    async def embed(self, text: str) -> list[float]:
        """Return a fixed-dim vector for ``text``. Callers may cache the
        result keyed by text to amortise the network round-trip."""
        ...


class NoopEmbeddingClient:
    """Returns a zero vector of the schema's declared dim (768). Used as
    the default so the bot works without a Google API key. The vector
    index sees a real (if meaningless) embedding; the word-overlap
    pre-filter in ``hybrid_similar_issues`` does the actual work."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim


class GoogleEmbeddingClient:
    """Calls the Google Generative Language ``embedContent`` endpoint.

    Free tier: 1.5K req/day via Google AI Studio. The endpoint expects
    the API key in the URL (``?key=...``) — that's the standard pattern
    for the free tier; the paid tier uses Bearer auth. We default to
    ``text-embedding-004`` (768 dims) to match the schema's
    ``vector.dimensions``. ``gemini-embedding-001`` is the newer model
    but its dim depends on the request's ``output_dimensionality``;
    the fixed-dim model is simpler to test.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        transport: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # The transport is injectable for testability (httpx.MockTransport).
        # If not provided, build a real client.
        self._owns_client = transport is None
        self._client = httpx.AsyncClient(transport=transport, timeout=10.0)

    async def embed(self, text: str) -> list[float]:
        url = (
            f"{self.BASE_URL}/models/{self._model}"
            f":embedContent?key={self._api_key}"
        )
        body = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
        }
        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, httpx.RequestError, json.JSONDecodeError) as e:
            logger.warning("embedding: google embedContent failed: {}", e)
            raise RuntimeError(f"embedding failed: {e}") from e
        values: list[float] = data["embedding"]["values"]
        return values

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def get_embedding_client() -> EmbeddingClient:
    """Factory: return the configured embedding backend.

    The default is ``NoopEmbeddingClient`` so dev never needs an API key.
    Setting ``EMBEDDING_PROVIDER=google`` returns ``GoogleEmbeddingClient``
    (requires ``GOOGLE_AI_API_KEY``). An unknown value raises — fail fast
    at startup so a typo in the env does not silently degrade the dedup
    story.
    """
    from config.settings import settings

    provider = settings.EMBEDDING_PROVIDER
    if provider == "noop":
        return NoopEmbeddingClient()
    if provider == "google":
        if not settings.GOOGLE_AI_API_KEY:
            raise ValueError(
                "EMBEDDING_PROVIDER=google requires GOOGLE_AI_API_KEY"
            )
        return GoogleEmbeddingClient(api_key=settings.GOOGLE_AI_API_KEY)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        f"Expected 'noop' or 'google'."
    )


# Module-level singleton (mirrors ``usage_store``). Callers can pass an
# override; the default is whatever the factory returns.
embedding_client: EmbeddingClient = get_embedding_client()
