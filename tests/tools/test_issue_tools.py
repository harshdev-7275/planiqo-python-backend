"""Tests for the issue write/similarity tools.

Focus: the Sprint 2.2 vector-index wiring that the write/similarity tools must
actually exercise —
  * CreateIssueTool / UpdateIssueTool embed the title and pass the vector to
    upsert_issue (otherwise the vector index stays empty).
  * FindSimilarIssuesTool uses the hybrid (word-overlap + vector) retrieval,
    not the legacy word-overlap-only path.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.issue_tools import CreateIssueTool, FindSimilarIssuesTool, UpdateIssueTool

CTX = {"org_slug": "acme", "project_id": "proj-1"}


class _FakeEmbeddingClient:
    """Records the texts it was asked to embed and returns a fixed vector."""

    def __init__(self, dim: int = 3) -> None:
        self.calls: list[str] = []
        self._vec = [0.1] * dim

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vec


def _api_for_create() -> AsyncMock:
    api = AsyncMock()
    api.get_default_status = AsyncMock(
        return_value={"id": "s1", "name": "Todo", "isDefault": True}
    )
    api.post = AsyncMock(return_value={"number": 7, "title": "login bug"})
    return api


# --- item 3: embeddings are populated on create -----------------------------


@pytest.mark.asyncio
async def test_create_issue_embeds_title_and_passes_vector_to_upsert(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_upsert(client: Any, issue: dict[str, Any], embedding: Any = None) -> None:
        captured["issue"] = issue
        captured["embedding"] = embedding

    monkeypatch.setattr("tools.issue_tools.upsert_issue", fake_upsert)

    emb = _FakeEmbeddingClient()
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[])  # suggest_assignee path returns nothing
    tool = CreateIssueTool(
        api=_api_for_create(), neo4j_client=neo, embedding_client=emb, **CTX
    )

    result = await tool._arun(title="login bug", type="bug", priority="high")

    assert "Created #7" in result
    # The title was embedded and the vector reached upsert_issue.
    assert emb.calls == ["login bug"]
    assert captured["embedding"] == [0.1, 0.1, 0.1]


@pytest.mark.asyncio
async def test_create_issue_still_succeeds_when_embedding_fails(monkeypatch: Any) -> None:
    """An embedding-provider outage must NOT break the create — the issue is
    upserted with embedding=None and the user still gets the success line."""
    captured: dict[str, Any] = {}

    async def fake_upsert(client: Any, issue: dict[str, Any], embedding: Any = None) -> None:
        captured["embedding"] = embedding

    monkeypatch.setattr("tools.issue_tools.upsert_issue", fake_upsert)

    class _Boom:
        async def embed(self, text: str) -> list[float]:
            raise RuntimeError("google down")

    neo = MagicMock()
    neo.run = AsyncMock(return_value=[])
    tool = CreateIssueTool(
        api=_api_for_create(), neo4j_client=neo, embedding_client=_Boom(), **CTX
    )

    result = await tool._arun(title="login bug")
    assert "Created #7" in result
    assert captured["embedding"] is None


# --- item 3: embeddings are populated on update -----------------------------


@pytest.mark.asyncio
async def test_update_issue_reembeds_on_effective_title(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_upsert(client: Any, issue: dict[str, Any], embedding: Any = None) -> None:
        captured["issue"] = issue
        captured["embedding"] = embedding

    monkeypatch.setattr("tools.issue_tools.upsert_issue", fake_upsert)

    api = AsyncMock()
    api.get_issues = AsyncMock(
        return_value=[{"number": 5, "id": "uuid-5", "title": "old title"}]
    )
    api.patch = AsyncMock(return_value={"number": 5})

    emb = _FakeEmbeddingClient()
    neo = MagicMock()
    neo.run = AsyncMock(return_value=[])
    tool = UpdateIssueTool(
        api=api, neo4j_client=neo, embedding_client=emb, **CTX
    )

    await tool._arun(issue_number=5, title="new title")

    # The NEW title is what gets embedded and stored.
    assert emb.calls == ["new title"]
    assert captured["embedding"] == [0.1, 0.1, 0.1]


# --- item 4: FindSimilarIssuesTool uses the hybrid path ---------------------


@pytest.mark.asyncio
async def test_find_similar_uses_hybrid_with_embedding(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_hybrid(client: Any, project_id: str, title: str, embedding: Any, k: int = 5) -> list[dict[str, Any]]:
        calls["embedding"] = embedding
        calls["title"] = title
        return [{"number": 3, "title": "can't sign in"}]

    async def fake_legacy(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls["legacy_called"] = True
        return []

    monkeypatch.setattr("tools.issue_tools.hybrid_similar_issues", fake_hybrid)
    monkeypatch.setattr("tools.issue_tools.find_similar_issues", fake_legacy)

    emb = _FakeEmbeddingClient()
    tool = FindSimilarIssuesTool(neo4j_client=MagicMock(), embedding_client=emb, **CTX)

    result = await tool._arun(title="login fails")

    assert "#3" in result
    assert calls["embedding"] == [0.1, 0.1, 0.1]   # hybrid received the vector
    assert "legacy_called" not in calls            # word-overlap path NOT used


@pytest.mark.asyncio
async def test_find_similar_falls_back_to_word_overlap_when_embedding_fails(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_hybrid(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls["hybrid_called"] = True
        return []

    async def fake_legacy(client: Any, project_id: str, title: str) -> list[dict[str, Any]]:
        calls["legacy_called"] = True
        return [{"number": 9, "title": "overlap match"}]

    monkeypatch.setattr("tools.issue_tools.hybrid_similar_issues", fake_hybrid)
    monkeypatch.setattr("tools.issue_tools.find_similar_issues", fake_legacy)

    class _Boom:
        async def embed(self, text: str) -> list[float]:
            raise RuntimeError("provider down")

    tool = FindSimilarIssuesTool(neo4j_client=MagicMock(), embedding_client=_Boom(), **CTX)
    result = await tool._arun(title="login fails")

    assert "#9" in result
    assert "hybrid_called" not in calls   # no embedding → hybrid skipped
    assert calls["legacy_called"] is True
