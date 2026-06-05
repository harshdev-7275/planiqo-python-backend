import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from config.settings import settings
from graph import events
from main import app

_AUTH = {"X-Internal-Secret": settings.INTERNAL_SECRET}

_ISSUE = {
    "id":         "issue-1",
    "number":     7,
    "title":      "Fix login bug",
    "type":       "bug",
    "priority":   "high",
    "createdAt":  "2026-06-05T00:00:00Z",
    "projectId":  "proj-1",
    "reporterId": "user-1",
}


# =============================================================================
# /graph/events HTTP endpoint
# =============================================================================


@pytest.mark.asyncio
async def test_events_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/graph/events", json={"entity": "issue", "data": _ISSUE}
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_events_dispatches_issue_upsert():
    with patch("main.apply_event", new=AsyncMock(return_value={"ok": True, "entity": "issue"})) as mock_fn:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/graph/events",
                json={"entity": "issue", "data": _ISSUE},
                headers=_AUTH,
            )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "entity": "issue"}
    mock_fn.assert_called_once()
    _, args, kwargs = mock_fn.mock_calls[0]
    assert args[1] == "issue"
    assert args[2] == _ISSUE


@pytest.mark.asyncio
async def test_events_missing_entity_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/graph/events", json={"data": _ISSUE}, headers=_AUTH
        )
    assert response.status_code == 422


# =============================================================================
# apply_event dispatch logic
# =============================================================================


@pytest.mark.asyncio
async def test_apply_event_routes_to_sync_function():
    neo = AsyncMock()
    with patch("graph.events.sync.upsert_issue", new=AsyncMock()) as mock_upsert:
        result = await events.apply_event(neo, "issue", _ISSUE)
    assert result == {"ok": True, "entity": "issue"}
    mock_upsert.assert_awaited_once_with(neo, _ISSUE)


@pytest.mark.asyncio
async def test_apply_event_member_unpacks_fields():
    neo = AsyncMock()
    data = {"userId": "u-1", "projectId": "p-1", "role": "member"}
    with patch("graph.events.sync.upsert_member", new=AsyncMock()) as mock_upsert:
        result = await events.apply_event(neo, "member", data)
    assert result["ok"] is True
    mock_upsert.assert_awaited_once_with(neo, "u-1", "p-1", "member")


@pytest.mark.asyncio
async def test_apply_event_unknown_entity_is_soft_error():
    neo = AsyncMock()
    result = await events.apply_event(neo, "widget", {"id": "x"})
    assert result == {"ok": False, "reason": "unknown_entity", "entity": "widget"}


@pytest.mark.asyncio
async def test_apply_event_graph_down_degrades_gracefully():
    neo = AsyncMock()
    with patch(
        "graph.events.sync.upsert_issue",
        new=AsyncMock(side_effect=RuntimeError("neo4j unreachable")),
    ):
        result = await events.apply_event(neo, "issue", _ISSUE)
    assert result == {"ok": False, "reason": "graph_unavailable", "entity": "issue"}


@pytest.mark.asyncio
async def test_apply_event_bad_payload_is_soft_error():
    neo = AsyncMock()
    # member handler needs userId/projectId/role — omit them to trigger KeyError
    result = await events.apply_event(neo, "member", {"userId": "u-1"})
    assert result == {"ok": False, "reason": "bad_payload", "entity": "member"}
