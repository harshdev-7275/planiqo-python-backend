"""Tests for the POST /v1/chat endpoint.

The LLM is mocked at the graph level — these tests verify the HTTP contract,
dependency injection, error handling, and the audit-trail extraction from
the LangGraph message history.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from ai_service.api.chat import get_agent_deps, router


def _configured_settings_mock(app_env: str = "production") -> MagicMock:
    """Return a settings mock with llm_configured=True and a real model string."""
    settings = MagicMock()
    settings.llm_configured = True
    settings.llm_provider = "groq"
    settings.groq_model = "llama-3.3-70b-versatile"
    settings.minimax_model = "MiniMax-Text-01"
    settings.app_env = app_env
    return settings


def _build_app(node_backend: Any = None) -> FastAPI:
    """Build a minimal FastAPI app for chat tests with a mock client on state."""
    app = FastAPI()
    app.include_router(router)
    app.state.node_backend = node_backend if node_backend is not None else MagicMock()
    return app


def _fake_graph_result(
    output_text: str,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a fake LangGraph result dict.

    Mirrors the AgentState TypedDict.
    """
    if messages is None:
        messages = [AIMessage(content=output_text)]
    return {
        "messages": messages,
        "org_id": "00000000-0000-0000-0000-000000000001",
        "org_slug": "acme",
        "project_id": None,
        "user_id": "00000000-0000-0000-0000-000000000002",
    }


class TestChatEndpoint:
    def test_returns_200_on_success(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("Hello from PM!"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Hello from PM!"
        assert body["model"] == "groq:llama-3.3-70b-versatile"
        assert body["steps"] == 1

    def test_rejects_empty_message(self) -> None:
        app = _build_app()
        with TestClient(app) as client:
            response = client.post("/v1/chat", json={"message": ""})
        assert response.status_code == 422

    def test_rejects_missing_message(self) -> None:
        app = _build_app()
        with TestClient(app) as client:
            response = client.post("/v1/chat", json={})
        assert response.status_code == 422

    def test_rejects_message_over_2000_chars(self) -> None:
        app = _build_app()
        with TestClient(app) as client:
            response = client.post("/v1/chat", json={"message": "x" * 2001})
        assert response.status_code == 422

    def test_503_when_node_backend_not_initialized(self) -> None:
        app = FastAPI()
        app.include_router(router)
        app.state.node_backend = None
        with TestClient(app) as client:
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 503
        assert "NODE_BACKEND" in response.json()["detail"]

    def test_503_when_llm_not_configured(self) -> None:
        app = _build_app()
        bad = MagicMock()
        bad.llm_configured = False
        bad.llm_provider = "groq"
        with (
            patch("ai_service.api.chat.get_settings", return_value=bad),
            TestClient(app) as client,
        ):
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 503
        assert "LLM provider" in response.json()["detail"]

    def test_502_when_agent_raises(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 502

    def test_scopes_to_request_project_id(self) -> None:
        """If the user passes project_id, the agent state must carry it through."""
        app = _build_app()
        captured: dict[str, Any] = {}

        async def fake_ainvoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["state"] = state
            captured["config"] = kwargs.get("config")
            return _fake_graph_result("ok")

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_graph_factory.return_value = mock_graph
            project_id = "11111111-1111-1111-1111-111111111111"
            response = client.post(
                "/v1/chat",
                json={"message": "what's open?", "project_id": project_id},
            )
        assert response.status_code == 200
        assert captured["state"]["project_id"] == project_id
        # A recursion_limit is always passed so the loop is bounded.
        assert captured["config"]["recursion_limit"] > 0

    def test_returns_graceful_message_on_recursion_limit(self) -> None:
        """A GraphRecursionError must yield a friendly 200, not a 502."""
        from langgraph.errors import GraphRecursionError

        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(side_effect=GraphRecursionError("loop"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "who has the most issues?"})
        assert response.status_code == 200
        body = response.json()
        assert body["steps"] == 0
        assert body["tool_calls"] == []
        assert "narrow" in body["message"].lower() or "sync" in body["message"].lower()

    def test_extracts_tool_calls_from_ai_messages(self) -> None:
        """Tool calls in the agent's AI messages must show up in the audit trail."""
        tool_call = ToolCall(name="list_issues", args={"project_id": "abc"}, id="call_1")
        messages = [
            HumanMessage(content="what's open?"),
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="12 open issues"),
        ]
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(
                return_value=_fake_graph_result("12 open issues", messages)
            )
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "what's open?"})
        body = response.json()
        assert body["message"] == "12 open issues"
        assert len(body["tool_calls"]) == 1
        assert body["tool_calls"][0]["tool"] == "list_issues"
        assert body["tool_calls"][0]["args"] == {"project_id": "abc"}

    def test_logs_formatted_chat_block_in_dev_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """In dev, the chat handler logs a single formatted block with query/tools/answer."""
        tool_call = ToolCall(name="list_issues", args={"project_id": "abc"}, id="call_1")
        tool_result = ToolMessage(content='[{"id":"ISS-1","title":"Fix bug"}]', tool_call_id="call_1")
        messages = [
            HumanMessage(content="what's open?"),
            AIMessage(content="", tool_calls=[tool_call]),
            tool_result,
            AIMessage(content="There is 1 open issue: ISS-1."),
        ]
        app = _build_app()
        with (
            caplog.at_level("INFO", logger="ai_service.api.chat"),
            patch(
                "ai_service.api.chat.get_settings",
                return_value=_configured_settings_mock(app_env="development"),
            ),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(
                return_value=_fake_graph_result("There is 1 open issue: ISS-1.", messages)
            )
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "what's open?"})
        assert response.status_code == 200
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "what's open?" in joined
        assert "list_issues" in joined
        assert "project_id" in joined
        assert "ISS-1" in joined  # tool result preview
        assert "There is 1 open issue" in joined
        assert "CHAT REQUEST" in joined
        assert "TOOL CALLS" in joined
        assert "AI RESPONSE" in joined

    def test_does_not_log_formatted_block_in_production(self, caplog: pytest.LogCaptureFixture) -> None:
        """In production, the chat handler must NOT emit the dev console block."""
        app = _build_app()
        with (
            caplog.at_level("INFO", logger="ai_service.api.chat"),
            patch(
                "ai_service.api.chat.get_settings",
                return_value=_configured_settings_mock(app_env="production"),
            ),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("ok"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "CHAT REQUEST" not in joined
        assert "AI RESPONSE" not in joined

    def test_populates_tool_result_preview_from_tool_message(self) -> None:
        """ToolCallRecord.result_preview must be filled from the matching ToolMessage."""
        tool_call = ToolCall(name="list_issues", args={"project_id": "abc"}, id="call_1")
        tool_result = ToolMessage(content="hello world", tool_call_id="call_1")
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[tool_call]),
            tool_result,
            AIMessage(content="done"),
        ]
        app = _build_app()
        with (
            patch(
                "ai_service.api.chat.get_settings",
                return_value=_configured_settings_mock(),
            ),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            patch("ai_service.api.chat.build_all_tools", return_value=[]),
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("done", messages))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        body = response.json()
        assert body["tool_calls"][0]["result_preview"] == "hello world"


class TestGetAgentDeps:
    def test_builds_deps_from_app_state(self) -> None:
        import base64
        import json

        client = MagicMock()
        app = _build_app(node_backend=client)
        # Simulate lifespan-resolved org (app.state has service_org_slug set).
        app.state.service_org_slug = "test-org"
        app.state.service_org_id = "00000000-0000-0000-0000-000000000001"
        app.state.service_user_id = "00000000-0000-0000-0000-000000000002"
        # Build a real-shaped JWT (header.payload.signature) so base64 decode
        # succeeds and the org context in the payload is picked up.
        payload = {
            "orgId": "00000000-0000-0000-0000-000000000099",
            "orgSlug": "from-jwt",
            "userId": "00000000-0000-0000-0000-000000000098",
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"hdr.{encoded}.sig"
        request = MagicMock(app=app)
        request.headers.get.return_value = f"Bearer {token}"
        deps = get_agent_deps(request)
        assert deps.node_backend is client
        assert deps.org_slug == "from-jwt"
        assert deps.org_id is not None
        assert deps.user_id is not None
