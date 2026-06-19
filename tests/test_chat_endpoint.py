"""Tests for the POST /v1/chat endpoint.

The LLM is mocked at the graph level — these tests verify the HTTP contract,
dependency injection, error handling, and the audit-trail extraction from
the LangGraph message history.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolCall, ToolMessage

from ai_service.api.chat import (
    StreamingReasoningFilter,
    _strip_reasoning,
    get_agent_deps,
    router,
)


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


class TestStripReasoning:
    def test_removes_complete_think_block(self) -> None:
        text = "<think>let me check the issues</think>You have 28 open issues."
        assert _strip_reasoning(text) == "You have 28 open issues."

    def test_removes_multiple_blocks_and_is_case_insensitive(self) -> None:
        text = "<THINK>a</THINK>Answer one. <think>b</think>Answer two."
        assert _strip_reasoning(text) == "Answer one. Answer two."

    def test_removes_trailing_unclosed_block(self) -> None:
        text = "Here is the answer.\n<think>reasoning that got cut off"
        assert _strip_reasoning(text) == "Here is the answer."

    def test_passes_through_text_without_tags(self) -> None:
        assert _strip_reasoning("Just a normal answer.") == "Just a normal answer."

    def test_reasoning_only_falls_back_to_untagged_text(self) -> None:
        # Degenerate: the model produced only reasoning, no answer.
        assert _strip_reasoning("<think>only reasoning</think>") == "only reasoning"


class TestAgentSelection:
    def test_defaults_to_pm_agent_and_passes_spec_to_graph(self) -> None:
        """No `agent` field → PM spec is resolved and handed to the graph factory."""
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("ok"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        spec = mock_graph_factory.call_args.kwargs["spec"]
        assert spec.name == "pm"

    def test_explicit_pm_agent_works(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("ok"))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi", "agent": "pm"})
        assert response.status_code == 200

    def test_unknown_agent_returns_400(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            TestClient(app) as client,
        ):
            response = client.post("/v1/chat", json={"message": "hi", "agent": "nope"})
        assert response.status_code == 400
        assert "Unknown agent" in response.json()["detail"]


class TestChatEndpoint:
    def test_returns_200_on_success(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
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

    def test_strips_think_tags_from_response_message(self) -> None:
        """A model answer containing <think> reasoning must be cleaned before return."""
        app = _build_app()
        raw = "<think>I should list issues, the scope is NP</think>You have 28 open issues."
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result(raw))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "list open issues"})
        assert response.status_code == 200
        assert response.json()["message"] == "You have 28 open issues."

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

    def test_injects_resolved_project_label_into_state(self) -> None:
        """When scoped, the endpoint resolves the project name and seeds it in state."""
        nb = MagicMock()
        nb.get_project = AsyncMock(return_value={"name": "New Project", "key": "NP"})
        app = _build_app(node_backend=nb)
        captured: dict[str, Any] = {}

        async def fake_ainvoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["state"] = state
            return _fake_graph_result("ok")

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_graph_factory.return_value = mock_graph
            response = client.post(
                "/v1/chat",
                json={
                    "message": "list open issues",
                    "project_id": "7b6f4200-249a-446c-a039-6d711d4a02e4",
                },
            )
        assert response.status_code == 200
        nb.get_project.assert_awaited_once()
        assert captured["state"]["project_label"] == "New Project (NP)"

    def test_no_project_label_when_unscoped(self) -> None:
        app = _build_app()
        captured: dict[str, Any] = {}

        async def fake_ainvoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["state"] = state
            return _fake_graph_result("ok")

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert captured["state"]["project_label"] is None

    def test_replays_history_into_agent_state(self) -> None:
        """Prior turns sent by the client must seed the agent's message list."""
        app = _build_app()
        captured: dict[str, Any] = {}

        async def fake_ainvoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["state"] = state
            return _fake_graph_result("ok")

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_graph_factory.return_value = mock_graph
            response = client.post(
                "/v1/chat",
                json={
                    "message": "From TP",
                    "history": [
                        {"role": "user", "content": "give me all issues"},
                        {"role": "assistant", "content": "Which project? TP or NP"},
                    ],
                },
            )
        assert response.status_code == 200
        msgs = captured["state"]["messages"]
        # 2 history turns + the new user message, in order.
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "give me all issues"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Which project? TP or NP"
        assert isinstance(msgs[2], HumanMessage)
        assert msgs[2].content == "From TP"

    def test_rejects_invalid_history_role(self) -> None:
        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/chat",
                json={"message": "hi", "history": [{"role": "system", "content": "x"}]},
            )
        assert response.status_code == 422

    def test_attaches_tool_logging_callback_to_run(self) -> None:
        """The chat run must carry a callback handler so tool calls are logged."""
        from ai_service.agents.callbacks import ToolLoggingCallbackHandler

        app = _build_app()
        captured: dict[str, Any] = {}

        async def fake_ainvoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["config"] = kwargs.get("config")
            return _fake_graph_result("ok")

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        callbacks = captured["config"]["callbacks"]
        assert any(isinstance(cb, ToolLoggingCallbackHandler) for cb in callbacks)

    def test_returns_graceful_message_on_recursion_limit(self) -> None:
        """A GraphRecursionError must yield a friendly 200, not a 502."""
        from langgraph.errors import GraphRecursionError

        app = _build_app()
        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
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

    def test_logs_structured_chat_completed_event(self, caplog: pytest.LogCaptureFixture) -> None:
        """One structured `chat.completed` line is emitted with operational metadata."""
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
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(
                return_value=_fake_graph_result("There is 1 open issue: ISS-1.", messages)
            )
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "what's open?"})
        assert response.status_code == 200
        records = [r for r in caplog.records if r.getMessage() == "chat.completed"]
        assert len(records) == 1
        rec = records[0]
        assert rec.agent == "pm"  # type: ignore[attr-defined]
        assert rec.tool_call_count == 1  # type: ignore[attr-defined]
        assert rec.tools_used == ["list_issues"]  # type: ignore[attr-defined]
        assert rec.scoped is False  # type: ignore[attr-defined]

    def test_does_not_log_user_message_or_answer_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """User message text and the answer must never appear in logs (PII)."""
        app = _build_app()
        with (
            caplog.at_level("INFO", logger="ai_service.api.chat"),
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph") as mock_graph_factory,
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(
                return_value=_fake_graph_result("the secret answer")
            )
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "my secret question"})
        assert response.status_code == 200
        joined = "\n".join(
            record.getMessage() + str(record.__dict__) for record in caplog.records
        )
        assert "my secret question" not in joined
        assert "the secret answer" not in joined

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
            TestClient(app) as client,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=_fake_graph_result("done", messages))
            mock_graph_factory.return_value = mock_graph
            response = client.post("/v1/chat", json={"message": "hi"})
        body = response.json()
        assert body["tool_calls"][0]["result_preview"] == "hello world"


def _request_with_headers(app: FastAPI, headers: dict[str, str]) -> Any:
    """A MagicMock request whose .headers.get reads from a real dict (lowercased)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    request = MagicMock(app=app)
    request.headers.get.side_effect = lambda key, default="": lowered.get(key.lower(), default)
    return request


class TestGetAgentDeps:
    def test_builds_deps_from_trusted_identity_headers(self) -> None:
        client = MagicMock()
        app = _build_app(node_backend=client)
        app.state.service_token = ""  # token auth disabled
        request = _request_with_headers(
            app,
            {
                "X-Org-Id": "00000000-0000-0000-0000-000000000099",
                "X-Org-Slug": "from-headers",
                "X-User-Id": "00000000-0000-0000-0000-000000000098",
            },
        )
        deps = get_agent_deps(request)
        assert deps.node_backend is client
        assert deps.org_slug == "from-headers"
        assert str(deps.org_id) == "00000000-0000-0000-0000-000000000099"
        assert str(deps.user_id) == "00000000-0000-0000-0000-000000000098"

    def test_accepts_matching_service_token(self) -> None:
        app = _build_app()
        app.state.service_token = "s" * 32
        request = _request_with_headers(
            app,
            {
                "X-Service-Token": "s" * 32,
                "X-Org-Id": "00000000-0000-0000-0000-000000000099",
                "X-Org-Slug": "acme",
                "X-User-Id": "00000000-0000-0000-0000-000000000098",
            },
        )
        deps = get_agent_deps(request)
        assert deps.org_slug == "acme"

    def test_rejects_missing_service_token(self) -> None:
        from fastapi import HTTPException

        app = _build_app()
        app.state.service_token = "s" * 32
        request = _request_with_headers(app, {"X-Org-Slug": "acme"})
        with pytest.raises(HTTPException) as exc:
            get_agent_deps(request)
        assert exc.value.status_code == 401

    def test_rejects_wrong_service_token(self) -> None:
        from fastapi import HTTPException

        app = _build_app()
        app.state.service_token = "s" * 32
        request = _request_with_headers(app, {"X-Service-Token": "wrong-token"})
        with pytest.raises(HTTPException) as exc:
            get_agent_deps(request)
        assert exc.value.status_code == 401

    def test_503_when_node_backend_missing(self) -> None:
        from fastapi import HTTPException

        app = FastAPI()
        app.state.node_backend = None
        request = _request_with_headers(app, {})
        with pytest.raises(HTTPException) as exc:
            get_agent_deps(request)
        assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Streaming reasoning filter
# ---------------------------------------------------------------------------


class TestStreamingReasoningFilter:
    def test_strips_complete_think_block_in_single_chunk(self) -> None:
        f = StreamingReasoningFilter()
        assert f.feed("<think>reasoning</think>hello") == "hello"

    def test_strips_think_block_split_across_multiple_feeds(self) -> None:
        f = StreamingReasoningFilter()
        assert f.feed("<think>rea") == ""
        assert f.feed("soning</think>") == ""
        assert f.feed("hello") == "hello"

    def test_strips_multiple_think_blocks(self) -> None:
        f = StreamingReasoningFilter()
        assert f.feed("<think>a</think>one. <think>b</think>two.") == "one. two."

    def test_holds_back_text_that_could_become_partial_open_tag(self) -> None:
        # A trailing "<" could be the start of "<think>" — hold it back so it
        # never leaks to the user. The next feed resolves it.
        f = StreamingReasoningFilter()
        assert f.feed("Hello ") == "Hello "
        assert f.feed("<") == ""  # held back, could be "<think>" coming
        # The "<" resolves into a think tag, the safe text is just "bar"
        # (the "Hello " was already emitted in feed 1).
        assert f.feed("think>foo</think>bar") == "bar"

    def test_emits_trailing_text_when_no_think_block(self) -> None:
        f = StreamingReasoningFilter()
        assert f.feed("just text") == "just text"

    def test_flush_emits_remaining_text_after_unclosed_think(self) -> None:
        # Degenerate: the model never closed the think block. Strip the leading
        # <think> tag (if present) so the user sees the intended answer, not
        # raw reasoning — same fallback as _strip_reasoning.
        f = StreamingReasoningFilter()
        f.feed("<think>only reasoning")
        assert f.flush() == "only reasoning"

    def test_flush_emits_text_when_no_pending(self) -> None:
        f = StreamingReasoningFilter()
        assert f.flush() == ""

    def test_case_insensitive_tag_match(self) -> None:
        f = StreamingReasoningFilter()
        assert f.feed("<THINK>foo</THINK>answer") == "answer"


# ---------------------------------------------------------------------------
# /v1/chat/stream — SSE endpoint
# ---------------------------------------------------------------------------


def _parse_sse_frames(body: str) -> list[dict[str, Any]]:
    """Parse a `data: {...}\\n\\n` SSE body into a list of event dicts."""
    frames: list[dict[str, Any]] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data:"):
            payload = chunk[len("data:"):].strip()
            frames.append(json.loads(payload))
    return frames


class TestChatStreamEndpoint:
    def test_streams_sse_frames_and_strips_think_tags_mid_stream(self) -> None:
        """Token events become SSE frames; <think> blocks are stripped before emit."""
        app = _build_app()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            # One model invocation → on_chat_model_start bumps `steps` to 1,
            # then the streamed chunks are the two <think>-...-answer pieces.
            yield {
                "event": "on_chat_model_start",
                "name": "agent",
                "data": {},
            }
            yield {
                "event": "on_chat_model_stream",
                "name": "agent",
                "data": {"chunk": AIMessageChunk(content="<think>reasoning")},
            }
            yield {
                "event": "on_chat_model_stream",
                "name": "agent",
                "data": {"chunk": AIMessageChunk(content="</think>You have 28 open issues.")},
            }

        mock_graph = MagicMock()
        mock_graph.astream_events = fake_astream_events

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph", return_value=mock_graph),
            TestClient(app) as client,
        ):
            response = client.post("/v1/chat/stream", json={"message": "list open issues"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers.get("x-accel-buffering") == "no"

        frames = _parse_sse_frames(response.text)
        token_deltas = [f["delta"] for f in frames if f.get("type") == "token"]
        done = next((f for f in frames if f.get("type") == "done"), None)

        # Reasoning must never reach the user.
        full = "".join(token_deltas)
        assert "reasoning" not in full
        assert "<think>" not in full
        assert "You have 28 open issues" in full

        # Final frame carries the cleaned, accumulated message + model.
        assert done is not None
        assert done["message"] == "You have 28 open issues."
        assert done["model"] == "groq:llama-3.3-70b-versatile"
        assert done["steps"] == 1

    def test_emits_tool_start_and_tool_end_frames(self) -> None:
        """A tool call produces paired start/end SSE frames with matching tool_call_id."""
        app = _build_app()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield {
                "event": "on_tool_start",
                "name": "list_issues",
                "run_id": "t1",
                "data": {"input": {"project_id": "p1"}},
            }
            yield {
                "event": "on_tool_end",
                "name": "list_issues",
                "run_id": "t1",
                "data": {"output": "result text"},
            }
            yield {
                "event": "on_chat_model_stream",
                "name": "agent",
                "data": {"chunk": AIMessageChunk(content="done.")},
            }

        mock_graph = MagicMock()
        mock_graph.astream_events = fake_astream_events

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph", return_value=mock_graph),
            TestClient(app) as client,
        ):
            response = client.post("/v1/chat/stream", json={"message": "hi"})

        frames = _parse_sse_frames(response.text)
        starts = [f for f in frames if f.get("type") == "tool_start"]
        ends = [f for f in frames if f.get("type") == "tool_end"]
        assert len(starts) == 1
        assert starts[0]["tool"] == "list_issues"
        assert starts[0]["args"] == {"project_id": "p1"}
        assert len(ends) == 1
        assert ends[0]["result_preview"] == "result text"
        assert starts[0]["tool_call_id"] == ends[0]["tool_call_id"]

    def test_emits_error_frame_on_graph_recursion_error(self) -> None:
        """A GraphRecursionError surfaces as an `error` SSE frame, not a 502."""
        from langgraph.errors import GraphRecursionError

        app = _build_app()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            raise GraphRecursionError("loop")
            yield  # pragma: no cover

        mock_graph = MagicMock()
        mock_graph.astream_events = fake_astream_events

        with (
            patch("ai_service.api.chat.get_settings", return_value=_configured_settings_mock()),
            patch("ai_service.api.chat.get_or_build_graph", return_value=mock_graph),
            TestClient(app) as client,
        ):
            response = client.post("/v1/chat/stream", json={"message": "hi"})

        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        errors = [f for f in frames if f.get("type") == "error"]
        assert len(errors) == 1
        assert errors[0]["code"] == "RECURSION_LIMIT"
