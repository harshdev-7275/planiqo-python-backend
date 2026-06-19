"""Tests for the LangGraph agent construction (pm_agent.py)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolCall

from ai_service.agents.pm_agent import (
    AgentState,
    astream_agent,
    build_graph,
    build_llm,
    call_model,
    get_or_build_graph,
    should_continue,
)
from ai_service.config import Settings


def _groq_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "llm_provider": "groq",
        "groq_api_key": "test-groq-key",
        "groq_model": "llama-3.3-70b-versatile",
        "minimax_api_key": "",
        "minimax_base_url": "https://api.minimaxi.com/v1",
        "minimax_model": "MiniMax-Text-01",
        "llm_temperature": 0.2,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _minimax_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "llm_provider": "minimax",
        "groq_api_key": "",
        "minimax_api_key": "test-minimax-key",
        "minimax_base_url": "https://api.minimaxi.com/v1",
        "minimax_model": "MiniMax-Text-01",
        "llm_temperature": 0.2,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestBuildLlm:
    def test_builds_groq_llm_when_provider_is_groq(self) -> None:
        # Patch the source module — pm_agent does lazy imports inside build_llm.
        with patch("langchain_groq.ChatGroq") as mock_groq:
            build_llm(_groq_settings())
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args.kwargs
            assert call_kwargs["model_name"] == "llama-3.3-70b-versatile"
            assert call_kwargs["api_key"] == "test-groq-key"
            assert call_kwargs["temperature"] == 0.2

    def test_builds_minimax_llm_via_chatopenai(self) -> None:
        with patch("langchain_openai.ChatOpenAI") as mock_oai:
            build_llm(_minimax_settings())
            mock_oai.assert_called_once()
            call_kwargs = mock_oai.call_args.kwargs
            assert call_kwargs["model"] == "MiniMax-Text-01"
            assert call_kwargs["api_key"] == "test-minimax-key"
            assert call_kwargs["base_url"] == "https://api.minimaxi.com/v1"

    def test_raises_when_groq_key_missing(self) -> None:
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            build_llm(_groq_settings(groq_api_key=""))

    def test_raises_when_minimax_key_missing(self) -> None:
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            build_llm(_minimax_settings(minimax_api_key=""))


class TestShouldContinue:
    def test_routes_to_tools_when_last_ai_has_tool_calls(self) -> None:
        state: AgentState = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="list_issues", args={}, id="t1")],
                ),
            ],
        }
        assert should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self) -> None:
        state: AgentState = {
            "messages": [HumanMessage(content="hi"), AIMessage(content="Done.")],
        }
        assert should_continue(state) == "__end__"


class TestCallModel:
    def _capture_system(self, state: AgentState) -> str:
        captured: dict[str, str] = {}
        mock_llm = MagicMock()

        def _invoke(messages: list[Any]) -> Any:
            captured["system"] = messages[0].content
            return MagicMock(tool_calls=[])

        mock_llm.invoke.side_effect = _invoke
        call_model(state, mock_llm)
        return captured["system"]

    def test_injects_scope_when_project_label_present(self) -> None:
        system = self._capture_system(
            {"messages": [HumanMessage(content="list open issues")], "project_label": "New Project (NP)"}
        )
        assert "New Project (NP)" in system
        assert "scoped" in system.lower()
        # The model is told not to ask for a project id it already has.
        assert "project id" in system.lower()

    def test_no_scope_section_without_label(self) -> None:
        system = self._capture_system({"messages": [HumanMessage(content="hi")]})
        assert "Current scope" not in system


class TestBuildGraph:
    def test_compiles_state_graph(self) -> None:
        """Smoke test: building the graph with mock tools returns a compiled graph."""
        tools: list[Any] = []
        with patch("ai_service.agents.pm_agent.build_llm") as mock_build_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_build_llm.return_value = mock_llm
            graph = build_graph(tools, _groq_settings())
        # The graph should be compiled (has .ainvoke).
        assert graph is not None
        assert hasattr(graph, "ainvoke")


class TestGetOrBuildGraph:
    def test_caches_graph_per_settings_key(self) -> None:
        tools: list[Any] = []
        with patch("ai_service.agents.pm_agent.build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            settings = _groq_settings()
            g1 = get_or_build_graph(tools, settings)
            g2 = get_or_build_graph(tools, settings)
        assert g1 is g2
        # Same tools + same settings → only one build.
        assert mock_build.call_count == 1


# ---------------------------------------------------------------------------
# Streaming — astream_agent
# ---------------------------------------------------------------------------

_MODEL = "groq:llama-3.3-70b-versatile"


def _event(kind: str, **payload: Any) -> dict[str, Any]:
    """Build a LangGraph astream_events event dict (version='v2' shape)."""
    return {"event": kind, "name": "agent", **payload}


class TestAstreamAgent:
    async def test_yields_token_events_for_model_deltas(self) -> None:
        """on_chat_model_stream chunks become {type: 'token', delta: ...} events."""
        graph = MagicMock()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield _event(
                "on_chat_model_stream",
                data={"chunk": AIMessageChunk(content="Hello")},
            )
            yield _event(
                "on_chat_model_stream",
                data={"chunk": AIMessageChunk(content=" world")},
            )

        graph.astream_events = fake_astream_events

        events: list[dict[str, Any]] = []
        async for ev in astream_agent(
            graph,
            "hi",
            org_id="00000000-0000-0000-0000-000000000001",
            org_slug="acme",
            user_id="00000000-0000-0000-0000-000000000002",
            model=_MODEL,
        ):
            events.append(ev)

        tokens = [e for e in events if e.get("type") == "token"]
        assert [e["delta"] for e in tokens] == ["Hello", " world"]

    async def test_yields_tool_start_and_end_events_with_matching_ids(self) -> None:
        """on_tool_start → tool_start, on_tool_end → tool_end, same tool_call_id."""
        graph = MagicMock()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield _event(
                "on_tool_start",
                name="list_issues",
                run_id="t1",
                data={"input": {"project_id": "p1"}},
            )
            yield _event(
                "on_tool_end",
                name="list_issues",
                run_id="t1",
                data={"output": "result text"},
            )

        graph.astream_events = fake_astream_events

        events: list[dict[str, Any]] = []
        async for ev in astream_agent(
            graph,
            "hi",
            org_id="00000000-0000-0000-0000-000000000001",
            org_slug="acme",
            user_id="00000000-0000-0000-0000-000000000002",
            model=_MODEL,
        ):
            events.append(ev)

        starts = [e for e in events if e.get("type") == "tool_start"]
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(starts) == 1
        assert starts[0]["tool"] == "list_issues"
        assert starts[0]["args"] == {"project_id": "p1"}
        assert len(ends) == 1
        assert ends[0]["result_preview"] == "result text"
        # Caller uses tool_call_id to pair start/end UI chips.
        assert starts[0]["tool_call_id"] == ends[0]["tool_call_id"]

    async def test_yields_done_event_with_accumulated_message(self) -> None:
        """The 'done' event carries the concatenated streamed content + the model."""
        graph = MagicMock()

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield _event(
                "on_chat_model_stream",
                data={"chunk": AIMessageChunk(content="Hello")},
            )
            yield _event(
                "on_chat_model_stream",
                data={"chunk": AIMessageChunk(content=" world")},
            )

        graph.astream_events = fake_astream_events

        events: list[dict[str, Any]] = []
        async for ev in astream_agent(
            graph,
            "hi",
            org_id="00000000-0000-0000-0000-000000000001",
            org_slug="acme",
            user_id="00000000-0000-0000-0000-000000000002",
            model=_MODEL,
        ):
            events.append(ev)

        done = [e for e in events if e.get("type") == "done"]
        assert len(done) == 1
        assert done[0]["message"] == "Hello world"
        assert done[0]["model"] == _MODEL
        assert "steps" in done[0]

    async def test_passes_recursion_limit_to_graph(self) -> None:
        """Bounded recursion — same limit as the non-streaming run_agent."""
        graph = MagicMock()
        captured: dict[str, Any] = {}

        async def fake_astream_events(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            captured.update(kwargs.get("config", {}))
            if False:  # pragma: no cover — keep this an async generator
                yield {}

        graph.astream_events = fake_astream_events

        async for _ in astream_agent(
            graph,
            "hi",
            org_id="00000000-0000-0000-0000-000000000001",
            org_slug="acme",
            user_id="00000000-0000-0000-0000-000000000002",
            model=_MODEL,
        ):
            pass

        assert captured.get("recursion_limit") == 40
