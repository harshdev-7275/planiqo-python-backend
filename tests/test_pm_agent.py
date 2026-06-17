"""Tests for the LangGraph agent construction (pm_agent.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall

from ai_service.agents.pm_agent import (
    AgentState,
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
