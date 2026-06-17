"""Tests for the agent abstraction and registry.

These prove the extension point: a new agent is added by registering an
``AgentSpec`` — no change to the graph engine or the chat endpoint.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_service.agents.base import AgentSpec
from ai_service.agents.registry import (
    DEFAULT_AGENT,
    PM_SPEC,
    available_agents,
    get_agent_spec,
    register_agent,
)
from ai_service.config import Settings


def _groq_settings() -> Settings:
    return Settings(
        llm_provider="groq",
        groq_api_key="test-groq-key",
        groq_model="llama-3.3-70b-versatile",
        minimax_api_key="",
        minimax_base_url="https://api.minimaxi.com/v1",
        minimax_model="MiniMax-Text-01",
        llm_temperature=0.2,
    )


class TestRegistry:
    def test_pm_is_the_default_agent(self) -> None:
        assert get_agent_spec() is PM_SPEC
        assert get_agent_spec(None) is PM_SPEC
        assert get_agent_spec(DEFAULT_AGENT) is PM_SPEC

    def test_unknown_agent_raises_keyerror_listing_available(self) -> None:
        with pytest.raises(KeyError, match="Unknown agent"):
            get_agent_spec("does-not-exist")

    def test_register_and_resolve_new_agent(self) -> None:
        def _tools(*_: Any, **__: Any) -> list[Any]:
            return []

        spec = AgentSpec(name="triage-test", prompt="P", select_tools=_tools)
        register_agent(spec)
        try:
            assert get_agent_spec("triage-test") is spec
            assert "triage-test" in available_agents()
        finally:
            # Keep the global registry clean for other tests.
            from ai_service.agents.registry import _REGISTRY

            _REGISTRY.pop("triage-test", None)


class TestSpecDrivesGraph:
    def test_custom_prompt_is_used_in_model_call(self) -> None:
        """A spec's prompt — not the hardcoded PM persona — reaches the model."""
        from ai_service.agents.pm_agent import build_graph, call_model

        captured: dict[str, Any] = {}
        mock_llm = MagicMock()

        def _invoke(messages: list[Any]) -> Any:
            captured["system"] = messages[0].content
            return MagicMock(tool_calls=[])

        mock_llm.invoke.side_effect = _invoke
        call_model({"messages": [MagicMock()]}, mock_llm, "CUSTOM PERSONA")
        assert captured["system"] == "CUSTOM PERSONA"

        # And build_graph threads the prompt through.
        with patch("ai_service.agents.pm_agent.build_llm") as mock_build:
            inner = MagicMock()
            inner.bind_tools.return_value = inner
            mock_build.return_value = inner
            graph = build_graph([], _groq_settings(), prompt="X")
        assert graph is not None

    def test_get_or_build_graph_honors_custom_graph_factory(self) -> None:
        from ai_service.agents.pm_agent import get_or_build_graph

        sentinel = object()
        calls: dict[str, Any] = {}

        def _factory(tools: list[Any], settings: Settings, *, prompt: str) -> Any:
            calls["prompt"] = prompt
            return sentinel

        spec = AgentSpec(
            name="factory-test",
            prompt="FP",
            select_tools=lambda *a, **k: [],
            graph_factory=_factory,
        )
        graph = get_or_build_graph([], _groq_settings(), spec=spec)
        assert graph is sentinel
        assert calls["prompt"] == "FP"

    def test_cache_key_separates_agents(self) -> None:
        """Two agents with the same tools must not share a compiled graph."""
        from ai_service.agents.pm_agent import get_or_build_graph

        spec_a = AgentSpec(
            name="cache-a",
            prompt="A",
            select_tools=lambda *a, **k: [],
            graph_factory=lambda *a, **k: object(),
        )
        spec_b = AgentSpec(
            name="cache-b",
            prompt="B",
            select_tools=lambda *a, **k: [],
            graph_factory=lambda *a, **k: object(),
        )
        tools: list[Any] = []
        g_a = get_or_build_graph(tools, _groq_settings(), spec=spec_a)
        g_b = get_or_build_graph(tools, _groq_settings(), spec=spec_b)
        assert g_a is not g_b
