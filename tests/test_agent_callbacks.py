"""Tests for ToolLoggingCallbackHandler — per-step agent observability."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from ai_service.agents.callbacks import ToolLoggingCallbackHandler


class TestToolLoggingCallbackHandler:
    def test_logs_tool_start_and_end_with_name_and_latency(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        handler = ToolLoggingCallbackHandler()
        run_id = uuid4()
        with caplog.at_level("INFO", logger="ai_service.agent"):
            asyncio.run(
                handler.on_tool_start(
                    {"name": "graph_suggest_assignee"},
                    "issue_id=abc",
                    run_id=run_id,
                    inputs={"issue_id": "abc"},
                )
            )
            asyncio.run(handler.on_tool_end("[{...}]", run_id=run_id))

        start = next(r for r in caplog.records if r.getMessage() == "tool.start")
        end = next(r for r in caplog.records if r.getMessage() == "tool.end")
        assert start.tool == "graph_suggest_assignee"  # type: ignore[attr-defined]
        assert "issue_id" in start.tool_args  # type: ignore[attr-defined]
        assert end.tool == "graph_suggest_assignee"  # type: ignore[attr-defined]
        assert isinstance(end.duration_ms, float)  # type: ignore[attr-defined]
        assert end.result_chars > 0  # type: ignore[attr-defined]

    def test_logs_tool_error(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = ToolLoggingCallbackHandler()
        run_id = uuid4()
        with caplog.at_level("INFO", logger="ai_service.agent"):
            asyncio.run(handler.on_tool_start({"name": "list_issues"}, "", run_id=run_id))
            asyncio.run(handler.on_tool_error(ValueError("bad limit"), run_id=run_id))

        err = next(r for r in caplog.records if r.getMessage() == "tool.error")
        assert err.tool == "list_issues"  # type: ignore[attr-defined]
        assert err.error == "ValueError"  # type: ignore[attr-defined]
        assert "bad limit" in err.error_message  # type: ignore[attr-defined]

    def test_logs_model_round_trip(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage

        handler = ToolLoggingCallbackHandler()
        run_id = uuid4()
        with caplog.at_level("INFO", logger="ai_service.agent"):
            asyncio.run(
                handler.on_chat_model_start(
                    {},
                    [[SystemMessage(content="p"), HumanMessage(content="hi")]],
                    run_id=run_id,
                )
            )
            asyncio.run(handler.on_llm_end(None, run_id=run_id))  # type: ignore[arg-type]

        start = next(r for r in caplog.records if r.getMessage() == "model.start")
        assert start.message_count == 2  # type: ignore[attr-defined]
        assert any(r.getMessage() == "model.end" for r in caplog.records)

    def test_unknown_run_id_does_not_crash(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = ToolLoggingCallbackHandler()
        with caplog.at_level("INFO", logger="ai_service.agent"):
            # end without a matching start — must degrade gracefully.
            asyncio.run(handler.on_tool_end("x", run_id=uuid4()))
        end = next(r for r in caplog.records if r.getMessage() == "tool.end")
        assert end.tool == "unknown"  # type: ignore[attr-defined]
        assert end.duration_ms is None  # type: ignore[attr-defined]
