"""Tests for the shared ReAct read-agent helpers — in particular the
false-negative guard (item 7): an upstream fetch failure must never be reported
to the user as "found none"."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.react_utils import finalize, strip_think, tool_error_in_trace


def _tool_msg(content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="t1")


# --- strip_think ------------------------------------------------------------


def test_strip_think_removes_block() -> None:
    assert strip_think("<think>reasoning</think>answer") == "answer"


# --- tool_error_in_trace ----------------------------------------------------


def test_detects_failed_tool_result() -> None:
    trace = [HumanMessage(content="hi"), _tool_msg("Failed to get issues: timeout")]
    assert tool_error_in_trace(trace) is True


def test_plain_empty_result_is_not_an_error() -> None:
    """'No issues found.' is a real empty result — must NOT count as a failure."""
    trace = [HumanMessage(content="hi"), _tool_msg("No issues found.")]
    assert tool_error_in_trace(trace) is False


def test_ai_message_is_ignored() -> None:
    trace = [AIMessage(content="Failed because I felt like it")]  # not a tool msg
    assert tool_error_in_trace(trace) is False


# --- finalize ---------------------------------------------------------------


def test_finalize_appends_caveat_when_tool_failed() -> None:
    trace = [
        _tool_msg("Failed to get issues: connection refused"),
        AIMessage(content="There are no issues."),
    ]
    out = finalize(trace)
    assert "no issues" in out.lower()
    assert "couldn't reach the server" in out.lower()


def test_finalize_no_caveat_on_genuine_empty() -> None:
    trace = [
        _tool_msg("No issues found."),
        AIMessage(content="You have no open issues."),
    ]
    out = finalize(trace)
    assert "couldn't reach the server" not in out.lower()


def test_finalize_does_not_double_flag_when_model_already_said_it() -> None:
    trace = [
        _tool_msg("Failed to get issues: 500"),
        AIMessage(content="I couldn't reach the server, please try again."),
    ]
    out = finalize(trace)
    # Caveat text is not appended twice.
    assert out.lower().count("try again") == 1


# --- agent integration ------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_agent_surfaces_fetch_failure_not_empty() -> None:
    """End-to-end: when the get_issues tool fails, the issue agent's answer
    must flag the unreachable server, not parrot a confident 'no issues'."""
    from agents import issue_agent

    trace = [
        _tool_msg("Failed to get issues: timeout"),
        AIMessage(content="There are no issues in this project."),
    ]
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": trace})
    with patch.object(issue_agent, "_build_agent", return_value=fake_graph):
        out = await issue_agent.run({
            "messages": [], "user_id": "u1", "org_slug": "acme",
            "project_id": "proj-1", "intent": None, "result": None,
            "error": None, "next": "",
        })
    assert "couldn't reach the server" in out["result"]["message"].lower()
