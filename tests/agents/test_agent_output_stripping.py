"""Regression tests for the two bugs that the UI transcript caught:

1. Reasoning models (MiniMax-M2.7) emit ``<think>…</think>`` blocks before
   their final answer. Without stripping, the scratchpad leaks into the chat.
2. The issue agent's tool list was missing AddIssueToSprintTool, so the
   preview's promise of "put it in sprint Sprint 23" could not be fulfilled.
"""

from unittest.mock import patch

import pytest

from tools.sprint_tools import AddIssueToSprintTool
from tools.issue_tools import (
    CreateIssueTool,
    FindSimilarIssuesTool,
    GetIssuesTool,
    GetStatusesTool,
    SuggestAssigneeTool,
    UpdateIssueStatusTool,
)


CTX = {"org_slug": "acme", "project_id": "proj-1"}


# --- Bug 1: <think> stripping -----------------------------------------------


def test_strip_think_removes_think_block() -> None:
    from agents.issue_agent import _strip_think

    text = "<think>The user wants to create a bug.\nLet me think...\n</think>\nCreated #5: foo"
    assert _strip_think(text) == "Created #5: foo"


def test_strip_think_handles_multiline_think() -> None:
    from agents.issue_agent import _strip_think

    text = (
        "<think>\n- Title: foo\n- Priority: high\n- Assignee: Alice\n</think>\n"
        "I'll create the bug now."
    )
    assert _strip_think(text) == "I'll create the bug now."


def test_strip_think_passthrough_when_absent() -> None:
    from agents.issue_agent import _strip_think

    assert _strip_think("plain answer") == "plain answer"


def test_sprint_agent_strip_think_exists() -> None:
    """Same fix lives in sprint_agent — make sure the helper is there."""
    from agents.sprint_agent import _strip_think

    text = "<think>reasoning</think>Sprint 2 is active."
    assert _strip_think(text) == "Sprint 2 is active."


# --- Bug 2: AddIssueToSprintTool in issue_agent -----------------------------


def test_issue_agent_tool_list_contains_add_to_sprint_tool() -> None:
    """The issue agent's ReAct loop must have AddIssueToSprintTool so it can
    actually fulfill the preview's 'in Sprint X' promise."""
    from agents.issue_agent import _build_agent

    # The build function returns a CompiledStateGraph. The tools are
    # available on its nodes — easiest check is to look at the bound tool
    # objects directly via the agent's expected interface. We verify by
    # importing the tool class and asserting it's in the source list.
    import inspect
    from agents import issue_agent

    src = inspect.getsource(issue_agent._build_agent)
    assert "AddIssueToSprintTool" in src, (
        "AddIssueToSprintTool must be in _build_agent so the model can call it"
    )
    # Also confirm it's wired (instantiation path) — if the wiring is dropped
    # but the import is left, the test above would pass falsely. The literal
    # string must appear inside a tools=[...] list comprehension.
    assert "AddIssueToSprintTool(**api_ctx)" in src


def test_issue_agent_still_has_all_original_tools() -> None:
    """Adding AddIssueToSprintTool must not regress the original 6 tools."""
    from agents import issue_agent
    import inspect

    src = inspect.getsource(issue_agent._build_agent)
    for tool_cls in (
        "GetIssuesTool",
        "GetStatusesTool",
        "CreateIssueTool",
        "UpdateIssueStatusTool",
        "SuggestAssigneeTool",
        "FindSimilarIssuesTool",
        "AddIssueToSprintTool",
    ):
        assert tool_cls in src, f"{tool_cls} missing from issue_agent tools"


def test_add_issue_to_sprint_tool_is_importable_from_issue_agent_module() -> None:
    """The sprint tool must be importable in the issue_agent's namespace."""
    from agents import issue_agent
    # AddIssueToSprintTool is in tools.sprint_tools; issue_agent imports it.
    # Verify the symbol resolves to the same class.
    from tools.sprint_tools import AddIssueToSprintTool as _Sprint

    assert issue_agent.AddIssueToSprintTool is _Sprint


# --- E2E: simulate the failing UI transcript -------------------------------


@pytest.mark.asyncio
async def test_issue_agent_run_strips_think_from_final_message() -> None:
    """Simulates the transcript: the model emits a <think> scratchpad, the
    agent.run() result must NOT contain it."""
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.messages import AIMessage

    from agents import issue_agent

    think_message = AIMessage(
        content=(
            "<think>The user wants to create a bug with high priority. "
            "I should use create_issue. The assignee is Alice but I don't have "
            "her user ID — I'll use the literal name.</think>\n"
            "Created bug #5: \"login crash\" with high priority, assigned to Alice."
        )
    )

    # Patch _build_agent to return a stub whose ainvoke returns the think-laden
    # AIMessage — the same shape the real ReAct graph returns.
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [think_message]})

    with patch.object(issue_agent, "_build_agent", return_value=fake_graph):
        out = await issue_agent.run({
            "messages": [],
            "user_id": "u1",
            "org_slug": "acme",
            "project_id": "proj-1",
            "intent": None,
            "result": None,
            "error": None,
            "next": "",
        })

    msg = out["result"]["message"]
    assert "<think>" not in msg
    assert "</think>" not in msg
    assert msg.startswith("Created bug #5")


@pytest.mark.asyncio
async def test_sprint_agent_run_strips_think_from_final_message() -> None:
    """Same guarantee for the sprint agent."""
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.messages import AIMessage

    from agents import sprint_agent

    think_message = AIMessage(
        content="<think>Let me list the sprints.</think>Sprint 2 is active (May 15 - May 28)."
    )
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [think_message]})

    with patch.object(sprint_agent, "_build_agent", return_value=fake_graph):
        out = await sprint_agent.run({
            "messages": [],
            "user_id": "u1",
            "org_slug": "acme",
            "project_id": "proj-1",
            "intent": None,
            "result": None,
            "error": None,
            "next": "",
        })

    msg = out["result"]["message"]
    assert "<think>" not in msg
    assert "Sprint 2 is active" in msg
