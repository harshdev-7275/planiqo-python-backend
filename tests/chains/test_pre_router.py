"""Tests for the regex/keyword pre-router (item 12).

The cardinal rule: it must NEVER pre-route a write, and must never swallow a
member query as a plain issues list. A miss (None) is always safe — it just
falls through to the LLM."""

from __future__ import annotations

import pytest

from chains.pre_router import pre_route
from models.intents import Intent


# --- confident routes -------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "show me all issues",
    "show open issues",
    "list the bugs",
    "view tickets",
    "display all tasks",
    "show me the backlog",
])
def test_query_issues_routes(msg: str) -> None:
    result = pre_route(msg)
    assert result is not None
    assert result.intent == Intent.QUERY_ISSUES


@pytest.mark.parametrize("msg", [
    "summarize sprint 2",
    "recap the last sprint",
    "how did the last sprint go?",
    "give me a summary",
])
def test_summarize_routes(msg: str) -> None:
    result = pre_route(msg)
    assert result is not None
    assert result.intent == Intent.SUMMARIZE


@pytest.mark.parametrize("msg", ["hi", "hello", "thanks", "thank you", "ok", "cool", ""])
def test_greetings_and_empty_route_to_unknown(msg: str) -> None:
    result = pre_route(msg)
    assert result is not None
    assert result.intent == Intent.UNKNOWN


# --- the critical safety property: writes are NEVER pre-routed ---------------


@pytest.mark.parametrize("msg", [
    "create a bug for login",
    "add a task for the dashboard",
    "open an issue about checkout",
    "file a bug",
    "set issue 5 priority to high",
    "rename issue 17 to deploy",
    "reassign issue 23 to Bob",
    "move issue 8 to done",
    "close issue 4",
    "delete issue 9",
    "make a story for onboarding",
])
def test_writes_are_never_pre_routed(msg: str) -> None:
    assert pre_route(msg) is None


# --- the critical precision property: member queries are not swallowed -------


@pytest.mark.parametrize("msg", [
    "what issues are assigned to Harsh?",
    "show me Bob's issues",
    "list issues assigned to Alice",
    "what is Alice working on?",
])
def test_member_flavoured_queries_defer_to_llm(msg: str) -> None:
    # These could be QUERY_MEMBER; the pre-router must not guess QUERY_ISSUES.
    assert pre_route(msg) is None


# --- ambiguous → defer ------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "what about the other one?",
    "sprint status",
    "how is the team doing",
])
def test_ambiguous_defers_to_llm(msg: str) -> None:
    assert pre_route(msg) is None
