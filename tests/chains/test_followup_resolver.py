"""Tests for the context-aware follow-up resolver.

The resolver rewrites a short, context-dependent reply ("yes", "list members")
into a standalone request, using the assistant's prior question as context, so
the normal classify pipeline can handle it. It must NEVER raise — on any
failure it returns the reply unchanged (the status-quo, never a 500).

The LLM is mocked via ``RunnableLambda`` per AIService.md — never at the httpx
level.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.followup_resolver import resolve_followup


def _llm(text: str) -> RunnableLambda:
    return RunnableLambda(lambda _x: AIMessage(content=text))


@pytest.mark.asyncio
async def test_rewrites_short_reply_into_standalone_request() -> None:
    out = await resolve_followup(
        question="Would you like me to look up a specific issue, or list members first?",
        reply="list members",
        llm=_llm("list project members"),
    )
    assert out == "list project members"


@pytest.mark.asyncio
async def test_strips_think_wrappers_and_code_fences() -> None:
    """Reasoning-model <think> wrappers and markdown fences must be stripped."""
    out = await resolve_followup(
        question="...",
        reply="members",
        llm=_llm("<think>they mean members</think>```\nlist project members\n```"),
    )
    assert out == "list project members"


@pytest.mark.asyncio
async def test_empty_reply_passthrough_without_calling_llm() -> None:
    def _boom(_x: object) -> AIMessage:
        raise AssertionError("LLM must not be called for an empty reply")

    out = await resolve_followup(question="...", reply="   ", llm=RunnableLambda(_boom))
    assert out.strip() == ""


@pytest.mark.asyncio
async def test_llm_failure_returns_reply_unchanged() -> None:
    """A provider outage degrades to the status quo: the reply is returned as-is
    and classified normally — never a crash."""
    def _boom(_x: object) -> AIMessage:
        raise RuntimeError("provider down")

    out = await resolve_followup(question="...", reply="yes", llm=RunnableLambda(_boom))
    assert out == "yes"


@pytest.mark.asyncio
async def test_empty_model_output_falls_back_to_reply() -> None:
    out = await resolve_followup(question="...", reply="yes", llm=_llm("   "))
    assert out == "yes"
