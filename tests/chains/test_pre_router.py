"""Tests for the LLM-based pre-router.

The pre-router is a fast classification pass BEFORE the full ``classify`` chain.
It catches obvious read intents ("show me all issues", "summarize sprint 3",
"what is Alice working on") so the heavy ``classify`` call (which also extracts
entities) only runs when the pre-router can't confidently answer.

The cardinal rules:
  * **Never pre-route a WRITE.** Writes need entity extraction that only
    ``classify`` does. A half-extracted write would mis-execute. If the LLM
    returns a WRITE intent, the pre-router drops it and returns ``None``.
  * **Confidence gate.** Only fire on high-confidence classifications.
    A miss just costs one ``classify`` call; a wrong hit routes the user to
    the wrong agent, so we bias hard toward missing.
  * **Empty / greetings short-circuit** without an LLM call.

The LLM is mocked via ``RunnableLambda`` per AIService.md — never mock at
the httpx level.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.pre_router import pre_route
from models.intents import Intent, IntentResult


def _mock_llm_returning(intent: Intent, confidence: float = 0.95):
    """Build a RunnableLambda that returns a valid JSON IntentResult as an
    ``AIMessage`` — matching the real LLM's return shape (per AIService.md,
    the chain is ``_PROMPT | llm | _parse_response`` so the parser expects
    an ``AIMessage`` with ``.content``).
    """
    payload = IntentResult(intent=intent, confidence=confidence, entities={})
    return RunnableLambda(lambda _x: AIMessage(content=payload.model_dump_json()))


def _failing_llm(msg: str):
    """LLM that fails the test if called — used to prove short-circuits."""

    def _boom(_x):
        raise AssertionError(f"LLM should not be called for: {msg!r}")

    return RunnableLambda(_boom)


# --- confident routes -------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "show me all issues",
        "show open issues",
        "list the bugs",
        "view tickets",
        "display all tasks",
        "show me the backlog",
        # the whole point of the LLM version — natural variations
        "what are the open issues?",
        "give me a list of bugs",
        "can you show me the issues",
        "what's on the backlog?",
        "fetch the open bugs",
        "i want to see the open issues",
        "what bugs are open?",
    ],
)
async def test_query_issues_routes(msg: str) -> None:
    """The LLM handles natural variations of 'show me' without hardcoding."""
    mock_llm = _mock_llm_returning(Intent.QUERY_ISSUES, confidence=0.95)
    result = await pre_route(msg, llm=mock_llm)
    assert result is not None
    assert result.intent == Intent.QUERY_ISSUES


@pytest.mark.parametrize(
    "msg",
    [
        "summarize sprint 2",
        "recap the last sprint",
        "how did the last sprint go?",
        "give me a summary",
        "what did we get done this sprint?",
        "sprint recap please",
    ],
)
async def test_summarize_routes(msg: str) -> None:
    mock_llm = _mock_llm_returning(Intent.SUMMARIZE, confidence=0.9)
    result = await pre_route(msg, llm=mock_llm)
    assert result is not None
    assert result.intent == Intent.SUMMARIZE


@pytest.mark.parametrize(
    "msg", ["hi", "hello", "thanks", "thank you", "ok", "cool", ""]
)
async def test_greetings_and_empty_short_circuit_without_llm(msg: str) -> None:
    """Empty / greetings short-circuit WITHOUT calling the LLM.

    The failing_llm raises AssertionError on any call — passing this test
    proves the short-circuit fired.
    """
    result = await pre_route(msg, llm=_failing_llm(msg))
    assert result is not None
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 1.0


# --- the critical safety property: writes are NEVER pre-routed ---------------


@pytest.mark.parametrize(
    "msg",
    [
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
        "create a sprint called Q3 Hardening",
    ],
)
async def test_writes_are_never_pre_routed(msg: str) -> None:
    """Even if the LLM returns a WRITE intent with high confidence, we drop it.

    The pre-router is for read-intent shortcuts only. Entity extraction lives
    in ``classify`` — a write that bypassed it would mis-execute.
    """
    mock_llm = _mock_llm_returning(Intent.CREATE_ISSUE, confidence=0.99)
    result = await pre_route(msg, llm=mock_llm)
    assert result is None


@pytest.mark.parametrize(
    "write_intent",
    [Intent.CREATE_ISSUE, Intent.UPDATE_ISSUE, Intent.CREATE_SPRINT],
)
async def test_every_write_intent_is_blocked(write_intent: Intent) -> None:
    """All three WRITE intents are explicitly blocked — defence in depth."""
    mock_llm = _mock_llm_returning(write_intent, confidence=0.99)
    result = await pre_route("do something", llm=mock_llm)
    assert result is None


# --- the precision property: member queries are routed, not swallowed --------


@pytest.mark.parametrize(
    "msg",
    [
        "what issues are assigned to Harsh?",
        "show me Bob's issues",
        "list issues assigned to Alice",
        "what is Alice working on?",
        "what's Ranu doing?",
        "give me everything Harsh has open",
    ],
)
async def test_member_flavoured_queries_route_to_query_member(msg: str) -> None:
    """The LLM correctly identifies member queries and routes them.

    Under the old regex pre-router these would have been swallowed as a
    plain issues list. The LLM is precise about the 'assigned to X' signal.
    """
    mock_llm = _mock_llm_returning(Intent.QUERY_MEMBER, confidence=0.92)
    result = await pre_route(msg, llm=mock_llm)
    assert result is not None
    assert result.intent == Intent.QUERY_MEMBER


# --- low confidence → defer --------------------------------------------------


async def test_low_confidence_defers_to_full_classify() -> None:
    """If the LLM isn't confident, defer to the full classify chain."""
    mock_llm = _mock_llm_returning(Intent.QUERY_ISSUES, confidence=0.4)
    result = await pre_route("maybe show me issues?", llm=mock_llm)
    assert result is None


async def test_unknown_with_low_confidence_defers() -> None:
    """UNKNOWN with low confidence = the LLM punted. Defer to classify."""
    mock_llm = _mock_llm_returning(Intent.UNKNOWN, confidence=0.2)
    result = await pre_route("what about that thing?", llm=mock_llm)
    assert result is None


# --- LLM errors must not crash the pre-router --------------------------------


async def test_garbage_llm_output_returns_none() -> None:
    """If the LLM returns garbage JSON, we defer to classify (never crash)."""
    garbage_llm = RunnableLambda(lambda _x: AIMessage(content="this is not json at all"))
    result = await pre_route("show me issues", llm=garbage_llm)
    assert result is None


async def test_llm_exception_returns_none() -> None:
    """If the LLM raises, we defer to classify (never crash the chat)."""

    def _boom(_x):
        raise RuntimeError("LLM is down")

    failing_llm = RunnableLambda(_boom)
    result = await pre_route("show me issues", llm=failing_llm)
    assert result is None


# --- the parser is robust to reasoning-model output --------------------------


async def test_think_tags_are_stripped() -> None:
    """Reasoning models (e.g. MiniMax-M1) wrap output in <think>…</think>.

    The pre-router must strip the wrapper before parsing.
    """
    wrapped = (
        '<think>The user wants to see all issues.</think>'
        '{"intent": "QUERY_ISSUES", "confidence": 0.95, "entities": {}}'
    )
    mock_llm = RunnableLambda(lambda _x: AIMessage(content=wrapped))
    result = await pre_route("show me all issues", llm=mock_llm)
    assert result is not None
    assert result.intent == Intent.QUERY_ISSUES


async def test_markdown_code_fences_are_stripped() -> None:
    """A common model failure mode is wrapping JSON in ```json fences."""
    wrapped = (
        '```json\n'
        '{"intent": "QUERY_ISSUES", "confidence": 0.95, "entities": {}}\n'
        '```'
    )
    mock_llm = RunnableLambda(lambda _x: AIMessage(content=wrapped))
    result = await pre_route("show me all issues", llm=mock_llm)
    assert result is not None
    assert result.intent == Intent.QUERY_ISSUES
