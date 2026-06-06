"""Tests for the insight layer.

The insight layer is grounded — every claim must point at real data.
The LLM is told the closed set of known issue numbers; the post-parse
filter is defence in depth against a misbehaving model.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.insight import Insight, generate_insight
from models.intents import Intent


def _mock_llm(payload: dict) -> RunnableLambda:
    return RunnableLambda(lambda _x: AIMessage(content=json.dumps(payload)))


def _failing_llm(_x: object) -> AIMessage:
    raise RuntimeError("LLM is down")


# --- the deterministic skip-by-intent gate ---------------------------------


@pytest.mark.parametrize(
    "intent",
    [Intent.QUERY_ISSUES, Intent.QUERY_SPRINT, Intent.SUMMARIZE,
     Intent.UNKNOWN, Intent.TEAMS_CONTEXT],
)
async def test_skip_intents_return_none(intent: Intent) -> None:
    """Data queries + UNKNOWN skip the LLM call entirely (no proactive
    insight on a list of 25 issues — the data IS the answer)."""
    # A failing LLM proves the call was never made.
    result = await generate_insight(
        intent=intent,
        user_message="show me issues",
        data="[1,2,3]",
        known_issue_numbers=[1, 2, 3],
        llm=RunnableLambda(_failing_llm),
    )
    assert result is None


# --- happy path: LLM returns a real insight --------------------------------


async def test_duplicate_suspect_routes_through() -> None:
    """The classic PM insight: two issues with the same title."""
    mock = _mock_llm({
        "kind": "duplicate_suspect",
        "message": "Issues #5 and #6 look like duplicates — both titled 'login crash'.",
        "confidence": 0.92,
        "related_issue_numbers": [5, 6],
        "suggested_intent": None,
    })
    result = await generate_insight(
        intent=Intent.UPDATE_ISSUE,  # not in skip set — this is a write path
        user_message="link issues 5 and 6 as duplicates",
        data="...",
        known_issue_numbers=[1, 5, 6, 14, 18],
        llm=mock,
    )
    assert result is not None
    assert result.kind == "duplicate_suspect"


async def test_priority_audit_returns_insight() -> None:
    """Non-skip intent: priority audit comes through."""
    mock = _mock_llm({
        "kind": "priority_audit",
        "message": "You've got 4 critical issues — #14 is the only one blocking login, the rest are billing/admin.",
        "confidence": 0.85,
        "related_issue_numbers": [14, 18, 22, 25],
        "suggested_intent": None,
    })
    result = await generate_insight(
        intent=Intent.UPDATE_ISSUE,  # not in skip set — write path, insight is useful
        user_message="create a critical issue about login",
        data="25 issues, 4 critical, 9 high, 12 medium",
        known_issue_numbers=[14, 18, 22, 25],
        llm=mock,
    )
    assert result is not None
    assert result.kind == "priority_audit"
    assert 14 in result.related_issue_numbers


# --- low confidence → drop the insight --------------------------------------


async def test_low_confidence_returns_none() -> None:
    """confidence < settings.INSIGHT_MIN_CONFIDENCE (default 0.6) drops it."""
    mock = _mock_llm({
        "kind": "pattern",
        "message": "There might be a pattern here.",
        "confidence": 0.3,
        "related_issue_numbers": [],
        "suggested_intent": None,
    })
    result = await generate_insight(
        intent=Intent.QUERY_ISSUES,
        user_message="...",
        data="...",
        known_issue_numbers=[1, 2, 3],
        llm=mock,
    )
    assert result is None


# --- kind='none' from the LLM → no insight ---------------------------------


async def test_kind_none_returns_none() -> None:
    """The LLM can explicitly return 'none' — that's the cheap path."""
    mock = _mock_llm({
        "kind": "none",
        "message": "",
        "confidence": 0.0,
        "related_issue_numbers": [],
        "suggested_intent": None,
    })
    result = await generate_insight(
        intent=Intent.QUERY_ISSUES,
        user_message="...",
        data="...",
        known_issue_numbers=[1, 2, 3],
        llm=mock,
    )
    assert result is None


# --- defence in depth: invented issue numbers are dropped ------------------


async def test_invented_issue_numbers_are_dropped() -> None:
    """A misbehaving LLM that returns numbers not in the known set — those
    are silently dropped. The insight survives with the real numbers."""
    mock = _mock_llm({
        "kind": "duplicate_suspect",
        "message": "These look like duplicates.",
        "confidence": 0.9,
        "related_issue_numbers": [5, 999, 6, 12345],   # 5 and 6 are real, 999 + 12345 are invented
        "suggested_intent": None,
    })
    result = await generate_insight(
        intent=Intent.UPDATE_ISSUE,
        user_message="...",
        data="...",
        known_issue_numbers=[5, 6, 14],
        llm=mock,
    )
    assert result is not None
    assert 5 in result.related_issue_numbers
    assert 6 in result.related_issue_numbers
    assert 999 not in result.related_issue_numbers
    assert 12345 not in result.related_issue_numbers


# --- the LLM itself crashing returns None (no chat crash) ------------------


async def test_llm_exception_returns_none() -> None:
    """If the LLM crashes (rate limit, network), we defer to a flat
    response — the chat never crashes because of a missing insight."""
    result = await generate_insight(
        intent=Intent.QUERY_ISSUES,
        user_message="...",
        data="...",
        known_issue_numbers=[1, 2, 3],
        llm=RunnableLambda(_failing_llm),
    )
    assert result is None


async def test_garbage_llm_output_returns_none() -> None:
    """Garbage JSON from the LLM → empty Insight (kind='none') → None."""
    mock = RunnableLambda(lambda _x: AIMessage(content="not json at all"))
    result = await generate_insight(
        intent=Intent.QUERY_ISSUES,
        user_message="...",
        data="...",
        known_issue_numbers=[1, 2, 3],
        llm=mock,
    )
    assert result is None


# --- the schema is itself a sanity check -----------------------------------


def test_insight_schema_validates_basic_shape() -> None:
    i = Insight(
        kind="priority_audit",
        message="X",
        confidence=0.8,
        related_issue_numbers=[1, 2, 3],
        suggested_intent=Intent.QUERY_ISSUES,
    )
    assert i.kind == "priority_audit"


def test_insight_schema_rejects_invalid_kind() -> None:
    import pytest as _pytest
    with _pytest.raises(ValueError):
        Insight(kind="not-a-real-kind", message="x", confidence=0.5)  # type: ignore[arg-type]


def test_insight_schema_rejects_confidence_above_1() -> None:
    with pytest.raises(ValueError):
        Insight(kind="none", message="x", confidence=1.5)
