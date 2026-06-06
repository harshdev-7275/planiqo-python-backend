"""Tests for the presentation layer.

Two halves:
  1. ``generate_presentation`` — the LLM call. Mocked, like all LLM calls.
  2. ``render_presentation`` — a pure function. Same input → same output.

The split matters: the LLM is creative, the renderer is deterministic.
The renderer is what the user actually sees, and it's covered by
exhaustive tests because it has no I/O.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.insight import Insight
from chains.persona_resolver import Persona
from chains.presentation import (
    Presentation,
    Suggestion,
    generate_presentation,
    render_presentation,
)
from models.intents import Intent


def _mock_llm(payload: dict) -> RunnableLambda:
    return RunnableLambda(lambda _x: AIMessage(content=json.dumps(payload)))


# --- the deterministic skip-by-intent gate ---------------------------------


@pytest.mark.parametrize(
    "intent",
    [Intent.QUERY_ISSUES, Intent.QUERY_SPRINT, Intent.SUMMARIZE,
     Intent.UNKNOWN, Intent.TEAMS_CONTEXT],
)
async def test_skip_intents_return_none(intent: Intent) -> None:
    """Data queries + UNKNOWN skip the LLM call entirely."""
    def _boom(_x: object) -> AIMessage:
        raise RuntimeError("LLM should not be called for skipped intents")
    result = await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=intent,
        user_message="...",
        llm=RunnableLambda(_boom),
    )
    assert result is None


# --- happy path -------------------------------------------------------------


async def test_presentation_with_suggestions() -> None:
    """A full Presentation with narrative + suggestions + data_block."""
    mock = _mock_llm({
        "opener": "Heads up —",
        "narrative": "Ranu has 6 issues; #25 is the most urgent (critical).",
        "data_block": {"summary": "Ranu Singh: 6 open issues, 1 critical."},
        "suggestions": [
            {"label": "Reassign #25 to someone with capacity",
             "intent_preview": "UPDATE_ISSUE",
             "requires_confirmation": True,
             "confidence": 0.9},
        ],
        "closer": "Want me to draft an update for her?",
        "tone_markers": ["concise", "warm"],
    })
    result = await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=Intent.QUERY_MEMBER,
        user_message="what is Ranu working on?",
        data="Ranu has 6 issues...",
        llm=mock,
    )
    assert result is not None
    assert result.opener == "Heads up —"
    assert "Ranu" in result.narrative
    assert len(result.suggestions) == 1
    assert result.suggestions[0].requires_confirmation is True


async def test_insight_is_passed_through_to_prompt() -> None:
    """The insight is woven into the prompt so the LLM can use it
    naturally (not pasted verbatim). The LLM receives a ChatPromptValue;
    stringify it to inspect what the model saw."""
    captured: list = []

    def _capture(prompt_value: object) -> AIMessage:
        captured.append(prompt_value)
        return AIMessage(content=json.dumps({
            "opener": "",
            "narrative": "You've got 4 critical issues.",
            "data_block": None,
            "suggestions": [],
            "closer": "",
            "tone_markers": [],
        }))

    insight = Insight(
        kind="priority_audit",
        message="4 critical issues, #14 is the only login blocker",
        confidence=0.85,
        related_issue_numbers=[14, 18, 22, 25],
    )
    await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=Intent.UPDATE_ISSUE,
        user_message="...",
        data="...",
        insight=insight,
        llm=RunnableLambda(_capture),
    )
    assert len(captured) == 1
    prompt_repr = repr(captured[0])  # ChatPromptValue's repr shows message contents
    # The insight text should be in the prompt the LLM saw.
    assert "4 critical issues" in prompt_repr
    assert "priority_audit" in prompt_repr


async def test_data_block_passed_in_is_preserved_if_llm_omits_it() -> None:
    """If the caller passes a data_block and the LLM doesn't echo one,
    the caller's data_block is carried through."""
    mock = _mock_llm({
        "opener": "",
        "narrative": "Done.",
        "data_block": None,   # LLM didn't include one
        "suggestions": [],
        "closer": "",
        "tone_markers": [],
    })
    block = {"summary": "3 critical issues"}
    result = await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=Intent.UPDATE_ISSUE,
        user_message="...",
        data_block=block,
        llm=mock,
    )
    assert result is not None
    assert result.data_block == block


async def test_llm_exception_returns_none() -> None:
    """LLM crash → None → caller falls back to template."""
    def _boom(_x: object) -> AIMessage:
        raise RuntimeError("LLM is down")
    result = await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=Intent.UPDATE_ISSUE,
        user_message="...",
        llm=RunnableLambda(_boom),
    )
    assert result is None


async def test_garbage_llm_output_returns_none() -> None:
    mock = RunnableLambda(lambda _x: AIMessage(content="not json"))
    result = await generate_presentation(
        persona=Persona(name="senior_pm", voice_directive="x"),
        intent=Intent.UPDATE_ISSUE,
        user_message="...",
        llm=mock,
    )
    assert result is None


# --- render_presentation is a pure function --------------------------------


def test_render_empty_presentation() -> None:
    out = render_presentation(Presentation(narrative=""))
    assert out == "(no message)"


def test_render_narrative_only() -> None:
    p = Presentation(narrative="Updated #1.")
    out = render_presentation(p)
    assert out == "Updated #1."


def test_render_opener_and_narrative_and_closer() -> None:
    p = Presentation(
        opener="Heads up —",
        narrative="Ranu has 6 issues.",
        closer="Want me to draft an update?",
    )
    out = render_presentation(p)
    assert "Heads up —" in out
    assert "Ranu has 6 issues." in out
    assert "Want me to draft an update?" in out


def test_render_suggestions_as_bullets() -> None:
    p = Presentation(
        narrative="Done.",
        suggestions=[
            Suggestion(label="Reassign #25", requires_confirmation=True),
            Suggestion(label="View all open", requires_confirmation=False),
        ],
    )
    out = render_presentation(p)
    assert "→ Reassign #25" in out
    assert "→ View all open" in out


def test_render_data_block_as_table() -> None:
    p = Presentation(
        narrative="Here you go:",
        data_block={"table": [
            {"#": 1, "Title": "Foo"},
            {"#": 2, "Title": "Bar"},
        ]},
    )
    out = render_presentation(p)
    assert "| # | Title |" in out
    assert "| 1 | Foo |" in out
    assert "| 2 | Bar |" in out


def test_render_data_block_as_summary() -> None:
    p = Presentation(
        narrative="Sprint is on track.",
        data_block={"summary": "8/16 issues done, 2 critical remaining."},
    )
    out = render_presentation(p)
    assert "8/16 issues done" in out


def test_render_with_auditor_persona_strips_leading_punctuation() -> None:
    """The auditor persona's emoji_policy='none' strips leading
    decorative punctuation from the opener."""
    auditor = Persona(
        name="auditor",
        voice_directive="x",
        emoji_policy="none",
    )
    p = Presentation(opener="⚠ Heads up —", narrative="Done.")
    out = render_presentation(p, persona=auditor)
    # The ⚠ is stripped, "Heads up —" remains.
    assert "⚠" not in out.split("\n")[0]
    assert "Heads up" in out
