"""Unit tests for the LLM block emitter. The LLM is mocked (like every LLM
call in this suite); the focus is the flag gate, intent gate, and the
validate-or-fall-back parsing."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.block_emitter import emit_blocks
from config.settings import settings
from models.intents import Intent


def _llm(payload: str) -> RunnableLambda:
    return RunnableLambda(lambda _x: AIMessage(content=payload))


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RENDER_BLOCKS_LLM_ENABLED", True)


async def test_skips_when_flag_off() -> None:
    # Flag defaults to False — no LLM call, no blocks.
    out = await emit_blocks(
        intent=Intent.QUERY_ISSUES, user_message="x", answer="a", llm=_llm("{}")
    )
    assert out is None


async def test_skips_non_emit_intent(enabled: None) -> None:
    payload = '{"version":"v1","blocks":[{"type":"prose","markdown":"hi"}]}'
    out = await emit_blocks(
        intent=Intent.UNKNOWN, user_message="x", answer="a", llm=_llm(payload)
    )
    assert out is None


async def test_skips_empty_answer(enabled: None) -> None:
    out = await emit_blocks(
        intent=Intent.QUERY_ISSUES, user_message="x", answer="   ", llm=_llm("{}")
    )
    assert out is None


async def test_emits_valid_blocks(enabled: None) -> None:
    payload = (
        '{"version":"v1","blocks":[{"type":"ranking_table",'
        '"columns":[{"key":"n","label":"Name"}],'
        '"rows":[{"cells":["Alice"],"highlight":true}]}]}'
    )
    out = await emit_blocks(
        intent=Intent.QUERY_MEMBER,
        user_message="who has the most issues",
        answer="Alice has the most open issues",
        llm=_llm(payload),
    )
    assert out is not None
    assert out[0].type == "ranking_table"


async def test_emits_from_fenced_json(enabled: None) -> None:
    payload = '```json\n{"blocks":[{"type":"prose","markdown":"hello"}]}\n```'
    out = await emit_blocks(
        intent=Intent.QUERY_ISSUES, user_message="x", answer="hello", llm=_llm(payload)
    )
    assert out is not None
    assert out[0].type == "prose"


async def test_non_json_returns_none(enabled: None) -> None:
    out = await emit_blocks(
        intent=Intent.QUERY_ISSUES, user_message="x", answer="a", llm=_llm("not json at all")
    )
    assert out is None


async def test_all_invalid_blocks_returns_none(enabled: None) -> None:
    out = await emit_blocks(
        intent=Intent.QUERY_ISSUES,
        user_message="x",
        answer="a",
        llm=_llm('{"blocks":[{"type":"definitely_not_a_block"}]}'),
    )
    assert out is None
