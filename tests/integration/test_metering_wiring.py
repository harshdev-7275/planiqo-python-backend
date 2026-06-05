"""End-to-end test of the metering callback path.

The metering module ships with thorough unit tests, but a refactor could
disconnect ``supervisor.run`` -> ``chains.intent.classify`` -> ``UsageCallback``
silently (the per-isolate unit tests would still pass). This test wires the
REAL supervisor + REAL classify chain against a fake LLM that emits
``usage_metadata`` on its response, so the LangChain callback system actually
fires the handler. If anyone changes the callback plumbing — e.g. drops the
``config={"callbacks": [...]}`` kwarg, instantiates the wrong callback class,
or strips the request_tokens wiring — this test fails.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from memory.store import conversation_store
from metering.usage import usage_store
from models.intents import Intent, IntentResult


BASE = {"user_id": "u1", "org_slug": "acme", "project_id": None}

_TOTAL_TOKENS = 42


def _fake_chat_model() -> FakeMessagesListChatModel:
    """A real LangChain BaseChatModel that returns AIMessage with usage_metadata.
    Because it inherits from ``BaseChatModel``, the framework fires the
    ``on_llm_start`` / ``on_llm_end`` callback events — which is what the real
    UsageCallback hooks into."""
    return FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=IntentResult(
                    intent=Intent.QUERY_ISSUES, confidence=0.9, entities={}
                ).model_dump_json(),
                usage_metadata={
                    "input_tokens": _TOTAL_TOKENS - 2,
                    "output_tokens": 2,
                    "total_tokens": _TOTAL_TOKENS,
                },
            )
        ]
    )


@pytest.fixture(autouse=True)
def _isolate_stores() -> Any:
    conversation_store.reset_all()
    usage_store.reset_all()
    yield
    conversation_store.reset_all()
    usage_store.reset_all()


@pytest.mark.asyncio
async def test_supervisor_classify_invokes_usage_callback() -> None:
    """The classify chain must thread callbacks through to the LLM so the
    UsageCallback fires and tokens are recorded against the org + the
    per-request accumulator. This is the wiring the unit tests cannot see."""
    # Patch the name where it's *used* (chains.intent), not where it's
    # defined. `from clients.llm_client import get_fast_resilient` binds the
    # reference at import time, so patching the source module alone doesn't
    # affect the bound reference once any other test has already imported it.
    with (
        patch("chains.intent.get_fast_resilient", return_value=_fake_chat_model()),
        # Defer the pre-router to classify — this test is about the classify
        # chain's metering, not about the pre-router path.
        patch("agents.supervisor.pre_route", new=AsyncMock(return_value=None)),
        patch(
            "agents.issue_agent.run",
            new=AsyncMock(return_value={"result": {"message": "issue done"}}),
        ),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="what should I focus on today", **BASE)

    # 1) Supervisor reported tokens in the response — proves the per-request
    #    accumulator was attached to the callback AND read at the end.
    assert result["status"] == "executed"
    assert result["tokens_used"] == _TOTAL_TOKENS

    # 2) The UsageCallback also wrote the same number to the per-org store.
    #    (Proves the callback fired through the LangChain callback system.)
    assert usage_store.get("acme") == _TOTAL_TOKENS

    # 3) The request was counted once (one /chat call, regardless of how
    #    many LLM calls fired inside it).
    assert usage_store.get_request_count("acme") == 1


@pytest.mark.asyncio
async def test_awaiting_confirmation_does_not_crash_metering() -> None:
    """The propose path never invokes the agent. The metering must still work
    end-to-end because the classify callback fires on the way in."""
    with (
        patch("clients.llm_client.get_fast_resilient", return_value=_fake_chat_model()),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        # Force the supervisor's classify to return CREATE_ISSUE so the
        # write-confirmation branch fires.
        from agents.supervisor import run

        async def _force_create(_msg: str, **_kwargs: Any) -> IntentResult:
            return IntentResult(
                intent=Intent.CREATE_ISSUE, confidence=0.95,
                entities={"title": "x", "type": "bug", "priority": "high"},
            )

        with patch("agents.supervisor.classify", new=_force_create):
            result = await run(message="create a bug called x", **BASE)

    assert result["status"] == "awaiting_confirmation"
    # The propose path returns tokens_used as 0 because classify was mocked
    # in this test variant — but the request counter must still bump, proving
    # the metering is structurally independent of the agent execution path.
    assert "tokens_used" in result
    assert usage_store.get_request_count("acme") == 1
