import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chains.intent import classify
from models.intents import Intent, IntentResult


def _mock_llm(intent: Intent, confidence: float = 0.95, entities: dict | None = None):
    """A Runnable LLM stub that emits the IntentResult as JSON, matching the real
    chain shape (_PROMPT | llm | _parse_response). Per AIService.md, mock at the
    LangChain interface with RunnableLambda."""
    fixed = IntentResult(intent=intent, confidence=confidence, entities=entities or {})
    return RunnableLambda(lambda _: AIMessage(content=fixed.model_dump_json()))


@pytest.mark.asyncio
async def test_create_issue():
    result = await classify(
        "create a bug for login",
        llm=_mock_llm(Intent.CREATE_ISSUE, entities={"type": "bug", "title": "login"}),
    )
    assert result.intent == Intent.CREATE_ISSUE
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_query_sprint():
    result = await classify(
        "show me sprint status",
        llm=_mock_llm(Intent.QUERY_SPRINT),
    )
    assert result.intent == Intent.QUERY_SPRINT


@pytest.mark.asyncio
async def test_query_member():
    result = await classify(
        "what is john working on",
        llm=_mock_llm(Intent.QUERY_MEMBER, entities={"member": "john"}),
    )
    assert result.intent == Intent.QUERY_MEMBER


@pytest.mark.asyncio
async def test_empty_string_returns_unknown():
    # empty input is short-circuited before any LLM call
    result = await classify("")
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.entities == {}


@pytest.mark.asyncio
async def test_whitespace_only_returns_unknown():
    result = await classify("   ")
    assert result.intent == Intent.UNKNOWN


@pytest.mark.asyncio
async def test_classify_falls_back_when_primary_model_errors():
    """A failing primary model must fall back to the next tier, not crash the chat."""
    def _boom(_: object) -> AIMessage:
        raise RuntimeError("rate limited")

    primary = RunnableLambda(_boom)
    fallback = RunnableLambda(
        lambda _: AIMessage(
            content=IntentResult(
                intent=Intent.QUERY_ISSUES, confidence=0.9, entities={}
            ).model_dump_json()
        )
    )
    result = await classify("show me issues", llm=primary.with_fallbacks([fallback]))
    assert result.intent == Intent.QUERY_ISSUES
