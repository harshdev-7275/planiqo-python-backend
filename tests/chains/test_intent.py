import pytest
from langchain_core.runnables import RunnableLambda

from chains.intent import classify
from models.intents import Intent, IntentResult


def _mock_llm(intent: Intent, confidence: float = 0.95, entities: dict | None = None):
    """Returns a fake ChatGroq-compatible object whose with_structured_output returns a fixed result."""
    fixed = IntentResult(intent=intent, confidence=confidence, entities=entities or {})

    class _FakeLLM:
        model_name = "mock"

        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: fixed)

    return _FakeLLM()


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
