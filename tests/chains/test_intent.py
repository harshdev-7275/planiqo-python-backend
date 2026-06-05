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


# --- un-crashable classify (Sprint 1.2) ------------------------------------


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_prose_only_response():
    """A model that returns free-text prose instead of JSON must NOT raise
    to the supervisor. Per AIService.md, every LLM call returns AIResult —
    the model is untrusted, the wrapper is responsible for graceful
    degradation. This is a regression test for the prod bug where a model
    swap caused classify() to raise ValidationError and the chat to 500."""
    prose_llm = RunnableLambda(lambda _: AIMessage(content="Sorry, I can't do that"))
    result = await classify("create a bug for login", llm=prose_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.entities == {}


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_markdown_only_response():
    """Markdown-wrapped prose with no JSON inside must be treated as
    UNKNOWN. The existing code already strips code fences — but if the
    content inside the fences is also prose, the parse must not raise."""
    md_llm = RunnableLambda(lambda _: AIMessage(
        content="```\nSome markdown content without any JSON\n```"
    ))
    result = await classify("create a bug for login", llm=md_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.entities == {}


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_empty_string_after_think_strip():
    """Some reasoning models emit only a <think>...</think> block with no
    user-visible content. The parse must not raise — UNKNOWN is the safe
    default."""
    empty_llm = RunnableLambda(lambda _: AIMessage(content="<think>...</think>"))
    result = await classify("create a bug for login", llm=empty_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_malformed_json():
    """Truncated JSON (missing closing brace) must be UNKNOWN, not a
    ValidationError bubbling up to the chat endpoint."""
    bad_llm = RunnableLambda(lambda _: AIMessage(
        content='{"intent": "CREATE_ISSUE", "confidence": 0.9'  # truncated
    ))
    result = await classify("create a bug for login", llm=bad_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.entities == {}


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_empty_content():
    """A model that returns an empty string must be UNKNOWN, not raise."""
    empty_llm = RunnableLambda(lambda _: AIMessage(content=""))
    result = await classify("create a bug for login", llm=empty_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_returns_unknown_on_json_missing_required_intent():
    """A well-formed JSON object that lacks the ``intent`` field must be
    UNKNOWN — the schema is enforced, the model is untrusted."""
    no_intent_llm = RunnableLambda(lambda _: AIMessage(
        content='{"confidence": 0.9, "entities": {}}'
    ))
    result = await classify("create a bug for login", llm=no_intent_llm)
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_never_raises_on_garbage_input():
    """Property test: for a sample of garbage strings, classify must return
    a valid IntentResult, never raise. This is the operational invariant —
    a 500 from the chat endpoint because classify raised is a sev-1."""
    garbage = [
        "not json at all",
        "{",
        "{}",
        "null",
        "[]",
        '{"intent": "NOT_A_REAL_INTENT"}',  # valid JSON, invalid enum
        '{"intent": null}',                # explicit null
    ]
    for content in garbage:
        llm = RunnableLambda(lambda _: AIMessage(content=content))
        result = await classify("create a bug for login", llm=llm)
        assert isinstance(result, IntentResult), f"raised on content={content!r}"
        assert result.intent == Intent.UNKNOWN, f"expected UNKNOWN for {content!r}, got {result.intent}"
