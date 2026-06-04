from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from metering.callback import UsageCallback, extract_total_tokens
from metering.usage import RequestTokens, UsageStore


def _result_with_usage_metadata(total: int) -> LLMResult:
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": total},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _result_with_llm_output(total: int) -> LLMResult:
    return LLMResult(generations=[[]], llm_output={"token_usage": {"total_tokens": total}})


def test_extract_from_usage_metadata() -> None:
    assert extract_total_tokens(_result_with_usage_metadata(15)) == 15


def test_extract_from_llm_output() -> None:
    assert extract_total_tokens(_result_with_llm_output(42)) == 42


def test_extract_missing_tokens_is_zero() -> None:
    assert extract_total_tokens(LLMResult(generations=[[]])) == 0


def test_callback_records_tokens_to_store() -> None:
    store = UsageStore()
    callback = UsageCallback(org_slug="acme", user_id="u1", store=store)
    callback.on_llm_end(_result_with_usage_metadata(20))
    assert store.get("acme") == 20


def test_callback_ignores_zero_token_calls() -> None:
    store = UsageStore()
    callback = UsageCallback(org_slug="acme", user_id="u1", store=store)
    callback.on_llm_end(LLMResult(generations=[[]]))
    assert store.get("acme") == 0


def test_callback_accumulates_into_request_tokens() -> None:
    """Per-request accumulator: each LLM end-event bumps the running total so
    the supervisor can report tokens_used in the /chat response."""
    store = UsageStore()
    rt = RequestTokens()
    callback = UsageCallback(org_slug="acme", user_id="u1", store=store, request_tokens=rt)
    callback.on_llm_end(_result_with_usage_metadata(5))
    callback.on_llm_end(_result_with_usage_metadata(7))
    assert rt.total == 12
    # Both totals land in the org store AND the per-request bucket.
    assert store.get("acme") == 12


def test_callback_without_request_tokens_still_updates_store() -> None:
    """``request_tokens`` is optional — the store must still be updated."""
    store = UsageStore()
    callback = UsageCallback(org_slug="acme", user_id="u1", store=store)
    callback.on_llm_end(_result_with_usage_metadata(8))
    assert store.get("acme") == 8
