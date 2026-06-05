"""Tests for the intent eval harness (item 21).

The harness's *scoring logic* is unit-tested with a stubbed classifier (runs in
CI, no API). The actual *accuracy gate* hits the real classifier and is marked
``integration`` — run it manually before a model swap or in a scheduled job:

    uv run pytest tests/evals -m integration
"""

from __future__ import annotations

import pytest

from models.intents import Intent, IntentResult
from tests.evals.intent_accuracy import (
    ACCURACY_THRESHOLD,
    evaluate,
    load_golden,
)


def _stub(mapping: dict[str, Intent]):
    async def classify_fn(query: str) -> IntentResult:
        return IntentResult(
            intent=mapping.get(query, Intent.UNKNOWN), confidence=1.0, entities={}
        )
    return classify_fn


# --- dataset sanity ---------------------------------------------------------


def test_golden_set_loads_and_has_valid_intents() -> None:
    cases = load_golden()
    assert len(cases) >= 20
    valid = {i.value for i in Intent}
    for case in cases:
        assert case["query"]
        assert case["expected"] in valid, f"bad expected intent: {case}"


def test_every_write_and_read_intent_is_covered() -> None:
    """The golden set must exercise each user-facing intent, or a regression in
    an uncovered intent would pass silently."""
    covered = {c["expected"] for c in load_golden()}
    for intent in (
        Intent.CREATE_ISSUE, Intent.UPDATE_ISSUE, Intent.QUERY_ISSUES,
        Intent.QUERY_SPRINT, Intent.CREATE_SPRINT, Intent.QUERY_MEMBER,
        Intent.SUMMARIZE, Intent.UNKNOWN,
    ):
        assert intent.value in covered, f"{intent.value} not in golden set"


# --- scoring logic ----------------------------------------------------------


@pytest.mark.asyncio
async def test_perfect_classifier_scores_100() -> None:
    cases = load_golden()
    perfect = _stub({c["query"]: Intent(c["expected"]) for c in cases})
    result = await evaluate(perfect, cases)
    assert result.accuracy == 1.0
    assert result.mismatches == []


@pytest.mark.asyncio
async def test_mismatches_are_recorded() -> None:
    dataset = [
        {"query": "show me all issues", "expected": "QUERY_ISSUES"},
        {"query": "create a bug", "expected": "CREATE_ISSUE"},
    ]
    # Classifier gets the first right, the second wrong (returns UNKNOWN).
    stub = _stub({"show me all issues": Intent.QUERY_ISSUES})
    result = await evaluate(stub, dataset)
    assert result.total == 2
    assert result.correct == 1
    assert result.accuracy == 0.5
    assert len(result.mismatches) == 1
    m = result.mismatches[0]
    assert m.query == "create a bug"
    assert m.expected == "CREATE_ISSUE"
    assert m.got == "UNKNOWN"
    assert "create a bug" in result.report()


# --- the real gate (integration) --------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_classifier_meets_accuracy_threshold() -> None:
    """Hits the real LLM. Skipped on the default (non-integration) run."""
    from chains.intent import classify

    result = await evaluate(classify)
    assert result.accuracy >= ACCURACY_THRESHOLD, "\n" + result.report()
