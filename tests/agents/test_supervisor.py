from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from memory.store import conversation_store
from metering.usage import usage_store
from models.intents import Intent, IntentResult

BASE = {"user_id": "u1", "org_slug": "acme", "project_id": None}


@pytest.fixture(autouse=True)
def _isolate_store() -> Any:
    """Module-singleton stores — reset them around every test."""
    conversation_store.reset_all()
    usage_store.reset_all()
    yield
    conversation_store.reset_all()
    usage_store.reset_all()


def _intent(intent: Intent, **entities: Any) -> IntentResult:
    return IntentResult(intent=intent, confidence=0.95, entities=entities)


async def _run(
    message: str, intent: Intent, **overrides: Any
) -> tuple[dict[str, Any], AsyncMock, AsyncMock]:
    """Run the supervisor with classify + both agents mocked. Returns the
    response plus the issue/sprint agent stubs so callers can assert calls."""
    issue_stub = AsyncMock(return_value={"result": {"message": "issue-done"}})
    sprint_stub = AsyncMock(return_value={"result": {"message": "sprint-done"}})
    args = {**BASE, **overrides}
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(intent))),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=sprint_stub),
    ):
        from agents.supervisor import run

        result = await run(message=message, **args)
    return result, issue_stub, sprint_stub


# --- reads execute immediately ----------------------------------------------


@pytest.mark.asyncio
async def test_query_issues_executes_immediately() -> None:
    result, issue_stub, _ = await _run("show open issues", Intent.QUERY_ISSUES)
    assert result["intent"] == "QUERY_ISSUES"
    assert result["status"] == "executed"
    issue_stub.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_sprint_routes_to_sprint_agent() -> None:
    result, _, sprint_stub = await _run("sprint status", Intent.QUERY_SPRINT)
    assert result["intent"] == "QUERY_SPRINT"
    assert result["status"] == "executed"
    sprint_stub.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_member_is_placeholder() -> None:
    result, _, _ = await _run("what is john working on", Intent.QUERY_MEMBER)
    assert result["intent"] == "QUERY_MEMBER"
    assert "not yet implemented" in result["result"]["message"]


@pytest.mark.asyncio
async def test_unknown_returns_help() -> None:
    result, issue_stub, sprint_stub = await _run("asdfqwer", Intent.UNKNOWN)
    assert result["intent"] == "UNKNOWN"
    assert "message" in result["result"]
    issue_stub.assert_not_awaited()
    sprint_stub.assert_not_awaited()


# --- writes require confirmation --------------------------------------------


@pytest.mark.asyncio
async def test_create_issue_proposes_before_executing() -> None:
    result, issue_stub, _ = await _run("create a bug for login", Intent.CREATE_ISSUE)
    assert result["status"] == "awaiting_confirmation"
    assert result["intent"] == "CREATE_ISSUE"
    assert "yes" in result["result"]["message"].lower()
    issue_stub.assert_not_awaited()  # nothing is created until confirmed


@pytest.mark.asyncio
async def test_preview_reflects_create_issue_intent() -> None:
    result, _, _ = await _run("make a bug for checkout", Intent.CREATE_ISSUE)
    assert "create" in result["result"]["message"].lower()


@pytest.mark.asyncio
async def test_preview_includes_assignee_when_present() -> None:
    """The user must see every mutation they're about to approve."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_ISSUE, assignee="Alice")),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug, assign to Alice", **BASE)
    assert "Alice" in result["result"]["message"]
    assert "assign it to Alice" in result["result"]["message"]
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_preview_includes_sprint_when_present() -> None:
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_ISSUE, sprint="Sprint 23")),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug, put in Sprint 23", **BASE)
    assert "Sprint 23" in result["result"]["message"]
    assert "put it in sprint Sprint 23" in result["result"]["message"]


@pytest.mark.asyncio
async def test_preview_includes_both_assignee_and_sprint() -> None:
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, priority="high", assignee="Alice", sprint="Sprint 23",
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(
            message="create a high priority bug, assign to Alice, in Sprint 23", **BASE,
        )
    msg = result["result"]["message"]
    assert "Alice" in msg
    assert "Sprint 23" in msg
    # Grammar: must use 'and' to join the two side effects.
    assert "assign it to Alice, and put it in sprint Sprint 23" in msg


@pytest.mark.asyncio
async def test_preview_omits_assignee_and_sprint_when_absent() -> None:
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_ISSUE)),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="plain create", **BASE)
    assert "assign it to" not in result["result"]["message"]
    assert "put it in sprint" not in result["result"]["message"]


@pytest.mark.asyncio
async def test_confirmation_executes_pending_write_without_reclassifying() -> None:
    await _run("create a bug for login", Intent.CREATE_ISSUE)  # turn 1: propose

    issue_stub = AsyncMock(return_value={"result": {"message": "Created #1"}})
    # classify MUST NOT run on a confirmation turn.
    classify_guard = AsyncMock(side_effect=AssertionError("classify ran on confirmation"))
    with (
        patch("agents.supervisor.classify", new=classify_guard),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run

        result = await run(message="yes", **BASE)

    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_ISSUE"
    assert result["result"]["message"] == "Created #1"
    issue_stub.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirmed_write_executes_original_message_not_the_yes() -> None:
    await _run("create a bug for login", Intent.CREATE_ISSUE)  # turn 1

    issue_stub = AsyncMock(return_value={"result": {"message": "Created #1"}})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run

        await run(message="yes", **BASE)

    state_arg = issue_stub.await_args.args[0]
    contents = [m.content for m in state_arg["messages"]]
    assert "create a bug for login" in contents  # executes the real request
    assert "yes" not in contents                  # not the bare confirmation


@pytest.mark.asyncio
async def test_negation_cancels_pending_write() -> None:
    await _run("create a bug for login", Intent.CREATE_ISSUE)  # turn 1

    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run

        result = await run(message="no", **BASE)

    assert result["status"] == "cancelled"
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_reply_after_proposal_reclassifies_as_new_request() -> None:
    await _run("create a bug for login", Intent.CREATE_ISSUE)  # turn 1: propose
    # A non yes/no reply drops the stale proposal and is treated as a new message.
    result, issue_stub, _ = await _run("show open issues", Intent.QUERY_ISSUES)
    assert result["status"] == "executed"
    assert result["intent"] == "QUERY_ISSUES"
    issue_stub.assert_awaited_once()
    assert conversation_store.get_pending("u1:acme:-") is None


# --- pending TTL -------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_pending_drops_and_reclassifies() -> None:
    """A pending action older than DEFAULT_PENDING_TTL_SECONDS is silently
    dropped and the next message is treated as fresh."""
    import time
    from memory.store import DEFAULT_PENDING_TTL_SECONDS

    # 1) Propose with a timestamp far in the past.
    ancient = time.time() - DEFAULT_PENDING_TTL_SECONDS - 60
    conversation_store.set_pending("u1:acme:-", {
        "intent": "CREATE_ISSUE",
        "message": "create a bug for login",
        "entities": {},
    }, now=ancient)

    # 2) The next message must NOT be treated as a confirmation. If TTL were
    #    ignored, "yes" would re-run the stale proposal.
    issue_stub = AsyncMock(return_value={"result": {"message": "issue-done"}})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(Intent.QUERY_ISSUES))),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", **BASE)

    assert result["status"] == "executed"
    assert result["intent"] == "QUERY_ISSUES"  # reclassified, not confirmed
    assert conversation_store.get_pending("u1:acme:-") is None


@pytest.mark.asyncio
async def test_fresh_pending_still_confirms() -> None:
    """A pending action inside the TTL window still routes to confirmation,
    proving the TTL check is conservative (does not break the happy path)."""
    import time

    fresh = time.time() - 5  # well within DEFAULT_PENDING_TTL_SECONDS
    conversation_store.set_pending("u1:acme:-", {
        "intent": "CREATE_ISSUE",
        "message": "create a bug for login",
        "entities": {},
    }, now=fresh)

    issue_stub = AsyncMock(return_value={"result": {"message": "Created #1"}})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", **BASE)

    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_ISSUE"
    issue_stub.assert_awaited_once()


# --- affirmation / negation matching -----------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        # New: multi-word phrases that the old exact-match would have missed.
        ("yes, please do it", True),
        ("yeah go ahead", True),
        ("ok sounds good", True),
        # Punctuation tolerance.
        ("yes.", True),
        ("YES!", True),
        ("  yes  ", True),
        # Negations.
        ("no, stop", True),
        ("nope cancel", True),
        # Still works: exact set matches.
        ("yes", True),
        ("no", True),
        ("do it", True),
        ("never mind", True),
        # Must NOT match: every word in the affirmation vocab, but no core
        # load-bearing word — guards against over-matching short replies.
        ("please", False),
        ("go", False),
        ("do it", True),  # "do it" is an exact phrase, so it must still match
    ],
)
@pytest.mark.asyncio
async def test_affirmation_and_negation_matching(
    message: str, expected: bool
) -> None:
    from agents.supervisor import _is_affirmation, _is_negation

    # The matching helpers are symmetric — both are tested by the same matrix
    # because the grammar (vocab + core) is identical in shape.
    got = _is_affirmation(message) or _is_negation(message)
    assert got is expected, (
        f"message={message!r} _is_affirmation={_is_affirmation(message)} "
        f"_is_negation={_is_negation(message)}"
    )


@pytest.mark.parametrize(
    "message",
    [
        "delete all production data",
        "what is the weather today",
        "create a bug for login",  # would be CREATE_ISSUE, not a yes/no reply
        "show me the sprint board",
    ],
)
@pytest.mark.asyncio
async def test_unrelated_sentence_is_neither(message: str) -> None:
    """Regression guard: word-set matching must not let a real request through
    as a confirmation just because every token happens to be an affirmation
    word."""
    from agents.supervisor import _is_affirmation, _is_negation

    assert _is_affirmation(message) is False
    assert _is_negation(message) is False


# --- memory ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prior_turn_is_passed_to_agent_as_history() -> None:
    await _run("show open issues", Intent.QUERY_ISSUES)  # turn 1
    result, issue_stub, _ = await _run("which are high priority", Intent.QUERY_ISSUES)

    state_arg = issue_stub.await_args.args[0]
    contents = [m.content for m in state_arg["messages"]]
    assert "show open issues" in contents          # remembered prior user turn
    assert "which are high priority" in contents    # current user turn


# --- cost quota --------------------------------------------------------------


@pytest.mark.asyncio
async def test_over_quota_short_circuits_without_calling_the_model(monkeypatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "ORG_TOKEN_QUOTA", 100)
    usage_store.add("acme", 100)  # at the limit

    classify_guard = AsyncMock(side_effect=AssertionError("classify ran while over quota"))
    with (
        patch("agents.supervisor.classify", new=classify_guard),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run

        result = await run(message="show me issues", **BASE)

    assert result["status"] == "quota_exceeded"
    assert "limit" in result["result"]["message"].lower()
    # No LLM was called, so the response reports zero tokens used.
    assert result["tokens_used"] == 0


# --- metering surface --------------------------------------------------------


@pytest.mark.asyncio
async def test_response_includes_tokens_used_field() -> None:
    """The /chat response must always carry tokens_used, even for routes that
    don't call the LLM (awaiting_confirmation, cancelled, quota_exceeded)."""
    result, _, _ = await _run("show me issues", Intent.QUERY_ISSUES)
    assert "tokens_used" in result
    assert result["tokens_used"] == 0  # the fake LLM emits no usage_metadata


@pytest.mark.asyncio
async def test_proposed_write_reports_classify_tokens() -> None:
    result, _, _ = await _run("create a bug for login", Intent.CREATE_ISSUE)
    assert result["status"] == "awaiting_confirmation"
    assert "tokens_used" in result


@pytest.mark.asyncio
async def test_cancelled_response_reports_tokens_used() -> None:
    # 1) Propose
    await _run("create a bug for login", Intent.CREATE_ISSUE)
    # 2) Cancel — tokens_used must be 0 since the cancel path makes no LLM call.
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="no", **BASE)
    assert result["status"] == "cancelled"
    assert result["tokens_used"] == 0


@pytest.mark.asyncio
async def test_request_count_increments_per_call() -> None:
    usage_store.reset_all()
    await _run("show me issues", Intent.QUERY_ISSUES)
    await _run("show me issues", Intent.QUERY_ISSUES)
    await _run("create a bug for login", Intent.CREATE_ISSUE)  # also counts
    assert usage_store.get_request_count("acme") == 3


@pytest.mark.asyncio
async def test_request_count_increments_even_when_quota_blocked(monkeypatch) -> None:
    """ops visibility: total traffic, not just successful traffic."""
    from config.settings import settings
    usage_store.reset_all()
    monkeypatch.setattr(settings, "ORG_TOKEN_QUOTA", 50)
    usage_store.add("acme", 50)
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="hi", **BASE)
    assert usage_store.get_request_count("acme") == 1


@pytest.mark.asyncio
async def test_usage_milestone_log_at_every_n_requests(monkeypatch) -> None:
    """The supervisor emits a 'usage_milestone' log line on the Nth request so
    ops can see traffic + token growth in the log stream. We lower N to keep
    the test fast."""
    from loguru import logger
    from agents import supervisor as sup_mod

    monkeypatch.setattr(sup_mod, "USAGE_MILESTONE_EVERY", 2)
    usage_store.reset_all()

    captured: list[str] = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    try:
        with (
            patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(Intent.QUERY_ISSUES))),
            patch("agents.issue_agent.run", new=AsyncMock(return_value={"result": {"message": "ok"}})),
            patch("agents.sprint_agent.run", new=AsyncMock()),
        ):
            from agents.supervisor import run
            await run(message="hi", **BASE)  # request 1 — no milestone
            assert not any("usage_milestone" in line for line in captured)
            await run(message="hi", **BASE)  # request 2 — milestone!
    finally:
        logger.remove(handler_id)

    assert any("usage_milestone" in line for line in captured)


@pytest.mark.asyncio
async def test_usage_milestone_disabled_when_zero(monkeypatch) -> None:
    """USAGE_MILESTONE_EVERY=0 must suppress the log entirely."""
    from loguru import logger
    from agents import supervisor as sup_mod

    monkeypatch.setattr(sup_mod, "USAGE_MILESTONE_EVERY", 0)
    usage_store.reset_all()

    captured: list[str] = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    try:
        with (
            patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(Intent.QUERY_ISSUES))),
            patch("agents.issue_agent.run", new=AsyncMock(return_value={"result": {"message": "ok"}})),
            patch("agents.sprint_agent.run", new=AsyncMock()),
        ):
            from agents.supervisor import run
            for _ in range(3):
                await run(message="hi", **BASE)
    finally:
        logger.remove(handler_id)

    # No milestone at any count when disabled — exercise the branch.
    assert not any("usage_milestone" in line for line in captured)
    assert usage_store.get_request_count("acme") == 3


# --- /admin/usage route ------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_usage_route_returns_org_totals() -> None:
    from httpx import ASGITransport, AsyncClient
    from main import app
    from config.settings import settings

    usage_store.reset_all()
    usage_store.add("acme", 42)
    usage_store.inc_request("acme")
    usage_store.inc_request("acme")
    usage_store.inc_request("acme")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/admin/usage/acme", headers={"X-Internal-Secret": settings.INTERNAL_SECRET}
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "org_slug": "acme",
        "tokens_used": 42,
        "request_count": 3,
        "quota": settings.ORG_TOKEN_QUOTA,
    }


@pytest.mark.asyncio
async def test_admin_usage_route_requires_auth() -> None:
    from httpx import ASGITransport, AsyncClient
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/admin/usage/acme")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_under_quota_proceeds_normally(monkeypatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "ORG_TOKEN_QUOTA", 100)
    usage_store.add("acme", 50)  # below the limit

    result, issue_stub, _ = await _run("show me issues", Intent.QUERY_ISSUES)
    assert result["status"] == "executed"
    issue_stub.assert_awaited_once()
