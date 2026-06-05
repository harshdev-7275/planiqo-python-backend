from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    member_stub = AsyncMock(return_value={"result": {"message": "member-done"}})
    summarize_stub = AsyncMock(return_value={"result": {"message": "summary-done"}})
    args = {**BASE, **overrides}
    # A CREATE_ISSUE with no title now short-circuits to a "what should I title
    # this?" clarify (the missing-title gate), and a title with words the user
    # never typed is regrounded (the fabrication guard). The real classifier
    # extracts a title FROM the message, so the default fixture uses the message
    # itself — always grounded, never empty. Tests that exercise the missing /
    # fabricated paths set classify up themselves.
    default_entities = {"title": message} if intent == Intent.CREATE_ISSUE else {}
    with (
        # Defer the pre-router to classify so these tests exercise the
        # classify + metering path. Tests that want to exercise the pre-router
        # itself patch it separately.
        patch("agents.supervisor.pre_route", new=AsyncMock(return_value=None)),
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(intent, **default_entities)),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=sprint_stub),
        patch("agents.member_agent.run", new=member_stub),
        patch("agents.summarize_agent.run", new=summarize_stub),
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
async def test_query_member_routes_to_member_agent() -> None:
    result, issue_stub, sprint_stub = await _run(
        "what is john working on", Intent.QUERY_MEMBER
    )
    assert result["intent"] == "QUERY_MEMBER"
    assert result["status"] == "executed"
    assert result["result"]["message"] == "member-done"  # member_agent ran
    # Reads route to exactly one agent.
    issue_stub.assert_not_awaited()
    sprint_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_routes_to_summarize_agent() -> None:
    result, issue_stub, sprint_stub = await _run("summarize sprint 2", Intent.SUMMARIZE)
    assert result["intent"] == "SUMMARIZE"
    assert result["status"] == "executed"
    assert result["result"]["message"] == "summary-done"  # summarize_agent ran
    issue_stub.assert_not_awaited()
    sprint_stub.assert_not_awaited()


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
async def test_awaiting_confirmation_exposes_confirm_cancel_actions() -> None:
    """Item 18: a proposal carries Confirm/Cancel button affordances whose
    values are the same yes/no tokens a typed reply uses."""
    result, _, _ = await _run("create a bug for login", Intent.CREATE_ISSUE)
    assert result["status"] == "awaiting_confirmation"
    actions = result["actions"]
    titles = {a["title"] for a in actions}
    assert titles == {"Confirm", "Cancel"}
    # The values must be the literal yes/no the confirm path recognises.
    by_title = {a["title"]: a["value"] for a in actions}
    from agents.supervisor import _is_affirmation, _is_negation
    assert _is_affirmation(by_title["Confirm"]) is True
    assert _is_negation(by_title["Cancel"]) is True


@pytest.mark.asyncio
async def test_preview_includes_assignee_when_present() -> None:
    """The user must see every mutation they're about to approve."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, title="checkout bug", assignee="Alice",
            )),
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
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, title="checkout bug", sprint="Sprint 23",
            )),
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
                Intent.CREATE_ISSUE, title="checkout bug", priority="high",
                assignee="Alice", sprint="Sprint 23",
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
async def test_create_issue_without_title_asks_for_a_title() -> None:
    """When the LLM extracts no title, the bot must ASK for one rather than
    proposing an issue literally titled '(untitled)'. Regression for stress
    queries #2 ('create a bug'), #27 (the long ramble), #33 ('create a new
    task with high priority')."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, type="task", priority="high",
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a new task with high priority", **BASE)

    msg = result["result"]["message"]
    assert result["status"] == "needs_input"
    assert "(untitled)" not in msg
    assert "title" in msg.lower()
    # No pending proposal is set — there is nothing to confirm yet.
    assert conversation_store.get_pending("u1:acme:-") is None
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_issue_with_blank_title_asks_for_a_title() -> None:
    """A title that cleans down to nothing (whitespace / pure punctuation) is
    treated the same as a missing title."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_ISSUE, title="   ")),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug", **BASE)
    assert result["status"] == "needs_input"
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_fabricated_title_is_regrounded_in_preview() -> None:
    """Items 8/9: if the LLM invents title words the user never typed, the
    preview must use the grounded words, not the hallucination."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE,
                title="login authentication failure on production",
                type="bug", priority="high",
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug for the login page", **BASE)
    msg = result["result"]["message"]
    # "authentication"/"production" were never in the message — must be pruned.
    assert "authentication" not in msg
    assert "production" not in msg
    assert "login" in msg  # the grounded word survives
    assert result["status"] == "awaiting_confirmation"


@pytest.mark.asyncio
async def test_fully_fabricated_title_falls_through_to_missing_title_gate() -> None:
    """If NONE of the invented title is grounded, there's no usable title left
    — the bot asks rather than confirming a hallucination."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, title="database connection pool exhaustion",
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug for login", **BASE)
    assert result["status"] == "needs_input"


@pytest.mark.asyncio
async def test_fabricated_sprint_name_uses_quoted_original() -> None:
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_SPRINT, name="Performance Improvements")),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a sprint called 'Q3 Hardening'", **BASE)
    msg = result["result"]["message"]
    assert "Q3 Hardening" in msg
    assert "Performance" not in msg


@pytest.mark.asyncio
async def test_preview_defaults_priority_and_type_when_entities_are_null() -> None:
    """When the LLM returns `priority: null` / `type: null` (which the
    classifier does — the JSON schema does not enforce non-null), the preview
    must default to 'medium priority' / 'task' — never leak the literal
    Python 'None' into user copy. Regression for the transcript bug where
    the user saw 'with None priority' in the proposal."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, title="Login broken", type=None, priority=None,
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        # Message has no convention trigger word (no "bug"/"urgent"/…), so the
        # null entities fall through to the plain medium/task defaults.
        result = await run(message="create a login screen", **BASE)
    msg = result["result"]["message"]
    assert "medium priority" in msg, f"expected default 'medium' priority, got: {msg!r}"
    assert "None" not in msg, f"preview must not leak 'None' to the user, got: {msg!r}"
    assert "create a task titled" in msg, f"expected default 'task' type, got: {msg!r}"


@pytest.mark.asyncio
async def test_bug_convention_applied_through_supervisor() -> None:
    """End-to-end: when the LLM returns null type/priority but the request
    says 'bug', the convention normalizer fills type=bug + priority=high, and
    that is what the preview proposes (and what would be executed on confirm)."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.CREATE_ISSUE, title="Login broken", type=None, priority=None,
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="create a bug for the login page", **BASE)
    msg = result["result"]["message"]
    assert "create a bug" in msg, f"expected type=bug, got: {msg!r}"
    assert "high priority" in msg, f"expected priority=high, got: {msg!r}"
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_preview_omits_assignee_and_sprint_when_absent() -> None:
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(Intent.CREATE_ISSUE, title="plain bug")),
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
    """A confirmed write is dispatched to the deterministic executor (NOT the
    ReAct agent) — preview == execution by construction. The ReAct issue_agent
    must NOT be called on a confirmation turn."""
    # Turn 1: propose with populated entities.
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "login bug", "type": "bug", "priority": "high"},
    ))
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
        patch("agents.member_agent.run", new=AsyncMock()),
        patch("agents.summarize_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="create a bug for login", **BASE)

    # Turn 2: confirm. classify MUST NOT run.
    issue_stub = AsyncMock(return_value={"result": {"message": "should-not-run"}})
    fake_api = AsyncMock()
    fake_api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    fake_api.post = AsyncMock(return_value={"number": 1, "title": "login bug"})
    classify_guard = AsyncMock(side_effect=AssertionError("classify ran on confirmation"))
    with (
        patch("agents.supervisor.classify", new=classify_guard),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", **BASE)

    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_ISSUE"
    # Executor posted the validated payload; ReAct agent was never invoked.
    issue_stub.assert_not_awaited()
    fake_api.post.assert_awaited_once()
    # BASE has project_id=None → executor's path uses the empty project_id.
    assert fake_api.post.await_args.args[0] == "/orgs/acme/projects//issues"


@pytest.mark.asyncio
async def test_confirmed_write_posts_validated_entities_not_yes() -> None:
    """The POST body must be the VALIDATED entities (title="login bug", etc.),
    not the bare confirmation word "yes". Locks the architectural invariant
    that the executor uses stored validated entities, not the LLM re-derived
    ones from a re-replayed message."""
    # Turn 1: propose.
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "login bug", "type": "bug", "priority": "high"},
    ))
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
        patch("agents.member_agent.run", new=AsyncMock()),
        patch("agents.summarize_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="create a bug for login", **BASE)

    # Turn 2: confirm.
    fake_api = AsyncMock()
    fake_api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    fake_api.post = AsyncMock(return_value={"number": 1, "title": "login bug"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="yes", **BASE)

    body = fake_api.post.await_args.args[1]
    # The body uses the validated entities (title, type=bug, priority=high
    # from normalizer), NOT a re-derivation. It must not contain "yes".
    assert body["title"] == "login bug"
    assert body["type"] == "bug"
    assert body["priority"] == "high"
    assert "yes" not in str(body).lower()


# --- write_executor dispatch: full drift regression --------------------------


@pytest.mark.asyncio
async def test_drift_regression_create_bug_for_alice_to_sprint_1() -> None:
    """HEADLINE E2E DRIFT TEST. The chain of events is:

    1. User: "create a bug for the login page, assign to Alice, in Sprint 1"
    2. classify returns {type:bug, priority:null, assignee:Alice, sprint:Sprint 1}
    3. normalizer applies bug → priority=high convention
    4. validation resolves Alice → u-123 and Sprint 1 → s-456
    5. preview is shown, user says "yes"
    6. The POST body must be EXACTLY the validated payload:
       {title, type=bug, priority=high, assigneeId=u-123, sprintId=s-456, statusId=...}

    This test was the original motivation for the entire Sprint 1.1 — if it
    fails, the architecture is broken."""
    # Turn 1: propose. classify returns the raw LLM entities; the normalizer
    # in supervisor.run applies the bug→high convention; the validation gate
    # resolves Alice→u-123 and Sprint 1→s-456 and stashes them in pending.
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={
            "title": "login page", "type": "bug", "priority": None,
            "assignee": "Alice", "sprint": "Sprint 1",
        },
    ))
    # Mocks that the validation gate calls.
    fake_graph = MagicMock()
    fake_graph.run = AsyncMock(return_value=[{"id": "u-123", "name": "Alice"}])
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.supervisor.neo4j_client", new=fake_graph),
        patch("agents.supervisor.node_api_client", new=AsyncMock(
            get_sprints=AsyncMock(return_value=[
                {"id": "s-456", "name": "Sprint 1"},
            ]),
        )),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
        patch("agents.member_agent.run", new=AsyncMock()),
        patch("agents.summarize_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(
            message="create a bug for the login page, assign to Alice, in Sprint 1",
            user_id="u1", org_slug="acme", project_id="proj-1",
        )

    # Turn 2: confirm. The merged entities+resolved are what hit the POST.
    fake_api = AsyncMock()
    fake_api.get_default_status = AsyncMock(return_value={"id": "status-todo", "name": "Todo", "isDefault": True})
    fake_api.post = AsyncMock(return_value={"number": 7, "title": "login page"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(
            message="yes",
            user_id="u1", org_slug="acme", project_id="proj-1",
        )

    # Status + result shape unchanged for the user.
    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_ISSUE"

    # The headline assertions: validated IDs reached the POST, name-rederivation did not.
    assert fake_api.post.await_count == 1
    path, body = fake_api.post.await_args.args[:2]
    assert path == "/orgs/acme/projects/proj-1/issues"
    assert body["title"] == "login page"        # the cleaned title from the LLM
    assert body["type"] == "bug"                # from the bug-keyword convention
    assert body["priority"] == "high"           # from the bug-keyword convention
    assert body["assigneeId"] == "u-123"        # resolved at validation, NOT re-derived
    assert body["sprintId"] == "s-456"          # resolved at validation, NOT re-derived
    assert body["statusId"] == "status-todo"    # default status


@pytest.mark.asyncio
async def test_update_issue_canonicalizes_literal_p0_priority_in_preview() -> None:
    """'set issue 5 to P0' must preview (and later execute) priority=critical,
    not the literal 'P0' the LLM copied in. Mirror of the CREATE_ISSUE #32 fix
    for the update path, which does not run the full keyword normalizer."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    with (
        patch(
            "agents.supervisor.classify",
            new=AsyncMock(return_value=_intent(
                Intent.UPDATE_ISSUE, issue=5, priority="P0",
            )),
        ),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="set issue 5 to P0", **BASE)
    msg = result["result"]["message"]
    assert "critical" in msg
    assert "P0" not in msg
    issue_stub.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_update_issue_dispatches_to_executor() -> None:
    """UPDATE_ISSUE confirmed writes also bypass the ReAct agent."""
    # Turn 1: propose with populated entities.
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.UPDATE_ISSUE, confidence=0.95,
        entities={"issue": 5, "priority": "high"},
    ))
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
        patch("agents.member_agent.run", new=AsyncMock()),
        patch("agents.summarize_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(
            message="set issue 5 priority to high",
            user_id="u1", org_slug="acme", project_id="proj-1",
        )

    # Turn 2: confirm.
    fake_api = AsyncMock()
    fake_api.get_issues = AsyncMock(return_value=[
        {"number": 5, "id": "issue-uuid-5", "title": "Login broken", "priority": "low"},
    ])
    fake_api.patch = AsyncMock(return_value={"number": 5})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", user_id="u1", org_slug="acme", project_id="proj-1")

    assert result["status"] == "executed"
    assert result["intent"] == "UPDATE_ISSUE"
    # Executor PATCHed the resolved UUID with only the validated field.
    assert fake_api.patch.await_args.args[0] == "/orgs/acme/projects/proj-1/issues/issue-uuid-5"
    assert fake_api.patch.await_args.args[1] == {"priority": "high"}


@pytest.mark.asyncio
async def test_confirmed_create_issue_sends_idempotency_key() -> None:
    """Item 17: the confirmed create POSTs an Idempotency-Key, and it's the
    SAME key that was minted on the proposal turn (so a retried confirm dedupes)."""
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "login bug", "type": "bug", "priority": "high"},
    ))
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
        patch("agents.member_agent.run", new=AsyncMock()),
        patch("agents.summarize_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="create a bug for login", **BASE)

    key = conversation_store.get_pending("u1:acme:-")["idempotency_key"]
    assert key  # minted at proposal time

    fake_api = AsyncMock()
    fake_api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    fake_api.post = AsyncMock(return_value={"number": 1, "title": "login bug"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        await run(message="yes", **BASE)

    # The executor passed the minted key through to api.post.
    assert fake_api.post.await_args.kwargs["idempotency_key"] == key


@pytest.mark.asyncio
async def test_confirmed_create_sprint_dispatches_to_executor() -> None:
    """CREATE_SPRINT confirmed writes also bypass the ReAct agent."""
    # Turn 1: propose with a populated entity. We patch classify directly
    # so the entities land in the pending proposal.
    classify_mock = AsyncMock(return_value=IntentResult(
        intent=Intent.CREATE_SPRINT, confidence=0.95,
        entities={"name": "Sprint 5"},
    ))
    issue_stub = AsyncMock()
    sprint_stub = AsyncMock()
    member_stub = AsyncMock()
    summarize_stub = AsyncMock()
    with (
        patch("agents.supervisor.classify", new=classify_mock),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=sprint_stub),
        patch("agents.member_agent.run", new=member_stub),
        patch("agents.summarize_agent.run", new=summarize_stub),
    ):
        from agents.supervisor import run
        await run(
            message="create a sprint called Sprint 5",
            user_id="u1", org_slug="acme", project_id="proj-1",
        )

    # Turn 2: confirm.
    fake_api = AsyncMock()
    fake_api.post = AsyncMock(return_value={"name": "Sprint 5"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", user_id="u1", org_slug="acme", project_id="proj-1")

    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_SPRINT"
    assert fake_api.post.await_args.args[0] == "/orgs/acme/projects/proj-1/sprints"
    assert fake_api.post.await_args.args[1] == {"name": "Sprint 5"}


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
    proving the TTL check is conservative (does not break the happy path).
    Confirmed writes now go to the write_executor (not the ReAct agent)."""
    import time

    fresh = time.time() - 5  # well within DEFAULT_PENDING_TTL_SECONDS
    conversation_store.set_pending("u1:acme:-", {
        "intent": "CREATE_ISSUE",
        "message": "create a bug for login",
        "entities": {"title": "login bug", "type": "bug", "priority": "high"},
        "resolved": {},
    }, now=fresh)

    issue_stub = AsyncMock(return_value={"result": {"message": "should-not-run"}})
    fake_api = AsyncMock()
    fake_api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    fake_api.post = AsyncMock(return_value={"number": 1, "title": "login bug"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=AssertionError)),
        patch("agents.supervisor.node_api_client", new=fake_api),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="yes", **BASE)

    assert result["status"] == "executed"
    assert result["intent"] == "CREATE_ISSUE"
    issue_stub.assert_not_awaited()
    fake_api.post.assert_awaited_once()


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
        # Contextual negations — common replies after a validation failure
        # where the bot said "No sprint matching 'X'…". The user is saying
        # "yeah I see, cancel" without literally typing "no".
        ("not found", True),
        ("doesn't exist", True),
        ("wrong one", True),
        ("missing", True),
        ("none of those", True),
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Drop leading verb + optional determiner.
        ("Create a bug", "bug"),
        ("Add login button", "login button"),
        ("File an issue", "issue"),
        ("Log a thing", "thing"),
        # Drop trailing prepositional meta.
        ("Login page to Alice", "Login page"),
        ("Bug for the login page to alice", "Bug for the login page"),
        ("Issue in the dashboard", "Issue in the dashboard"),
        # Drop leading AND trailing.
        ("Create a login page to Alice", "login page"),
        ("Add a bug for checkout to Bob", "bug for checkout"),
        # No-change cases.
        ("Login broken", "Login broken"),
        ("Bug", "Bug"),
        # Whitespace collapse + trailing punctuation.
        ("  Make   a   thing  ", "thing"),
        ("Login page,", "Login page"),
        # Strip that would leave too little — restored.
        ("Bug in", "Bug in"),
        # Unicode + straight quotes wrapping the title (item 13).
        ("'Onboarding wizard'", "Onboarding wizard"),
        ("“Onboarding wizard”", "Onboarding wizard"),
        ("‘Login broken’", "Login broken"),
        ("`Login broken`", "Login broken"),
        # Leading verb + quoted remainder: verb stripped, quotes stripped.
        ("Create 'Onboarding wizard'", "Onboarding wizard"),
        # Multi-line input collapses to a single line (item 13).
        ("Fix the\ncheckout flow", "Fix the checkout flow"),
        ("Login button\n\n   is broken", "Login button is broken"),
        # Empty / None.
        ("", ""),
    ],
)
def test_clean_title(raw: str | None, expected: str) -> None:
    """The preview uses a cleaned title so the user sees the issue name, not
    the verb phrase. Regression for 'Create a bug for the login page to Alice'
    producing a preview titled 'bug for the login page to Alice'."""
    from agents.supervisor import _clean_title

    assert _clean_title(raw) == expected


@pytest.mark.asyncio
async def test_prior_turn_is_passed_to_agent_as_history() -> None:
    await _run("show open issues", Intent.QUERY_ISSUES)  # turn 1
    result, issue_stub, _ = await _run("which are high priority", Intent.QUERY_ISSUES)

    state_arg = issue_stub.await_args.args[0]
    contents = [m.content for m in state_arg["messages"]]
    assert "show open issues" in contents          # remembered prior user turn
    assert "which are high priority" in contents    # current user turn


# --- low-confidence write guard (item 11) -----------------------------------


@pytest.mark.asyncio
async def test_low_confidence_write_asks_to_clarify() -> None:
    """A write intent below MIN_WRITE_CONFIDENCE must ask the user to clarify,
    not silently propose a mutation."""
    issue_stub = AsyncMock(return_value={"result": {"message": "should not run"}})
    low = IntentResult(intent=Intent.CREATE_ISSUE, confidence=0.4,
                       entities={"title": "something"})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=low)),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="uh do the thing maybe", **BASE)
    assert result["status"] == "needs_input"
    assert "create" in result["result"]["message"].lower()
    assert "update" in result["result"]["message"].lower()
    issue_stub.assert_not_awaited()
    assert conversation_store.get_pending("u1:acme:-") is None


@pytest.mark.asyncio
async def test_low_confidence_read_still_executes() -> None:
    """Reads are non-destructive — low confidence does NOT block them."""
    issue_stub = AsyncMock(return_value={"result": {"message": "issue-done"}})
    low = IntentResult(intent=Intent.QUERY_ISSUES, confidence=0.3, entities={})
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=low)),
        patch("agents.issue_agent.run", new=issue_stub),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="maybe show stuff", **BASE)
    assert result["status"] == "executed"
    issue_stub.assert_awaited_once()


# --- rate-limit handling (item 22) ------------------------------------------


class _FakeRateLimit(Exception):
    """Mimics groq/openai RateLimitError — exposes status_code == 429."""
    status_code = 429


@pytest.mark.asyncio
async def test_rate_limit_during_classify_returns_friendly_message() -> None:
    with (
        patch("agents.supervisor.classify", new=AsyncMock(side_effect=_FakeRateLimit("429"))),
        patch("agents.issue_agent.run", new=AsyncMock()),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        # A message the pre-router can't classify, so it reaches the LLM call.
        result = await run(message="what should I focus on today", **BASE)
    msg = result["result"]["message"].lower()
    assert "moment" in msg
    assert "rate-limited" in msg or "rate limited" in msg
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_rate_limit_during_read_dispatch_returns_friendly_message() -> None:
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(Intent.QUERY_ISSUES))),
        patch("agents.issue_agent.run", new=AsyncMock(side_effect=_FakeRateLimit("rate limit exceeded"))),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="show me issues", **BASE)
    assert "moment" in result["result"]["message"].lower()
    assert "rate" in result["result"]["message"].lower()


@pytest.mark.asyncio
async def test_non_rate_limit_error_during_read_uses_generic_message() -> None:
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=_intent(Intent.QUERY_ISSUES))),
        patch("agents.issue_agent.run", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        result = await run(message="show me issues", **BASE)
    msg = result["result"]["message"].lower()
    assert "unexpected error" in msg
    assert "rate" not in msg


def test_is_rate_limit_error_detects_wrapped_429() -> None:
    from agents.supervisor import _is_rate_limit_error

    inner = _FakeRateLimit("429 too many requests")
    wrapper = RuntimeError("llm call failed")
    wrapper.__cause__ = inner
    assert _is_rate_limit_error(wrapper) is True
    assert _is_rate_limit_error(RuntimeError("some other error")) is False


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
