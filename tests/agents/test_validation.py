"""Pre-flight validation: ensure write intents are rejected BEFORE the
proposal when entities the user mentioned don't resolve.

These cover the UX fix where the bot used to show a confident "yes"-able
preview, then fail at execution with "I couldn't find Alice" — leaving
the user having typed a wasted confirmation.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from memory.store import conversation_store
from metering.usage import usage_store
from models.intents import Intent, IntentResult


BASE = {"user_id": "u1", "org_slug": "acme", "project_id": "proj-1"}


@pytest.fixture(autouse=True)
def _isolate_stores() -> Any:
    conversation_store.reset_all()
    usage_store.reset_all()
    yield
    conversation_store.reset_all()
    usage_store.reset_all()


async def _run_with_classify(
    message: str, classify_result: IntentResult, project_id: str | None = "proj-1"
) -> dict:
    """Run the supervisor end-to-end with classify + agents mocked, but
    REAL validation. Lets us assert validation behavior in isolation."""
    with (
        patch("agents.supervisor.classify", new=AsyncMock(return_value=classify_result)),
        patch("agents.issue_agent.run", new=AsyncMock(return_value={"result": {"message": "should-not-run"}})),
        patch("agents.sprint_agent.run", new=AsyncMock()),
    ):
        from agents.supervisor import run
        return await run(
            message=message,
            user_id="u1",
            org_slug="acme",
            project_id=project_id,
        )


# --- _looks_like_id + _coerce helpers ---------------------------------------


def test_looks_like_id_true_for_uuid() -> None:
    from agents.supervisor import _looks_like_id

    assert _looks_like_id("660c528a-2259-4f49-923d-73ef3cc0c6da") is True
    assert _looks_like_id("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE") is True


def test_looks_like_id_false_for_name() -> None:
    from agents.supervisor import _looks_like_id

    assert _looks_like_id("Alice") is False
    assert _looks_like_id("Sprint 23") is False
    assert _looks_like_id("") is False
    assert _looks_like_id(None) is False


# --- assignee validation -----------------------------------------------------


@pytest.mark.asyncio
async def test_create_issue_with_unknown_assignee_returns_validation_failed() -> None:
    """The headline fix: 'Alice' is not in the project → fail at the gate."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    with (
        patch("agents.supervisor.find_user_by_name", new=AsyncMock(return_value=[])),
        patch(
            "agents.supervisor.node_api_client.get_project_members",
            new=AsyncMock(return_value=[
                {"userId": "u1", "name": "Harsh Singh"},
                {"userId": "u2", "name": "Sara Patel"},
            ]),
        ),
    ):
        result = await _run_with_classify("create a bug, assign to Alice", intent)

    assert result["status"] == "validation_failed"
    assert "No team member named 'Alice'" in result["result"]["message"]
    # The error must list real members so the user can pick one.
    assert "Harsh Singh" in result["result"]["message"]
    assert "Sara Patel" in result["result"]["message"]


@pytest.mark.asyncio
async def test_create_issue_with_known_assignee_proceeds_to_proposal() -> None:
    """If the name resolves, validation passes and the normal preview is shown."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    with (
        patch(
            "agents.supervisor.find_user_by_name",
            new=AsyncMock(return_value=[{"id": "u-alice", "name": "Alice"}]),
        ),
        patch(
            "agents.supervisor.node_api_client.get_project_members",
            new=AsyncMock(return_value=[{"userId": "u-alice", "name": "Alice"}]),
        ),
    ):
        result = await _run_with_classify("create a bug, assign to Alice", intent)

    assert result["status"] == "awaiting_confirmation"
    assert "Alice" in result["result"]["message"]


@pytest.mark.asyncio
async def test_validation_skipped_when_assignee_is_a_uuid() -> None:
    """If the LLM already resolved the assignee to a UUID, no name check needed."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={
            "title": "x", "type": "bug",
            "assignee_id": "660c528a-2259-4f49-923d-73ef3cc0c6da",
        },
    )
    with (
        patch(
            "agents.supervisor.find_user_by_name",
            new=AsyncMock(return_value=[]),  # would fail if called
        ) as find_mock,
    ):
        result = await _run_with_classify("create a bug", intent)
    find_mock.assert_not_awaited()
    assert result["status"] == "awaiting_confirmation"


@pytest.mark.asyncio
async def test_validation_skipped_when_neo4j_unreachable() -> None:
    """Upstream flakiness must NOT block the user. If Neo4j errors out,
    fall through to the normal proposal — the post-confirm error path
    still surfaces the issue."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    with (
        patch(
            "agents.supervisor.find_user_by_name",
            new=AsyncMock(side_effect=Exception("Connection lost")),
        ),
    ):
        result = await _run_with_classify("create a bug, assign to Alice", intent)

    # Should NOT have failed validation — fell through to proposal.
    assert result["status"] == "awaiting_confirmation"


# --- sprint validation ------------------------------------------------------


@pytest.mark.asyncio
async def test_create_issue_with_unknown_sprint_returns_validation_failed() -> None:
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "sprint": "Sprint 99"},
    )
    with (
        patch(
            "agents.supervisor.node_api_client.get_sprints",
            new=AsyncMock(return_value=[
                {"id": "s1", "name": "Sprint 1"},
                {"id": "s2", "name": "Sprint 2"},
            ]),
        ),
    ):
        result = await _run_with_classify("create a bug in Sprint 99", intent)

    assert result["status"] == "validation_failed"
    assert "No sprint matching 'Sprint 99'" in result["result"]["message"]
    # Lists actual available sprints.
    assert "Sprint 1" in result["result"]["message"]
    assert "Sprint 2" in result["result"]["message"]


@pytest.mark.asyncio
async def test_create_issue_with_known_sprint_proceeds() -> None:
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "sprint": "Sprint 2"},
    )
    with (
        patch(
            "agents.supervisor.node_api_client.get_sprints",
            new=AsyncMock(return_value=[
                {"id": "s1", "name": "Sprint 1"},
                {"id": "s2", "name": "Sprint 2"},
            ]),
        ),
    ):
        result = await _run_with_classify("create a bug in Sprint 2", intent)

    assert result["status"] == "awaiting_confirmation"


# --- update_issue validation ------------------------------------------------


@pytest.mark.asyncio
async def test_update_issue_with_unknown_number_returns_validation_failed() -> None:
    intent = IntentResult(
        intent=Intent.UPDATE_ISSUE, confidence=0.95,
        entities={"issue": "999", "status": "done"},
    )
    with (
        patch(
            "agents.supervisor.node_api_client.get_issues",
            new=AsyncMock(return_value=[
                {"number": 1, "title": "Foo"},
                {"number": 2, "title": "Bar"},
            ]),
        ),
    ):
        result = await _run_with_classify("mark issue #999 as done", intent)

    assert result["status"] == "validation_failed"
    assert "Issue #999 not found" in result["result"]["message"]


@pytest.mark.asyncio
async def test_update_issue_with_known_number_proceeds() -> None:
    intent = IntentResult(
        intent=Intent.UPDATE_ISSUE, confidence=0.95,
        entities={"issue": "1", "status": "done"},
    )
    with (
        patch(
            "agents.supervisor.node_api_client.get_issues",
            new=AsyncMock(return_value=[{"number": 1, "title": "Foo"}]),
        ),
    ):
        result = await _run_with_classify("mark issue #1 as done", intent)

    assert result["status"] == "awaiting_confirmation"


@pytest.mark.asyncio
async def test_update_issue_combines_issue_and_assignee_failures() -> None:
    intent = IntentResult(
        intent=Intent.UPDATE_ISSUE, confidence=0.95,
        entities={"issue": "99", "assignee": "Bob"},
    )
    with (
        patch(
            "agents.supervisor.node_api_client.get_issues",
            new=AsyncMock(return_value=[{"number": 1, "title": "Foo"}]),
        ),
        patch(
            "agents.supervisor.find_user_by_name",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "agents.supervisor.node_api_client.get_project_members",
            new=AsyncMock(return_value=[{"userId": "u1", "name": "Harsh"}]),
        ),
    ):
        result = await _run_with_classify("reassign #99 to Bob", intent)

    msg = result["result"]["message"]
    assert result["status"] == "validation_failed"
    assert "Issue #99 not found" in msg
    assert "No team member named 'Bob'" in msg


# --- skips validation in the right cases ------------------------------------


@pytest.mark.asyncio
async def test_validation_skipped_when_no_project_id() -> None:
    """Without a project_id we have no context to validate against. The
    request should fall through to the normal proposal flow (existing
    behavior)."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    result = await _run_with_classify(
        "create a bug, assign to Alice", intent, project_id=None
    )
    assert result["status"] == "awaiting_confirmation"


@pytest.mark.asyncio
async def test_validation_does_not_run_for_read_intents() -> None:
    """Reads (QUERY_ISSUES, etc.) don't propose, so no need to validate."""
    intent = IntentResult(
        intent=Intent.QUERY_ISSUES, confidence=0.95,
        entities={"assignee": "Alice"},  # even if name is in entities
    )
    with (
        patch(
            "agents.supervisor.find_user_by_name",
            new=AsyncMock(side_effect=AssertionError("should not be called")),
        ),
    ):
        result = await _run_with_classify("show me Alice's issues", intent)
    assert result["status"] == "executed"


@pytest.mark.asyncio
async def test_validation_does_not_set_pending_when_failed() -> None:
    """If validation fails, the thread must NOT have a pending action —
    otherwise the next 'yes' (perhaps from a stale context) would route
    to a non-existent proposal."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    with (
        patch("agents.supervisor.find_user_by_name", new=AsyncMock(return_value=[])),
        patch(
            "agents.supervisor.node_api_client.get_project_members",
            new=AsyncMock(return_value=[{"userId": "u1", "name": "Harsh"}]),
        ),
    ):
        await _run_with_classify("create a bug, assign to Alice", intent)

    assert conversation_store.get_pending("u1:acme:proj-1") is None


@pytest.mark.asyncio
async def test_validation_failed_response_includes_tokens_used() -> None:
    """Tokens were spent on the classify call; report the cost even when
    validation fails."""
    intent = IntentResult(
        intent=Intent.CREATE_ISSUE, confidence=0.95,
        entities={"title": "x", "type": "bug", "assignee": "Alice"},
    )
    with (
        patch("agents.supervisor.find_user_by_name", new=AsyncMock(return_value=[])),
        patch(
            "agents.supervisor.node_api_client.get_project_members",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await _run_with_classify("create a bug, assign to Alice", intent)
    assert "tokens_used" in result
