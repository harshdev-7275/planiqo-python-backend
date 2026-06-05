"""Tests for the deterministic write executor (Sprint 1.1).

Confirmed writes must execute from validated entities — NOT through a
ReAct LLM loop. This module asserts the architectural invariant:
  - the executor posts/patches the EXACT validated entities (no re-derivation),
  - the executor does NOT re-resolve names the validation gate already resolved,
  - the supervisor routes confirmed writes to the executor (not the ReAct agent).

The test layer mocks at the HTTP boundary (the NodeAPIClient) because the
executor is a stateless, single-pass HTTP wrapper — there is no LLM to mock
(AIService.md §LAYER RULES: tools are thin HTTP wrappers only — no logic).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# --- create_issue -----------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_create_issue_calls_post_with_validated_entities() -> None:
    """The executor posts EXACTLY the validated entities — no re-derivation."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(
        return_value={"id": "status-todo", "name": "Todo", "isDefault": True}
    )
    api.post = AsyncMock(
        return_value={"id": "issue-uuid-1", "number": 42, "title": "Login broken"}
    )

    result = await execute_create_issue(
        entities={
            "title": "Login broken",
            "type": "bug",
            "priority": "high",
            "assignee_id": "u-123",
        },
        api=api,
        org_slug="acme",
        project_id="proj-1",
        user_id="u-1",
    )

    # 1. Path is correct.
    assert api.post.call_args.args[0] == "/orgs/acme/projects/proj-1/issues"
    # 2. Body contains every validated entity (camelCase mapping for the backend).
    body = api.post.call_args.args[1]
    assert body["title"] == "Login broken"
    assert body["type"] == "bug"
    assert body["priority"] == "high"
    assert body["assigneeId"] == "u-123"
    # 3. Default status was resolved and added.
    assert body["statusId"] == "status-todo"
    api.get_default_status.assert_awaited_once_with("acme", "proj-1")
    # 4. Acting user forwarded for authorization.
    assert api.post.call_args.kwargs["user_id"] == "u-1"
    # 5. Result message has the new issue number and title.
    assert result["result"]["message"] == "Created #42: Login broken"


@pytest.mark.asyncio
async def test_execute_create_issue_omits_assignee_when_not_provided() -> None:
    """The body must NOT carry an assigneeId key when the user did not name
    one. Mirrors the existing tool behaviour and prevents accidental null
    overwrites on the backend."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    api.post = AsyncMock(return_value={"number": 1, "title": "t"})

    await execute_create_issue(
        entities={"title": "t", "type": "task", "priority": "medium"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    body = api.post.call_args.args[1]
    assert "assigneeId" not in body
    assert "sprintId" not in body


@pytest.mark.asyncio
async def test_execute_create_issue_uses_explicit_status_when_provided() -> None:
    """``status_id`` in entities bypasses the default-status lookup."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    # NO get_default_status mock — if the executor calls it, the test fails
    # with AttributeError (proves the explicit status bypassed the lookup).
    api.post = AsyncMock(return_value={"number": 1, "title": "t"})

    await execute_create_issue(
        entities={"title": "t", "status_id": "status-explicit"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    body = api.post.call_args.args[1]
    assert body["statusId"] == "status-explicit"


@pytest.mark.asyncio
async def test_execute_create_issue_returns_error_when_no_statuses() -> None:
    """A project with no statuses cannot create an issue — surface the
    error rather than crashing."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(return_value=None)
    api.post = AsyncMock(return_value={"number": 1, "title": "t"})

    result = await execute_create_issue(
        entities={"title": "t"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "no statuses" in result["result"]["message"].lower()
    api.post.assert_not_called()


@pytest.mark.asyncio
async def test_execute_create_issue_does_not_reresolve_assignee_name() -> None:
    """THE HEADLINE DRIFT TEST. The executor must accept the already-resolved
    ``assignee_id`` and NEVER call a name-resolution helper again. This is
    what makes preview == execution: validation resolved Alice→u_123 once,
    confirmation must use u_123, not re-derive from "Alice"."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    api.post = AsyncMock(return_value={"number": 9, "title": "Login broken"})
    # If the executor reaches for any name-resolution helper, this mock's
    # call_count will become non-zero and the test fails.
    api.find_user_by_name = MagicMock()

    await execute_create_issue(
        entities={"title": "Login broken", "type": "bug", "priority": "high", "assignee_id": "u-123"},
        api=api, org_slug="acme", project_id="proj-1", user_id="u-1",
    )
    body = api.post.call_args.args[1]
    assert body["assigneeId"] == "u-123"
    # The executor MUST NOT have re-resolved the assignee name.
    api.find_user_by_name.assert_not_called()
    # The executor only called the two expected methods.
    called = {call[0] for call in api.method_calls}
    assert called == {"get_default_status", "post"}


@pytest.mark.asyncio
async def test_execute_create_issue_returns_error_string_on_post_failure() -> None:
    """A failed POST must surface a user-readable error message, never raise."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(return_value={"id": "s1", "name": "Todo", "isDefault": True})
    api.post = AsyncMock(side_effect=Exception("403 forbidden"))

    result = await execute_create_issue(
        entities={"title": "t"},
        api=api, org_slug="acme", project_id="proj-1", user_id="u-1",
    )
    assert "Failed" in result["result"]["message"]
    assert "403" in result["result"]["message"]


@pytest.mark.asyncio
async def test_execute_create_issue_without_title_returns_error() -> None:
    """A title-less create must NOT POST — surface the error."""
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.post = AsyncMock(return_value={})

    result = await execute_create_issue(
        entities={"type": "task", "priority": "medium"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "no title" in result["result"]["message"].lower()
    api.post.assert_not_called()


# --- update_issue -----------------------------------------------------------


_UPD_ISSUES = [
    {"number": 5, "id": "issue-uuid-5", "title": "Login broken", "priority": "low"},
    {"number": 6, "id": "issue-uuid-6", "title": "Dark mode", "priority": "medium"},
]


@pytest.mark.asyncio
async def test_execute_update_issue_resolves_number_then_patches() -> None:
    """The number→UUID resolution is the executor's only LLM-free indirection.
    Confirm the resolved UUID (not the number) is what hits the PATCH path."""
    from agents.write_executor import execute_update_issue

    api = MagicMock()
    api.get_issues = AsyncMock(return_value=_UPD_ISSUES)
    api.patch = AsyncMock(return_value={"number": 5, "title": "Login broken"})

    result = await execute_update_issue(
        entities={"issue": 5, "priority": "high"},
        api=api, org_slug="acme", project_id="proj-1", user_id="u-1",
    )
    assert api.patch.call_args.args[0] == "/orgs/acme/projects/proj-1/issues/issue-uuid-5"
    assert api.patch.call_args.args[1] == {"priority": "high"}
    assert result["result"]["message"] == "Updated #5."


@pytest.mark.asyncio
async def test_execute_update_issue_accepts_number_key() -> None:
    """The supervisor's normalizer/validation may emit either ``issue`` or
    ``number`` — accept both."""
    from agents.write_executor import execute_update_issue

    api = MagicMock()
    api.get_issues = AsyncMock(return_value=_UPD_ISSUES)
    api.patch = AsyncMock(return_value={})

    await execute_update_issue(
        entities={"number": 6, "assignee_id": "u-9"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert api.patch.call_args.args[0] == "/orgs/acme/projects/proj-1/issues/issue-uuid-6"
    assert api.patch.call_args.args[1] == {"assigneeId": "u-9"}


@pytest.mark.asyncio
async def test_execute_update_issue_not_found_does_not_patch() -> None:
    """A number that doesn't match any issue must NOT result in a PATCH —
    that would 404 on the backend and produce a confusing error."""
    from agents.write_executor import execute_update_issue

    api = MagicMock()
    api.get_issues = AsyncMock(return_value=_UPD_ISSUES)
    api.patch = AsyncMock(return_value={})

    result = await execute_update_issue(
        entities={"issue": 99, "priority": "high"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "#99" in result["result"]["message"] and "not found" in result["result"]["message"].lower()
    api.patch.assert_not_called()


@pytest.mark.asyncio
async def test_execute_update_issue_with_no_fields_returns_nothing_to_update() -> None:
    """An empty update must not PATCH — return the same friendly message
    the existing UpdateIssueTool returns."""
    from agents.write_executor import execute_update_issue

    api = MagicMock()
    api.get_issues = AsyncMock(return_value=_UPD_ISSUES)
    api.patch = AsyncMock(return_value={})

    result = await execute_update_issue(
        entities={"issue": 5},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "nothing" in result["result"]["message"].lower()
    api.get_issues.assert_not_called()
    api.patch.assert_not_called()


@pytest.mark.asyncio
async def test_execute_update_issue_without_issue_number_returns_error() -> None:
    """A write with no target is a client bug — surface, don't POST."""
    from agents.write_executor import execute_update_issue

    api = MagicMock()
    api.patch = AsyncMock(return_value={})

    result = await execute_update_issue(
        entities={"priority": "high"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "no issue number" in result["result"]["message"].lower()
    api.patch.assert_not_called()


# --- create_sprint ----------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_create_sprint_posts_name_and_goal() -> None:
    """The sprint name is the only required field; goal is optional."""
    from agents.write_executor import execute_create_sprint

    api = MagicMock()
    api.post = AsyncMock(return_value={"name": "Sprint 1"})

    result = await execute_create_sprint(
        entities={"name": "Sprint 1", "goal": "Ship login fix"},
        api=api, org_slug="acme", project_id="proj-1", user_id="u-1",
    )
    assert api.post.call_args.args[0] == "/orgs/acme/projects/proj-1/sprints"
    assert api.post.call_args.args[1] == {"name": "Sprint 1", "goal": "Ship login fix"}
    assert result["result"]["message"] == "Created sprint 'Sprint 1'."


@pytest.mark.asyncio
async def test_execute_create_sprint_omits_goal_when_absent() -> None:
    """No goal → body must not carry a goal key (no accidental empty strings)."""
    from agents.write_executor import execute_create_sprint

    api = MagicMock()
    api.post = AsyncMock(return_value={"name": "Sprint 1"})

    await execute_create_sprint(
        entities={"name": "Sprint 1"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert api.post.call_args.args[1] == {"name": "Sprint 1"}


@pytest.mark.asyncio
async def test_execute_create_sprint_accepts_title_alias() -> None:
    """Some LLM outputs use ``title`` instead of ``name`` — accept both."""
    from agents.write_executor import execute_create_sprint

    api = MagicMock()
    api.post = AsyncMock(return_value={"name": "Sprint 1"})

    await execute_create_sprint(
        entities={"title": "Sprint 1"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert api.post.call_args.args[1] == {"name": "Sprint 1"}


@pytest.mark.asyncio
async def test_execute_create_sprint_without_name_returns_error() -> None:
    """No name → no POST."""
    from agents.write_executor import execute_create_sprint

    api = MagicMock()
    api.post = AsyncMock(return_value={})

    result = await execute_create_sprint(
        entities={"goal": "Ship it"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "no name" in result["result"]["message"].lower()
    api.post.assert_not_called()


@pytest.mark.asyncio
async def test_execute_create_issue_handles_status_lookup_500() -> None:
    """Sprint 3.x fix: a 500 from ``get_default_status`` must NOT propagate
    out of the executor and crash the /chat endpoint. The user sees a
    friendly error string, and the POST is never attempted.

    Regression: stress-test query #21 ("yes" follow-up to a CREATE_ISSUE)
    500'd the entire /chat endpoint because the node-api was unreachable.
    """
    from agents.write_executor import execute_create_issue

    api = MagicMock()
    api.get_default_status = AsyncMock(side_effect=Exception("500 from server"))
    api.post = AsyncMock(return_value={"number": 1, "title": "t"})

    result = await execute_create_issue(
        entities={"title": "t"},
        api=api, org_slug="acme", project_id="proj-1", user_id="u-1",
    )
    assert "Failed" in result["result"]["message"]
    assert "500" in result["result"]["message"]
    # The POST was never attempted — the user would be told to try again,
    # and the executor didn't create a partial issue.
    api.post.assert_not_called()


@pytest.mark.asyncio
async def test_execute_create_sprint_returns_error_string_on_failure() -> None:
    from agents.write_executor import execute_create_sprint

    api = MagicMock()
    api.post = AsyncMock(side_effect=Exception("timeout"))

    result = await execute_create_sprint(
        entities={"name": "Sprint 1"},
        api=api, org_slug="acme", project_id="proj-1", user_id=None,
    )
    assert "Failed" in result["result"]["message"]
    assert "timeout" in result["result"]["message"]
