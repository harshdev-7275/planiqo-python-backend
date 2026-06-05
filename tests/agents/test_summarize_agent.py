import pytest
from unittest.mock import AsyncMock, patch

from agents.summarize_agent import _format_sprint_summary, run
from models.intents import Intent, IntentResult

_STATUSES = [
    {"id": "s-todo", "name": "Todo", "category": "todo"},
    {"id": "s-prog", "name": "In Progress", "category": "in_progress"},
    {"id": "s-done", "name": "Done", "category": "done"},
]

_SPRINT = {
    "id": "sp1", "name": "Sprint 2", "status": "active",
    "startDate": "2026-05-15T00:00:00Z", "endDate": "2026-05-28T00:00:00Z",
}


def _issue(number, status_id, priority, sprint_id="sp1", completed_at=None):
    return {
        "number": number, "title": f"Issue {number}", "statusId": status_id,
        "priority": priority, "sprintId": sprint_id, "completedAt": completed_at,
    }


def _state(entities, project_id="proj-1"):
    return {
        "org_slug": "acme",
        "project_id": project_id,
        "intent": IntentResult(intent=Intent.SUMMARIZE, confidence=1.0, entities=entities),
        "messages": [],
    }


# --- pure formatter -------------------------------------------------------

def test_format_includes_name_and_dates():
    out = _format_sprint_summary(_SPRINT, [_issue(1, "s-todo", "high")], _STATUSES)
    assert "Sprint 2" in out
    assert "2026-05-15" in out


def test_format_progress_counts_done_by_category():
    issues = [_issue(1, "s-done", "low"), _issue(2, "s-done", "low"), _issue(3, "s-todo", "low")]
    out = _format_sprint_summary(_SPRINT, issues, _STATUSES)
    assert "2 of 3" in out


def test_format_status_breakdown():
    issues = [_issue(1, "s-todo", "low"), _issue(2, "s-todo", "low"), _issue(3, "s-prog", "low")]
    out = _format_sprint_summary(_SPRINT, issues, _STATUSES)
    assert "Todo 2" in out
    assert "In Progress 1" in out


def test_format_priority_breakdown():
    issues = [_issue(1, "s-todo", "critical"), _issue(2, "s-todo", "high"), _issue(3, "s-todo", "high")]
    out = _format_sprint_summary(_SPRINT, issues, _STATUSES)
    assert "critical 1" in out
    assert "high 2" in out


def test_format_empty_sprint():
    out = _format_sprint_summary(_SPRINT, [], _STATUSES)
    assert "no issues" in out.lower()


def test_format_done_falls_back_to_completedAt_when_status_unmapped():
    issues = [_issue(1, "s-unknown", "low", completed_at="2026-05-20T00:00:00Z")]
    out = _format_sprint_summary(_SPRINT, issues, _STATUSES)
    assert "1 of 1" in out


# --- run() with mocked Node API ------------------------------------------

@pytest.mark.asyncio
async def test_run_summarizes_named_sprint_and_filters_by_sprint_id():
    with patch("agents.summarize_agent.node_api_client") as api:
        api.get_sprints = AsyncMock(return_value=[_SPRINT])
        api.get_issues = AsyncMock(return_value=[
            _issue(1, "s-todo", "high"),
            _issue(2, "s-done", "low"),
            _issue(9, "s-todo", "low", sprint_id="OTHER"),  # different sprint — excluded
        ])
        api.get_statuses = AsyncMock(return_value=_STATUSES)
        api.get_active_sprint = AsyncMock(return_value=None)
        out = await run(_state({"sprint": "Sprint 2"}))
    msg = out["result"]["message"]
    assert "Sprint 2" in msg
    assert "1 of 2" in msg  # only the 2 sprint-sp1 issues counted, 1 done


@pytest.mark.asyncio
async def test_run_falls_back_to_active_sprint_when_no_name_given():
    with patch("agents.summarize_agent.node_api_client") as api:
        api.get_sprints = AsyncMock(return_value=[_SPRINT])
        api.get_active_sprint = AsyncMock(return_value=_SPRINT)
        api.get_issues = AsyncMock(return_value=[_issue(1, "s-todo", "high")])
        api.get_statuses = AsyncMock(return_value=_STATUSES)
        out = await run(_state({}))
    assert "Sprint 2" in out["result"]["message"]
    api.get_active_sprint.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_reports_when_no_sprint_found():
    with patch("agents.summarize_agent.node_api_client") as api:
        api.get_sprints = AsyncMock(return_value=[_SPRINT])
        api.get_active_sprint = AsyncMock(return_value=None)
        out = await run(_state({"sprint": "Nonexistent"}))
    assert "couldn't find" in out["result"]["message"].lower()


@pytest.mark.asyncio
async def test_run_requires_project_context():
    out = await run(_state({"sprint": "Sprint 2"}, project_id=None))
    assert "project" in out["result"]["message"].lower()


# --- graceful degradation (Sprint 3.x) -------------------------------------
#
# The stress test caught summarize returning 500 when the Node API was
# unreachable. The agent must NEVER crash the /chat endpoint — every
# node_api call here can fail (org doesn't exist, network blip, etc.) and
# the user must see a friendly message, not a 500.


@pytest.mark.asyncio
async def test_run_swallows_node_api_500_on_get_sprints() -> None:
    """A 500 from get_sprints must NOT propagate out of the agent — it
    returns a user-readable error message. Regression for stress-test
    query #13 ("summarize sprint 2" → http_500 when the stress org
    didn't exist)."""
    with patch("agents.summarize_agent.node_api_client") as api:
        api.get_sprints = AsyncMock(side_effect=Exception("500 from server"))
        out = await run(_state({"sprint": "Sprint 2"}))
    assert "couldn't" in out["result"]["message"].lower() or "server" in out["result"]["message"].lower()


@pytest.mark.asyncio
async def test_run_swallows_node_api_500_on_get_issues() -> None:
    with patch("agents.summarize_agent.node_api_client") as api:
        api.get_sprints = AsyncMock(return_value=[_SPRINT])
        api.get_issues = AsyncMock(side_effect=Exception("500 from server"))
        api.get_statuses = AsyncMock(return_value=_STATUSES)
        out = await run(_state({"sprint": "Sprint 2"}))
    assert "couldn't" in out["result"]["message"].lower() or "server" in out["result"]["message"].lower()
