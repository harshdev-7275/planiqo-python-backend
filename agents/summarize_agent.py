"""Sprint summary (SUMMARIZE intent).

Deliberately deterministic — a sprint summary is pure aggregation over data the
Node API already returns, so there is no LLM call (zero tokens, identical output
every time). Per AIService.md: don't spend a model on a task that doesn't reason.

The pure ``_format_sprint_summary`` does all the shaping and is unit-tested with
plain dicts; ``run`` is the thin I/O wrapper that fetches and resolves the sprint.
"""

from typing import Any, Literal

from langchain_core.callbacks import Callbacks
from loguru import logger

from agents.state import SupervisorState
from clients.node_api import node_api_client
from models.blocks import (
    Block,
    BreakdownBlock,
    BreakdownSegment,
    ProgressBlock,
    ProseBlock,
)

# Fixed display order for the priority line so output is deterministic.
_PRIORITY_ORDER = ("critical", "high", "medium", "low")

# Severity tone per priority for the breakdown bars (matches the render
# contract's tone vocabulary).
_PRIORITY_TONE: dict[str, Literal["neutral", "good", "warn", "bad"]] = {
    "critical": "bad",
    "high": "warn",
    "medium": "neutral",
    "low": "good",
}


def _date_only(value: str | None) -> str:
    """'2026-05-15T00:00:00Z' -> '2026-05-15'; passthrough for anything else."""
    if not value:
        return "no date"
    return value.split("T", 1)[0]


def _is_done(issue: dict[str, Any], status_by_id: dict[str, Any]) -> bool:
    """Done if the issue's status is in the 'done' category, or (when the status
    can't be mapped) it carries a completedAt timestamp."""
    status = status_by_id.get(issue.get("statusId") or "")
    if status and status.get("category") == "done":
        return True
    return bool(issue.get("completedAt"))


def _format_sprint_summary(
    sprint: dict[str, Any], issues: list[dict[str, Any]], statuses: list[dict[str, Any]]
) -> str:
    name = sprint.get("name") or "the sprint"
    state = sprint.get("status") or "?"
    span = f"{_date_only(sprint.get('startDate'))} → {_date_only(sprint.get('endDate'))}"
    header = f"Sprint '{name}' ({state}) — {span}"

    total = len(issues)
    if total == 0:
        return f"{header}\nThis sprint has no issues yet."

    status_by_id = {s["id"]: s for s in statuses}
    done = sum(1 for i in issues if _is_done(i, status_by_id))

    # Status breakdown in the project's own status order, then anything unmapped.
    status_lines: list[str] = []
    for s in statuses:
        count = sum(1 for i in issues if i.get("statusId") == s["id"])
        if count:
            status_lines.append(f"{s['name']} {count}")
    known_ids = {s["id"] for s in statuses}
    other = sum(1 for i in issues if i.get("statusId") not in known_ids)
    if other:
        status_lines.append(f"Other {other}")

    # Priority breakdown in fixed severity order.
    priority_lines = []
    for p in _PRIORITY_ORDER:
        count = sum(1 for i in issues if i.get("priority") == p)
        if count:
            priority_lines.append(f"{p} {count}")

    return (
        f"{header}\n"
        f"Progress: {done} of {total} done.\n"
        f"Status: {', '.join(status_lines)}\n"
        f"Priority: {', '.join(priority_lines)}"
    )


def _summary_blocks(
    sprint: dict[str, Any], issues: list[dict[str, Any]], statuses: list[dict[str, Any]]
) -> list[Block]:
    """Visual blocks for the sprint summary — the structured twin of
    ``_format_sprint_summary``. Deterministic, zero-LLM. Recomputes the same
    aggregates (kept separate so the battle-tested text formatter is untouched);
    the message string stays the text fallback, these blocks are what the UI
    renders.
    """
    name = sprint.get("name") or "the sprint"
    state = sprint.get("status") or "?"
    span = f"{_date_only(sprint.get('startDate'))} → {_date_only(sprint.get('endDate'))}"
    blocks: list[Block] = [ProseBlock(markdown=f"**Sprint '{name}'** ({state}) — {span}")]

    total = len(issues)
    if total == 0:
        blocks.append(ProseBlock(markdown="This sprint has no issues yet."))
        return blocks

    status_by_id = {s["id"]: s for s in statuses}
    done = sum(1 for i in issues if _is_done(i, status_by_id))
    blocks.append(ProgressBlock(kind="sprint", title="Progress", completed=done, total=total))

    # Status breakdown in the project's own status order, then anything unmapped.
    status_segments: list[BreakdownSegment] = []
    for s in statuses:
        count = sum(1 for i in issues if i.get("statusId") == s["id"])
        if count:
            status_segments.append(BreakdownSegment(label=s["name"], value=count))
    known_ids = {s["id"] for s in statuses}
    other = sum(1 for i in issues if i.get("statusId") not in known_ids)
    if other:
        status_segments.append(BreakdownSegment(label="Other", value=other))
    if status_segments:
        blocks.append(
            BreakdownBlock(title="Status", dimension="status", segments=status_segments)
        )

    # Priority breakdown in fixed severity order, tinted by severity.
    priority_segments: list[BreakdownSegment] = []
    for p in _PRIORITY_ORDER:
        count = sum(1 for i in issues if i.get("priority") == p)
        if count:
            priority_segments.append(
                BreakdownSegment(label=p, value=count, tone=_PRIORITY_TONE[p])
            )
    if priority_segments:
        blocks.append(
            BreakdownBlock(title="Priority", dimension="priority", segments=priority_segments)
        )

    return blocks


def _resolve_sprint(sprints: list[dict[str, Any]], query: str | None) -> dict[str, Any] | None:
    """Match a sprint by name: exact (case-insensitive) first, then substring.
    Returns None when there's no query or no match (caller falls back to active)."""
    if not query:
        return None
    q = query.strip().lower()
    for s in sprints:
        if (s.get("name") or "").strip().lower() == q:
            return s
    for s in sprints:
        if q in (s.get("name") or "").lower():
            return s
    return None


async def run(state: SupervisorState, callbacks: Callbacks = None) -> dict[str, Any]:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""
    if not project_id:
        return {"result": {"message": "I need a project to summarize a sprint. Open a project and try again."}}

    intent = state["intent"]
    entities = intent.entities if intent else {}
    query = (
        entities.get("sprint") or entities.get("sprint_id")
        or entities.get("name") or entities.get("title")
    )

    logger.info("summarize_agent org={} project={} query={}", org_slug, project_id, query)

    # Sprint 3.x: wrap every node_api call in try/except so a 500 (org
    # doesn't exist, network blip, transient outage) surfaces a friendly
    # user message instead of crashing the /chat endpoint. The stress test
    # caught this as a regression: query #13 ("summarize sprint 2") 500'd.
    try:
        sprints = await node_api_client.get_sprints(org_slug, project_id)
    except Exception as e:
        logger.warning("summarize_agent: get_sprints failed: {}", e)
        return {"result": {"message": (
            "I couldn't fetch the project's sprints right now — the server "
            "is unreachable. Please try again in a moment."
        )}}
    sprint = _resolve_sprint(sprints, query)
    if sprint is None:
        try:
            sprint = await node_api_client.get_active_sprint(org_slug, project_id)
        except Exception as e:
            logger.warning("summarize_agent: get_active_sprint failed: {}", e)
            return {"result": {"message": (
                "I couldn't find a sprint to summarize. Try naming one "
                "(e.g. 'summarize Sprint 2') or start a sprint so there's "
                "an active one."
            )}}
    if sprint is None:
        return {"result": {"message": (
            "I couldn't find a sprint to summarize. Try naming one (e.g. "
            "'summarize Sprint 2') or start a sprint so there's an active one."
        )}}

    try:
        issues = await node_api_client.get_issues(org_slug, project_id)
    except Exception as e:
        logger.warning("summarize_agent: get_issues failed: {}", e)
        return {"result": {"message": (
            "I found the sprint but couldn't fetch its issues — the server "
            "is unreachable. Please try again in a moment."
        )}}
    sprint_issues = [i for i in issues if i.get("sprintId") == sprint.get("id")]
    try:
        statuses = await node_api_client.get_statuses(org_slug, project_id)
    except Exception as e:
        logger.warning("summarize_agent: get_statuses failed: {}", e)
        return {"result": {"message": (
            "I found the sprint and its issues but couldn't fetch the "
            "project's statuses. The summary is unavailable until the "
            "server recovers."
        )}}

    summary = _format_sprint_summary(sprint, sprint_issues, statuses)
    blocks = _summary_blocks(sprint, sprint_issues, statuses)
    logger.info("summarize_agent summarized '{}' ({} issues)", sprint.get("name"), len(sprint_issues))
    return {"result": {"message": summary, "blocks": [b.model_dump() for b in blocks]}}
