"""Deterministic write executor for confirmed AI intents.

Replaces the ReAct LLM loop for confirmed writes (Sprint 1.1). The supervisor
stores the validated entities (after normalize + pre-flight validation) in
``conversation_store.pending``; on confirmation, this module posts / patches
them DIRECTLY to the Node.js API. No LLM, no re-derivation — preview == execution
by construction.

Why this exists
---------------
Before this module, a confirmed write re-invoked the ReAct ``issue_agent`` /
``sprint_agent`` with the original user message in the history. The LLM would
re-derive the assignee (sometimes a different "Alice"), the priority (sometimes
losing the bug→high convention), etc. — silently drifting from what the user
just confirmed. See ``claude/AIService.md`` for the layer rules this honours.

The ReAct agents stay for genuinely open-ended reads (``QUERY_ISSUES``,
``QUERY_SPRINT``, ``SUMMARIZE``) where the model chooses which tool to call.
Writes are now a one-line dispatch to a function.

Per AIService.md, the executor is a thin HTTP wrapper: zero business logic,
zero LLM, error strings on failure (never raises).
"""

from __future__ import annotations

from typing import Any

from loguru import logger


# --- helpers ----------------------------------------------------------------


async def _resolve_status_id(
    entities: dict[str, Any],
    api: Any,
    org_slug: str,
    project_id: str,
) -> tuple[str | None, str | None]:
    """Return ``(status_id, error_message)``.

    Uses ``entities['status_id']`` if set; otherwise asks the Node API for the
    project's default status. If the project has no statuses OR the API is
    unreachable, returns ``(None, error_message)`` so the caller can
    surface a user-readable error.

    Sprint 3.x fix: a node-api 500 here used to propagate out of the
    executor and crash the /chat endpoint (caught by the stress test
    on query #21, the "yes" follow-up to a CREATE_ISSUE).
    """
    if entities.get("status_id"):
        return entities["status_id"], None
    try:
        default = await api.get_default_status(org_slug, project_id)
    except Exception as e:
        logger.warning("executor: get_default_status failed: {}", e)
        return None, f"couldn't reach the server ({e})"
    if not default:
        return None, "no statuses configured for this project"
    return default["id"], None


# --- create_issue -----------------------------------------------------------


async def execute_create_issue(
    *,
    entities: dict[str, Any],
    api: Any,
    org_slug: str,
    project_id: str,
    user_id: str | None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create an issue from the validated entities. The function POSTs exactly
    the fields it was given — no name re-resolution, no priority re-derivation.

    Required key: ``title``.
    Optional keys: ``type`` (default ``task``), ``priority`` (default ``medium``),
    ``assignee_id`` (already a UUID — set by the supervisor's validation gate,
    NOT the display name), ``sprint_id`` (same), ``status_id`` (else default
    status is used).
    """
    title = entities.get("title")
    if not title:
        return {"result": {"message": "Cannot create issue: no title provided."}}

    status_id, err = await _resolve_status_id(entities, api, org_slug, project_id)
    if err:
        return {"result": {"message": f"Failed to create issue: {err}"}}

    body: dict[str, Any] = {
        "title": title,
        "type": entities.get("type") or "task",
        "priority": entities.get("priority") or "medium",
        "statusId": status_id,
    }
    if entities.get("assignee_id"):
        body["assigneeId"] = entities["assignee_id"]
    if entities.get("sprint_id"):
        body["sprintId"] = entities["sprint_id"]

    logger.info(
        "write_executor create_issue org={} project={} type={} priority={} assignee={}",
        org_slug, project_id, body["type"], body["priority"], body.get("assigneeId"),
    )

    try:
        result = await api.post(
            f"/orgs/{org_slug}/projects/{project_id}/issues",
            body,
            user_id=user_id,
            idempotency_key=idempotency_key,
        )
    except Exception as e:
        logger.error("write_executor create_issue failed: {}", e)
        return {"result": {"message": f"Failed to create issue: {e}"}}

    number = result.get("number", "?")
    created_title = result.get("title", title)
    return {"result": {"message": f"Created #{number}: {created_title}"}}


# --- update_issue -----------------------------------------------------------


async def execute_update_issue(
    *,
    entities: dict[str, Any],
    api: Any,
    org_slug: str,
    project_id: str,
    user_id: str | None,
) -> dict[str, Any]:
    """Update an issue's fields from the validated entities.

    Required key: ``issue`` or ``number`` (the issue's NUMBER, not UUID — the
    user always refers to issues by number). The function resolves the number
    to a UUID exactly once here.

    Optional keys: ``priority``, ``title``, ``assignee_id``, ``status_id``,
    ``sprint_id``. An empty update returns a friendly "nothing to update"
    message — matching the existing ``UpdateIssueTool`` behaviour.
    """
    issue_num = entities.get("issue") or entities.get("number")
    if issue_num is None:
        return {"result": {"message": "Cannot update issue: no issue number provided."}}

    body: dict[str, Any] = {}
    if entities.get("priority"):
        body["priority"] = entities["priority"]
    if entities.get("title"):
        body["title"] = entities["title"]
    if entities.get("assignee_id"):
        body["assigneeId"] = entities["assignee_id"]
    if entities.get("status_id"):
        body["statusId"] = entities["status_id"]
    if entities.get("sprint_id"):
        body["sprintId"] = entities["sprint_id"]

    if not body:
        return {
            "result": {
                "message": "Nothing to update — specify priority, title, or assignee."
            }
        }

    try:
        target_int = int(issue_num)
    except (TypeError, ValueError):
        return {"result": {"message": f"Invalid issue number: {issue_num!r}"}}

    try:
        issues = await api.get_issues(org_slug, project_id)
    except Exception as e:
        logger.error("write_executor update_issue list failed: {}", e)
        return {"result": {"message": f"Failed to list issues: {e}"}}

    match = next((i for i in issues if i.get("number") == target_int), None)
    if not match:
        return {"result": {"message": f"Issue #{target_int} not found in this project."}}

    logger.info(
        "write_executor update_issue org={} project={} #{} fields={}",
        org_slug, project_id, target_int, sorted(body.keys()),
    )
    try:
        await api.patch(
            f"/orgs/{org_slug}/projects/{project_id}/issues/{match['id']}",
            body,
            user_id=user_id,
        )
    except Exception as e:
        logger.error("write_executor update_issue failed: {}", e)
        return {"result": {"message": f"Failed to update issue: {e}"}}

    return {"result": {"message": f"Updated #{target_int}."}}


# --- create_sprint ----------------------------------------------------------


async def execute_create_sprint(
    *,
    entities: dict[str, Any],
    api: Any,
    org_slug: str,
    project_id: str,
    user_id: str | None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a sprint from the validated entities.

    Required key: ``name`` (or ``title`` — some LLM outputs use either).
    Optional key: ``goal``.
    """
    name = entities.get("name") or entities.get("title")
    if not name:
        return {"result": {"message": "Cannot create sprint: no name provided."}}

    body: dict[str, Any] = {"name": name}
    if entities.get("goal"):
        body["goal"] = entities["goal"]

    logger.info(
        "write_executor create_sprint org={} project={} name='{}'",
        org_slug, project_id, name,
    )
    try:
        result = await api.post(
            f"/orgs/{org_slug}/projects/{project_id}/sprints",
            body,
            user_id=user_id,
            idempotency_key=idempotency_key,
        )
    except Exception as e:
        logger.error("write_executor create_sprint failed: {}", e)
        return {"result": {"message": f"Failed to create sprint: {e}"}}

    created_name = result.get("name", name)
    return {"result": {"message": f"Created sprint '{created_name}'."}}
