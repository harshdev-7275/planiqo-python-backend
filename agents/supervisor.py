"""Conversation supervisor.

Adds two stateful concerns on top of intent routing:

1. **Memory** — per-thread windowed history (memory/store.py) is loaded on every
   turn and passed to the executing agent, and each turn is appended back.
2. **Write confirmation** — intents that mutate data are *proposed* (a preview is
   returned) and only executed after the user replies "yes". This keeps a single
   misread message from silently creating or changing work items.

Reads run immediately. The execution agents (issue/sprint) remain LangGraph
ReAct agents; this module dispatches to them directly.
"""

import re
import uuid
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

import agents.issue_agent as _issue_agent
import agents.member_agent as _member_agent
import agents.sprint_agent as _sprint_agent
import agents.summarize_agent as _summarize_agent
from agents.preview import (
    build_preview,
    effective_title,
    ground_create_issue_title,
    ground_sprint_name,
)
from agents.state import SupervisorState
from chains.intent import classify
from chains.normalizer import canonicalize_priority, normalize_entities
from chains.pre_router import pre_route
from chains.title_clean import clean_title
from chains.yes_no import is_affirmation, is_negation
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from config.settings import settings
from graph.queries import find_user_by_name
from memory.store import conversation_store
from metering.callback import UsageCallback
from metering.usage import RequestTokens, usage_store
from models.intents import Intent, IntentResult

# Intents that mutate data — proposed and confirmed before they run.
WRITE_INTENTS = {Intent.CREATE_ISSUE, Intent.UPDATE_ISSUE, Intent.CREATE_SPRINT}

# Below this classifier confidence, a WRITE intent is too uncertain to act on:
# guessing create-vs-update on an ambiguous message risks creating or changing
# the wrong thing. We ask the user to clarify instead. Reads are
# non-destructive and proceed even when uncertain. (plan 4.2)
MIN_WRITE_CONFIDENCE = 0.6

# Emit a per-org usage milestone at every Nth request so ops can see traffic
# patterns and per-tenant cost growth in the log stream without scraping the
# /admin/usage endpoint. ``0`` disables the log.
USAGE_MILESTONE_EVERY = 100

# Confirm/Cancel button affordances surfaced on an awaiting_confirmation
# response (plan 4.4, item 18). A Teams adapter renders these as Adaptive Card
# buttons; the dashboard ignores the field and lets the user type "yes"/"no".
# The button VALUES ("yes"/"no") are already recognised by _is_affirmation /
# _is_negation, so a button click and a typed reply travel the same code path —
# no separate Teams branch, no extra word-matching.
CONFIRM_ACTIONS = [
    {"title": "Confirm", "value": "yes"},
    {"title": "Cancel", "value": "no"},
]

_UNKNOWN_MESSAGE = (
    "I didn't understand that. Try asking about issues, sprints, or team members."
)

# Shown when the AI provider rate-limits us (HTTP 429). A clear, specific ask
# to retry beats the generic "unexpected error" — the user knows it's transient
# and that nothing is wrong with their request.
_RATE_LIMIT_MESSAGE = (
    "I'm being rate-limited by the AI provider right now. "
    "Please try again in a moment."
)

_GENERIC_ERROR_MESSAGE = (
    "I hit an unexpected error while handling that request. "
    "Please try again in a moment."
)

# Shown when a write intent is classified with low confidence — rather than
# guess and mutate the wrong thing, ask the user to restate.
_LOW_CONFIDENCE_WRITE_MESSAGE = (
    "I'm not sure I understood that correctly. Did you want to create a new "
    "item or update an existing one? Please rephrase with a bit more detail — "
    "for example \"create a bug: checkout fails on mobile\" or "
    "\"set issue 5 priority to high\"."
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if *exc* (or anything it wraps) is a provider 429 / rate-limit.

    LangChain wraps the provider exception, so we walk the ``__cause__`` /
    ``__context__`` chain. The reliable signals are a ``status_code == 429``
    attribute (groq + openai ``RateLimitError`` both expose it) and a class
    name containing 'ratelimit'; a string check is the last-resort fallback."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "status_code", None) == 429:
            return True
        if "ratelimit" in type(cur).__name__.lower():
            return True
        cur = cur.__cause__ or cur.__context__
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text


def _error_message(exc: BaseException) -> str:
    """The user-facing message for an exception on the LLM path — specific for
    a rate limit, generic otherwise."""
    return _RATE_LIMIT_MESSAGE if _is_rate_limit_error(exc) else _GENERIC_ERROR_MESSAGE

# Shown when a CREATE_ISSUE has no usable title (the LLM extracted none and the
# user's phrasing has nothing to clean into one). Asking is strictly better than
# proposing an issue literally titled "(untitled)" — the user can't meaningfully
# confirm a name they never gave.
_MISSING_TITLE_MESSAGE = (
    "What should I title this issue? For example: "
    "\"create a bug: login button does nothing on mobile\"."
)

def _thread_id(user_id: str, org_slug: str, project_id: str | None) -> str:
    return f"{user_id}:{org_slug}:{project_id or '-'}"


# Yes/no detection lives in chains.yes_no now. Re-exported under the old private
# names so existing imports/tests (``supervisor._is_affirmation``) keep working.
_is_affirmation = is_affirmation
_is_negation = is_negation


# --- routing -----------------------------------------------------------------


def _route_intent(intent_result: IntentResult) -> str:
    match intent_result.intent:
        case Intent.CREATE_ISSUE | Intent.QUERY_ISSUES | Intent.UPDATE_ISSUE:
            return "issue_agent"
        case Intent.QUERY_SPRINT | Intent.CREATE_SPRINT:
            return "sprint_agent"
        case Intent.QUERY_MEMBER:
            return "member_agent"
        case Intent.SUMMARIZE:
            return "summarize_agent"
        case Intent.TEAMS_CONTEXT:
            return "teams_agent"
        case _:
            return "handle_unknown"


# --- write confirmation preview ----------------------------------------------
#
# Preview construction + title/sprint grounding moved to agents.preview; title
# cleaning to chains.title_clean. Re-exported under the old private names so
# ``run`` (below) and any external imports keep working unchanged.
_clean_title = clean_title
_ground_create_issue_title = ground_create_issue_title
_ground_sprint_name = ground_sprint_name
_effective_title = effective_title
_preview = build_preview


# --- pre-flight entity validation -------------------------------------------
#
# Before showing the "I'll create X, confirm?" preview, verify that every
# entity the user mentioned by NAME actually resolves to a real record.
# Without this, the user confirms "yes" only to learn the assignee / sprint
# didn't exist — a frustrating two-step failure. With this, the error lands
# immediately and lists the valid options.
#
# As a side effect, validation also produces a ``resolved`` map of
# name-keys → ID-keys (e.g. ``assignee_id``/``sprint_id``). The supervisor
# persists this on the pending proposal so the write executor can post the
# *exact* validated IDs on confirmation — no name re-resolution at execute
# time, no preview/execute drift. This is the single-pass design: one
# ``find_user_by_name`` call serves both the error check and the ID stash.
#
# Resilience rule: if the upstream (Neo4j or Node) is unreachable, we LOG
# and SKIP validation rather than blocking the user on flaky infra. The
# post-confirmation error path still surfaces the problem.

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
# Cap on how many names we list in an error — past 10 it's just noise.
_MAX_LISTED = 10


def _looks_like_id(value: str | None) -> bool:
    return bool(value and _UUID_RE.match(value.strip()))


def _coerce_member_name(m: dict[str, Any]) -> str:
    return m.get("name") or m.get("userName") or m.get("userId") or "?"


def _coerce_sprint_name(s: dict[str, Any]) -> str:
    return s.get("name") or s.get("sprintName") or s.get("id") or "?"


async def _check_assignee_name(
    org_slug: str, project_id: str, name: str
) -> str | None:
    """Return an error message if ``name`` does not match any current project
    member, or None if it resolves (or we cannot tell because upstreams are
    down)."""
    if _looks_like_id(name):
        return None  # already an ID, no resolution needed
    try:
        matches = await find_user_by_name(neo4j_client, project_id, name)
    except Exception as e:  # Neo4j unreachable — don't block the user
        logger.warning("validation: Neo4j unreachable, skipping assignee check: {}", e)
        return None
    if matches:
        return None  # at least one match
    return await _build_assignee_error_message(org_slug, project_id, name)


async def _build_assignee_error_message(
    org_slug: str, project_id: str, name: str
) -> str | None:
    """Build the user-facing 'no team member named X' error and list real
    members so the user can pick. Extracted from ``_check_assignee_name`` so
    ``_validate_intent`` can do one ``find_user_by_name`` call and then ask
    for the error message separately when the name fails to resolve."""
    try:
        members = await node_api_client.get_project_members(org_slug, project_id)
    except Exception as e:
        logger.warning("validation: Node unreachable listing members: {}", e)
        members = []
    msg = f"No team member named '{name}' found in this project."
    if members:
        names = [_coerce_member_name(m) for m in members[:_MAX_LISTED]]
        # Put the valid options on their own line so the user sees the choices
        # at a glance rather than buried mid-sentence (item 10).
        msg += f"\nAvailable members: {', '.join(names)}."
        msg += "\nReply with one of those names (or the exact name)."
    else:
        msg += " There are no members in this project to assign to yet."
    return msg


async def _check_sprint_name(
    org_slug: str, project_id: str, name: str
) -> str | None:
    """Return an error message if ``name`` does not match any sprint in the
    project, or None if it resolves (or upstreams are down)."""
    if _looks_like_id(name):
        return None
    try:
        sprints = await node_api_client.get_sprints(org_slug, project_id)
    except Exception as e:
        logger.warning("validation: Node unreachable, skipping sprint check: {}", e)
        return None
    matches = [
        s for s in sprints
        if _match_sprint_name(name, _coerce_sprint_name(s))
    ]
    if matches:
        return None
    return _build_sprint_error_message(name, sprints)


def _build_sprint_error_message(
    name: str, sprints: list[dict[str, Any]]
) -> str | None:
    """Build the user-facing 'no sprint matching X' error with the list of
    real sprints. Extracted from ``_check_sprint_name`` for the same reason
    ``_build_assignee_error_message`` exists — single-pass validation."""
    names = [_coerce_sprint_name(s) for s in sprints[:_MAX_LISTED]]
    msg = f"No sprint matching '{name}' found in this project."
    if names:
        # Options on their own line — easier to scan than an inline list (item 10).
        msg += f"\nAvailable sprints: {', '.join(names)}."
        msg += "\nReply with one of those names."
    return msg


# Common sprint-name prefixes the user or the LLM may write that should
# not influence the match: "Sprint 1" and "sprin1" both reduce to "1" once
# the prefix is dropped, so "Sprint 1" resolves to a sprint actually named
# "sprin1" — but "Sprint 99" still does NOT match "Sprint 1" (the suffixes
# differ). Listed longest-first so "sprint " is tried before bare "sprint".
_SPRINT_PREFIXES: tuple[str, ...] = (
    "sprint ", "sprint-", "sprint_",
    "sprint",
    "sprin ", "sprin-", "sprin_",
    "sprin",
)


def _strip_sprint_prefix(s: str) -> str:
    for prefix in _SPRINT_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _match_sprint_name(query: str, candidate: str) -> bool:
    """True if ``query`` plausibly refers to ``candidate``.

    Three-strategy ladder — strict enough to reject "Sprint 99" vs "Sprint 1"
    (only the shared prefix matches) and lenient enough to accept
    "Sprint 1" vs "sprin1" (both reduce to "1" once the sprint prefix is
    dropped):

      1. exact equality (lowercased, trimmed)
      2. substring either way (handles "Sprint" → "Sprint Alpha")
      3. strip the sprint prefix from both and re-check (handles the
         "Sprint 1" → "sprin1" case; rejects "Sprint 99" → "Sprint 1" because
         the suffixes differ)
    """
    q = (query or "").strip().lower()
    c = (candidate or "").strip().lower()
    if not q or not c:
        return False
    if q == c:
        return True
    if q in c or c in q:
        return True
    qs = _strip_sprint_prefix(q).strip()
    cs = _strip_sprint_prefix(c).strip()
    if qs and cs and (qs == cs or qs in cs or cs in qs):
        return True
    return False


async def _check_issue_number(
    org_slug: str, project_id: str, raw: Any
) -> str | None:
    """Return an error if ``raw`` (e.g. ``42`` or ``"#42"``) does not match
    an existing issue in the project."""
    if raw is None:
        return None
    if isinstance(raw, int):
        target = raw
    else:
        text = str(raw).strip().lstrip("#")
        try:
            target = int(text)
        except (ValueError, AttributeError):
            return None  # not parseable — let the agent try
    try:
        issues = await node_api_client.get_issues(org_slug, project_id)
    except Exception as e:
        logger.warning("validation: Node unreachable, skipping issue-number check: {}", e)
        return None
    if any(i.get("number") == target for i in issues):
        return None
    return f"Issue #{target} not found in this project."


async def _validate_intent(
    intent_result: IntentResult,
    org_slug: str,
    project_id: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Pre-flight validation for write intents + side-effect ID resolution.

    Returns ``(error, resolved)``:
      * ``error`` — ``None`` if every referenced entity resolves; otherwise a
        user-facing message (e.g. "no team member named 'X'") that lists the
        valid options.
      * ``resolved`` — a side-car dict of name-keys → ID-keys for the
        entities the user mentioned by NAME (e.g. ``assignee_id``, ``sprint_id``).
        The supervisor persists this on the pending proposal so the write
        executor can POST the *exact* validated IDs on confirmation. One
        ``find_user_by_name`` call serves both the error check and the ID
        stash — no duplicate lookups on the request path.
    """
    if not project_id:
        return None, {}
    intent = intent_result.intent
    if intent not in WRITE_INTENTS:
        return None, {}
    entities = intent_result.entities
    errors: list[str] = []
    resolved: dict[str, Any] = {}

    if intent == Intent.CREATE_ISSUE or intent == Intent.UPDATE_ISSUE:
        if intent == Intent.UPDATE_ISSUE:
            issue_num = entities.get("issue") or entities.get("number")
            err = await _check_issue_number(org_slug, project_id, issue_num)
            if err:
                errors.append(err)

        # --- Assignee: resolve via Neo4j, fall back to user-facing error. ---
        assignee = entities.get("assignee") or entities.get("assignee_id")
        if assignee:
            if _looks_like_id(str(assignee)):
                resolved["assignee_id"] = str(assignee)
            else:
                rid = await _resolve_assignee_via_graph(project_id, str(assignee))
                if rid:
                    resolved["assignee_id"] = rid
                else:
                    err = await _check_assignee_name(org_slug, project_id, str(assignee))
                    if err:
                        errors.append(err)

        # --- Sprint: resolve via Node, fall back to user-facing error. ---
        sprint = entities.get("sprint") or entities.get("sprint_id")
        if sprint:
            if _looks_like_id(str(sprint)):
                resolved["sprint_id"] = str(sprint)
            else:
                rid = await _resolve_sprint_via_node(org_slug, project_id, str(sprint))
                if rid:
                    resolved["sprint_id"] = rid
                else:
                    err = await _check_sprint_name(org_slug, project_id, str(sprint))
                    if err:
                        errors.append(err)

    return ("\n\n".join(errors) if errors else None), resolved


# --- name → ID resolution helpers used by _validate_intent -----------------


async def _resolve_assignee_via_graph(project_id: str, name: str) -> str | None:
    """Return the project member's UUID for a name, or None if no match OR
    Neo4j is unreachable. The validation caller handles the error message in
    the failure case so the user still sees the list of valid members."""
    try:
        matches = await find_user_by_name(neo4j_client, project_id, name)
    except Exception as e:
        logger.warning("validation: Neo4j unreachable, skipping assignee resolve: {}", e)
        return None
    return matches[0]["id"] if matches else None


async def _resolve_sprint_via_node(
    org_slug: str, project_id: str, name: str
) -> str | None:
    """Return the sprint's UUID for a name, or None if no match OR Node is
    unreachable."""
    try:
        sprints = await node_api_client.get_sprints(org_slug, project_id)
    except Exception as e:
        logger.warning("validation: Node unreachable, skipping sprint resolve: {}", e)
        return None
    matches = [
        s for s in sprints if _match_sprint_name(name, _coerce_sprint_name(s))
    ]
    return matches[0]["id"] if matches else None


# --- execution ---------------------------------------------------------------


async def _execute(
    intent_result: IntentResult,
    messages: list[BaseMessage],
    user_id: str,
    org_slug: str,
    project_id: str | None,
    callbacks: Callbacks = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Dispatch a classified intent and return a result dict that always
    contains a 'message'.

    Sprint 1.1: CONFIRMED writes go to the deterministic
    ``write_executor`` (no LLM, no re-derivation). This is the architectural
    fix for the preview/execute drift — the validated entities (with
    pre-resolved IDs merged in) are posted exactly as the user saw them in
    the preview. The ReAct agents remain for genuinely open-ended reads.
    """
    intent = intent_result.intent

    # Confirmed-write fast path: skip the ReAct agent, call the executor.
    if intent in WRITE_INTENTS:
        from agents import write_executor  # local import to avoid a cycle

        entities = intent_result.entities
        if intent == Intent.CREATE_ISSUE:
            out = await write_executor.execute_create_issue(
                entities=entities,
                api=node_api_client,
                org_slug=org_slug,
                project_id=project_id or "",
                user_id=user_id,
                idempotency_key=idempotency_key,
            )
        elif intent == Intent.UPDATE_ISSUE:
            out = await write_executor.execute_update_issue(
                entities=entities,
                api=node_api_client,
                org_slug=org_slug,
                project_id=project_id or "",
                user_id=user_id,
            )
        else:  # CREATE_SPRINT
            out = await write_executor.execute_create_sprint(
                entities=entities,
                api=node_api_client,
                org_slug=org_slug,
                project_id=project_id or "",
                user_id=user_id,
                idempotency_key=idempotency_key,
            )
        # Executor returns the agent-shape ``{"result": {"message": str}}``;
        # unwrap to the inner ``{"message": str}`` the supervisor expects.
        unwrapped: dict[str, Any] = out.get("result", out)
        return unwrapped

    # Reads and any future intents: the ReAct agent chooses the tool.
    state: SupervisorState = {
        "messages": messages,
        "user_id": user_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "intent": intent_result,
        "result": None,
        "error": None,
        "next": "",
    }
    target = _route_intent(intent_result)
    logger.info("supervisor route intent={} -> {}", intent_result.intent.value, target)

    if target == "issue_agent":
        out = await _issue_agent.run(state, callbacks=callbacks)
    elif target == "sprint_agent":
        out = await _sprint_agent.run(state, callbacks=callbacks)
    elif target == "member_agent":
        out = await _member_agent.run(state, callbacks=callbacks)
    elif target == "summarize_agent":
        out = await _summarize_agent.run(state, callbacks=callbacks)
    elif target == "handle_unknown":
        return {"message": _UNKNOWN_MESSAGE}
    else:  # teams_agent — not built yet
        return {"message": f"{target} not yet implemented"}

    result = out.get("result")
    return result if result else {"message": "No response."}


# --- entry point -------------------------------------------------------------


async def run(
    message: str,
    user_id: str,
    org_slug: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    thread = _thread_id(user_id, org_slug, project_id)

    # Count every incoming request (including quota-blocked ones) so ops can
    # see total traffic, not just the requests that hit the LLM. Emits a
    # per-org milestone log line every USAGE_MILESTONE_EVERY requests.
    request_count = usage_store.inc_request(org_slug)
    if USAGE_MILESTONE_EVERY > 0 and request_count % USAGE_MILESTONE_EVERY == 0:
        logger.info(
            "usage_milestone org={} requests={} tokens={}",
            org_slug, request_count, usage_store.get(org_slug),
        )

    # Per-request token accumulator — the callback writes into it, we read
    # it once at the end and surface the number in the /chat response.
    request_tokens = RequestTokens()

    # Cost guard — block once a workspace is over its token quota (0 = disabled).
    if settings.ORG_TOKEN_QUOTA > 0 and usage_store.get(org_slug) >= settings.ORG_TOKEN_QUOTA:
        logger.warning(
            "supervisor quota exceeded org={} total={}", org_slug, usage_store.get(org_slug)
        )
        return {
            "intent": None,
            "result": {
                "message": "This workspace has reached its AI usage limit. "
                           "Please try again later or upgrade your plan."
            },
            "status": "quota_exceeded",
            "error": None,
            "tokens_used": 0,
        }

    usage_cb = UsageCallback(
        org_slug=org_slug, user_id=user_id, store=usage_store, request_tokens=request_tokens
    )
    callbacks: Callbacks = [usage_cb]

    history = conversation_store.history(thread, settings.MAX_HISTORY_TURNS)
    user_msg = HumanMessage(content=message)

    # 1) Resolve an outstanding confirmation before anything else. ``get_pending``
    # transparently drops a stale proposal so an absent user does not poison
    # the next unrelated message.
    pending = conversation_store.get_pending(thread, ttl_seconds=settings.PENDING_TTL_SECONDS)
    if pending is not None:
        if _is_affirmation(message):
            conversation_store.clear_pending(thread)
            # Merge the user-facing entities with the IDs the validation gate
            # already resolved (assignee_id, sprint_id). The write executor
            # reads the merged dict — it NEVER re-resolves names, so preview
            # == execution by construction.
            merged = {**pending.get("entities", {}), **pending.get("resolved", {})}
            intent_result = IntentResult(
                intent=Intent(pending["intent"]),
                confidence=1.0,
                entities=merged,
            )
            # Sprint 3.x: defense in depth — even though the executor and
            # the ReAct agents catch their own errors, an unexpected crash
            # (e.g. an LLM that throws instead of returning a string)
            # must not 500 the /chat endpoint. The user sees a friendly
            # message instead.
            try:
                result = await _execute(
                    intent_result, [], user_id, org_slug, project_id,
                    callbacks=callbacks,
                    # Reuse the key minted when the proposal was created, so a
                    # retried confirmation POSTs with the SAME Idempotency-Key
                    # and Node dedupes the create (item 17).
                    idempotency_key=pending.get("idempotency_key"),
                )
            except Exception as e:
                logger.error(
                    "supervisor confirmed intent crashed: {}", e, exc_info=True,
                )
                result = {
                    "message": (
                        "I couldn't complete that action — the system hit an "
                        "unexpected error. The change was NOT made. Please try "
                        "again or rephrase."
                    )
                }
            conversation_store.append(thread, user_msg, AIMessage(content=result["message"]))
            logger.info("supervisor confirmed intent={}", pending["intent"])
            return {
                "intent": pending["intent"],
                "result": result,
                "status": "executed",
                "error": None,
                "tokens_used": request_tokens.total,
            }

        if _is_negation(message):
            conversation_store.clear_pending(thread)
            msg = "Okay, cancelled. Anything else?"
            conversation_store.append(thread, user_msg, AIMessage(content=msg))
            logger.info("supervisor cancelled pending intent={}", pending["intent"])
            return {
                "intent": None,
                "result": {"message": msg},
                "status": "cancelled",
                "error": None,
                "tokens_used": 0,
            }

        # Neither yes nor no — drop the stale proposal and treat this as new.
        conversation_store.clear_pending(thread)
        logger.info("supervisor dropped stale pending: reply was neither yes nor no")

    # 2) Determine intent. Try the LLM-based pre-router first — a confident
    # read intent ("show me all issues", "what is Alice working on") skips
    # the full ``classify`` call (which does entity extraction too) and goes
    # straight to the read agent. The pre-router never returns a WRITE
    # (those need full entity extraction that only classify() does), so the
    # write paths below are unaffected. Empty / greetings short-circuit
    # inside pre_route without an LLM call.
    try:
        pre_routed = await pre_route(message, callbacks=callbacks)
    except Exception as e:
        logger.warning("supervisor pre_route crashed, falling through to classify: {}", e)
        pre_routed = None
    if pre_routed is not None:
        intent_result = pre_routed
        logger.info(
            "supervisor pre-routed intent={} (LLM hit, skipped full classify)",
            intent_result.intent.value,
        )
    else:
        # classify() never raises on a bad/parse response (it returns UNKNOWN),
        # but the LLM CALL itself can still raise — e.g. all model tiers are
        # rate-limited. Wrap it so a 429 becomes a clear "try again", not a 500.
        try:
            intent_result = await classify(message, callbacks=callbacks)
        except Exception as e:
            logger.error("supervisor classify crashed: {}", e, exc_info=True)
            msg = _error_message(e)
            conversation_store.append(thread, user_msg, AIMessage(content=msg))
            return {
                "intent": None,
                "result": {"message": msg},
                "status": "error",
                "error": None,
                "tokens_used": request_tokens.total,
            }
        logger.info(
            "supervisor intent={} confidence={:.2f}",
            intent_result.intent.value, intent_result.confidence,
        )

    # 2-pre) Low-confidence WRITE guard (plan 4.2): confirm the intent itself
    # before we propose a concrete mutation. A 0.4-confidence read of an
    # ambiguous message must not silently become a create/update proposal.
    if (
        intent_result.intent in WRITE_INTENTS
        and intent_result.confidence < MIN_WRITE_CONFIDENCE
    ):
        conversation_store.append(
            thread, user_msg, AIMessage(content=_LOW_CONFIDENCE_WRITE_MESSAGE)
        )
        logger.info(
            "supervisor low-confidence write intent={} conf={:.2f} — asking to clarify",
            intent_result.intent.value, intent_result.confidence,
        )
        return {
            "intent": intent_result.intent.value,
            "result": {"message": _LOW_CONFIDENCE_WRITE_MESSAGE},
            "status": "needs_input",
            "error": None,
            "tokens_used": request_tokens.total,
        }

    # 2a) Apply PM-domain conventions (a bug is high priority, "urgent" means
    # critical, …) as a deterministic post-pass over the LLM's extraction.
    # Done here, before validation/preview/pending, so the convention defaults
    # are what the user confirms AND what is executed — no preview/execute drift.
    if intent_result.intent == Intent.CREATE_ISSUE:
        intent_result.entities = normalize_entities(message, intent_result.entities)
        _ground_create_issue_title(intent_result.entities, message)
    elif intent_result.intent == Intent.CREATE_SPRINT:
        _ground_sprint_name(intent_result.entities, message)
    elif intent_result.intent == Intent.UPDATE_ISSUE:
        # An update must NOT run the keyword normalizer (it would invent a
        # type/priority from the sentence), but a literal "P0" the LLM copied
        # into the priority field still needs mapping to "critical".
        pr = intent_result.entities.get("priority")
        if pr:
            intent_result.entities["priority"] = canonicalize_priority(pr)

    # 2a-bis) A create with no usable title can't be meaningfully confirmed —
    # ask for one instead of proposing an issue titled "(untitled)". This runs
    # before validation so the user is asked for the missing essential first,
    # not told about an unresolved assignee on an unnamed issue.
    if (
        intent_result.intent == Intent.CREATE_ISSUE
        and not _effective_title(intent_result.entities)
    ):
        conversation_store.append(
            thread, user_msg, AIMessage(content=_MISSING_TITLE_MESSAGE)
        )
        logger.info("supervisor create_issue missing title — asking user")
        return {
            "intent": intent_result.intent.value,
            "result": {"message": _MISSING_TITLE_MESSAGE},
            "status": "needs_input",
            "error": None,
            "tokens_used": request_tokens.total,
        }

    # 2b) Pre-flight validation: for write intents, verify every entity the
    # user mentioned by NAME actually resolves AND stash the resolved IDs on
    # the proposal so the write executor can post the *exact* validated IDs
    # on confirmation (no re-resolution, no preview/execute drift). Failures
    # short-circuit BEFORE the proposal so the user gets a "no team member
    # named X" message instead of a "yes"-then-error flow. Upstream failures
    # (Neo4j/Node down) are swallowed and the proposal proceeds — the
    # post-confirm error path still surfaces the issue if needed.
    resolved: dict[str, Any] = {}
    if intent_result.intent in WRITE_INTENTS:
        validation_error, resolved = await _validate_intent(
            intent_result, org_slug, project_id
        )
        if validation_error:
            conversation_store.append(
                thread, user_msg, AIMessage(content=validation_error)
            )
            logger.info(
                "supervisor validation_failed intent={}",
                intent_result.intent.value,
            )
            return {
                "intent": intent_result.intent.value,
                "result": {"message": validation_error},
                "status": "validation_failed",
                "error": None,
                "tokens_used": request_tokens.total,
            }

    # 3) Writes are proposed, not executed, until the user confirms.
    if intent_result.intent in WRITE_INTENTS:
        preview = _preview(intent_result)
        conversation_store.set_pending(thread, {
            "intent": intent_result.intent.value,
            "message": message,
            "entities": intent_result.entities,
            "resolved": resolved,
            # Minted once per proposal; reused if the confirmation is retried
            # so the create is idempotent at the Node layer (item 17).
            "idempotency_key": uuid.uuid4().hex,
        })
        conversation_store.append(thread, user_msg, AIMessage(content=preview))
        logger.info("supervisor proposed intent={} (awaiting confirmation)", intent_result.intent.value)
        return {
            "intent": intent_result.intent.value,
            "result": {"message": preview},
            "status": "awaiting_confirmation",
            # Button affordances for an Adaptive Card (item 18). Clicking sends
            # the value ("yes"/"no") back as the next message — same path as a
            # typed reply. The dashboard ignores this field.
            "actions": CONFIRM_ACTIONS,
            "error": None,
            "tokens_used": request_tokens.total,
        }

    # 4) Reads and unknowns run immediately, with history for context.
    # Sprint 3.x: defense in depth — if the LLM rate-limits, the ReAct
    # agent's LLM step can raise, or the tool's HTTP call can bubble up
    # past its own try/except. Either way, the user must NOT see a 500.
    try:
        result = await _execute(
            intent_result, history + [user_msg], user_id, org_slug, project_id, callbacks=callbacks
        )
    except Exception as e:
        logger.error(
            "supervisor read dispatch crashed for intent={}: {}",
            intent_result.intent.value, e, exc_info=True,
        )
        result = {"message": _error_message(e)}
    conversation_store.append(thread, user_msg, AIMessage(content=result["message"]))
    return {
        "intent": intent_result.intent.value,
        "result": result,
        "status": "executed",
        "error": None,
        "tokens_used": request_tokens.total,
    }
