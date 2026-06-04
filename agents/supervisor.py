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
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

import agents.issue_agent as _issue_agent
import agents.sprint_agent as _sprint_agent
from agents.state import SupervisorState
from chains.intent import classify
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from config.settings import settings
from graph.queries import find_user_by_name
from memory.store import DEFAULT_PENDING_TTL_SECONDS, conversation_store
from metering.callback import UsageCallback
from metering.usage import RequestTokens, usage_store
from models.intents import Intent, IntentResult

# Intents that mutate data — proposed and confirmed before they run.
WRITE_INTENTS = {Intent.CREATE_ISSUE, Intent.UPDATE_ISSUE, Intent.CREATE_SPRINT}

# Emit a per-org usage milestone at every Nth request so ops can see traffic
# patterns and per-tenant cost growth in the log stream without scraping the
# /admin/usage endpoint. ``0`` disables the log.
USAGE_MILESTONE_EVERY = 100

# Exact-phrase matches. Kept for short replies where word-set matching is too
# permissive (e.g. "ok" alone vs. "ok to delete production" — the latter is
# not a confirmation, but every word *is* in _AFFIRM_WORDS).
_AFFIRMATIONS = {
    "yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "ok", "okay", "k",
    "sure", "do it", "go ahead", "yes please", "please do", "proceed", "sounds good",
}
_NEGATIONS = {
    "no", "n", "nope", "cancel", "stop", "nevermind", "never mind", "abort",
    "dont", "don't",
}

# Vocabulary for word-set matching — every word in the reply must come from
# this set AND at least one word from the core set must be present. This lets
# "yes please do it" / "yeah go ahead" / "ok sounds good" all match without
# letting unrelated sentences through.
_AFFIRM_WORDS: frozenset[str] = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "k", "sure", "confirm",
    "confirmed", "proceed", "please", "do", "it", "go", "ahead", "sounds", "good",
})
_AFFIRM_CORE: frozenset[str] = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ok", "okay", "k", "sure", "confirm",
    "confirmed", "proceed",
})
_NEGATE_WORDS: frozenset[str] = frozenset({
    "no", "n", "nope", "cancel", "stop", "abort", "never", "mind", "nevermind",
    "dont", "don't",
})
_NEGATE_CORE: frozenset[str] = frozenset({
    "no", "n", "nope", "cancel", "stop", "abort", "dont", "don't",
})

_UNKNOWN_MESSAGE = (
    "I didn't understand that. Try asking about issues, sprints, or team members."
)

# Common sentence punctuation that would otherwise survive ``.split()`` and
# block word-set matching (e.g. "yes," / "no.").
_PUNCT_RE = re.compile(r"[.!?,;:]")


def _thread_id(user_id: str, org_slug: str, project_id: str | None) -> str:
    return f"{user_id}:{org_slug}:{project_id or '-'}"


def _normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace, replace punctuation with spaces,
    collapse runs of spaces. Keeps word boundaries clean for matching."""
    cleaned = _PUNCT_RE.sub(" ", text.strip().lower())
    return " ".join(cleaned.split())


def _is_word_set_reply(normalized: str, vocab: frozenset[str], core: frozenset[str]) -> bool:
    """True iff the reply is non-empty, every word is in ``vocab``, and at
    least one core (load-bearing) word is present. Prevents stray confirmations
    like 'ok' from being triggered by unrelated sentences that happen to
    contain only affirmation words."""
    words = normalized.split()
    if not words:
        return False
    if not all(w in vocab for w in words):
        return False
    return any(w in core for w in words)


def _is_affirmation(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    if norm in _AFFIRMATIONS:  # exact phrase like "do it" / "sounds good"
        return True
    return _is_word_set_reply(norm, _AFFIRM_WORDS, _AFFIRM_CORE)


def _is_negation(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    if norm in _NEGATIONS:  # exact phrase like "never mind"
        return True
    return _is_word_set_reply(norm, _NEGATE_WORDS, _NEGATE_CORE)


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


def _join_clauses(parts: list[str]) -> str:
    """Join human-readable action clauses with Oxford-comma grammar.

    ``["a"]``         -> "a"
    ``["a", "b"]``    -> "a and b"
    ``["a", "b", "c"]`` -> "a, b, and c"
    """
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _preview(intent_result: IntentResult) -> str:
    """A human-readable confirmation prompt built from the extracted entities.
    All extracted mutations are surfaced so the user is never silently agreeing
    to a side-effect they didn't read (e.g. an assignee they didn't mention)."""
    entities = intent_result.entities
    if intent_result.intent == Intent.CREATE_ISSUE:
        title = entities.get("title") or entities.get("name") or "(untitled)"
        issue_type = entities.get("type", "task")
        priority = entities.get("priority", "medium")
        parts = [f"create a {issue_type} titled '{title}' with {priority} priority"]
        # The LLM may emit either 'assignee' or 'assignee_id' depending on the
        # prompt / model; accept both.
        assignee = entities.get("assignee") or entities.get("assignee_id")
        if assignee:
            parts.append(f"assign it to {assignee}")
        sprint = entities.get("sprint") or entities.get("sprint_id")
        if sprint:
            parts.append(f"put it in sprint {sprint}")
        return f"I'll {_join_clauses(parts)}. Reply 'yes' to confirm or 'no' to cancel."
    if intent_result.intent == Intent.UPDATE_ISSUE:
        target = entities.get("issue") or entities.get("number") or "that issue"
        parts = [f"update {target}"]
        for key, label in (("status", "status"), ("priority", "priority"),
                           ("assignee", "assignee"), ("assignee_id", "assignee"),
                           ("title", "title"), ("sprint", "sprint"), ("sprint_id", "sprint")):
            if key in entities and key not in ("issue", "number"):
                parts.append(f"set {label} to {entities[key]}")
        return f"I'll {_join_clauses(parts)}. Reply 'yes' to confirm or 'no' to cancel."
    if intent_result.intent == Intent.CREATE_SPRINT:
        name = entities.get("name") or entities.get("title") or "a new sprint"
        return f"I'll create sprint '{name}'. Reply 'yes' to confirm or 'no' to cancel."
    return "Please reply 'yes' to confirm or 'no' to cancel."


# --- pre-flight entity validation -------------------------------------------
#
# Before showing the "I'll create X, confirm?" preview, verify that every
# entity the user mentioned by NAME actually resolves to a real record.
# Without this, the user confirms "yes" only to learn the assignee / sprint
# didn't exist — a frustrating two-step failure. With this, the error lands
# immediately and lists the valid options.
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


def _coerce_member_name(m: dict) -> str:
    return m.get("name") or m.get("userName") or m.get("userId") or "?"


def _coerce_sprint_name(s: dict) -> str:
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
    # No match — list actual members so the user can pick.
    try:
        members = await node_api_client.get_project_members(org_slug, project_id)
    except Exception as e:
        logger.warning("validation: Node unreachable listing members: {}", e)
        members = []
    msg = f"No team member named '{name}' found in this project."
    if members:
        names = [_coerce_member_name(m) for m in members[:_MAX_LISTED]]
        msg += f" Team members: {', '.join(names)}."
    msg += " Please pick one of them or use the exact name."
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
        if name.lower() in (_coerce_sprint_name(s)).lower()
    ]
    if matches:
        return None
    names = [_coerce_sprint_name(s) for s in sprints[:_MAX_LISTED]]
    msg = f"No sprint matching '{name}' found in this project."
    if names:
        msg += f" Available sprints: {', '.join(names)}."
    return msg


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
) -> str | None:
    """Pre-flight validation for write intents. Returns None if all referenced
    entities resolve, else a user-facing error message describing what's
    missing and listing valid options."""
    if not project_id:
        return None  # no project context to validate against
    intent = intent_result.intent
    if intent not in WRITE_INTENTS:
        return None
    entities = intent_result.entities
    errors: list[str] = []

    if intent == Intent.CREATE_ISSUE:
        assignee = entities.get("assignee") or entities.get("assignee_id")
        if assignee:
            err = await _check_assignee_name(org_slug, project_id, str(assignee))
            if err:
                errors.append(err)
        sprint = entities.get("sprint") or entities.get("sprint_id")
        if sprint:
            err = await _check_sprint_name(org_slug, project_id, str(sprint))
            if err:
                errors.append(err)

    elif intent == Intent.UPDATE_ISSUE:
        # Resolve issue number + any name→ID change first
        issue_num = entities.get("issue") or entities.get("number")
        err = await _check_issue_number(org_slug, project_id, issue_num)
        if err:
            errors.append(err)
        assignee = entities.get("assignee") or entities.get("assignee_id")
        if assignee:
            err = await _check_assignee_name(org_slug, project_id, str(assignee))
            if err:
                errors.append(err)
        sprint = entities.get("sprint") or entities.get("sprint_id")
        if sprint:
            err = await _check_sprint_name(org_slug, project_id, str(sprint))
            if err:
                errors.append(err)

    return "\n\n".join(errors) if errors else None


# --- execution ---------------------------------------------------------------


async def _execute(
    intent_result: IntentResult,
    messages: list[BaseMessage],
    user_id: str,
    org_slug: str,
    project_id: str | None,
    callbacks: Callbacks = None,
) -> dict[str, Any]:
    """Dispatch a classified intent to its agent and return a result dict that
    always contains a 'message'."""
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
    elif target == "handle_unknown":
        return {"message": _UNKNOWN_MESSAGE}
    else:  # member_agent / summarize_agent / teams_agent — not built yet
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
    pending = conversation_store.get_pending(thread, ttl_seconds=DEFAULT_PENDING_TTL_SECONDS)
    if pending is not None:
        if _is_affirmation(message):
            conversation_store.clear_pending(thread)
            intent_result = IntentResult(
                intent=Intent(pending["intent"]),
                confidence=1.0,
                entities=pending.get("entities", {}),
            )
            # Execute the ORIGINAL request, not the bare "yes".
            exec_messages = history + [HumanMessage(content=pending["message"])]
            result = await _execute(
                intent_result, exec_messages, user_id, org_slug, project_id, callbacks=callbacks
            )
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

    # 2) Classify the new message.
    intent_result = await classify(message, callbacks=callbacks)
    logger.info(
        "supervisor intent={} confidence={:.2f}",
        intent_result.intent.value, intent_result.confidence,
    )

    # 2b) Pre-flight validation: for write intents, verify every entity the
    # user mentioned by NAME actually resolves. Failures short-circuit BEFORE
    # the proposal so the user gets a "no team member named X" message
    # instead of a "yes"-then-error flow. Upstream failures (Neo4j/Node
    # down) are swallowed and the proposal proceeds — the post-confirm
    # error path still surfaces the issue if needed.
    if intent_result.intent in WRITE_INTENTS:
        validation_error = await _validate_intent(intent_result, org_slug, project_id)
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
        })
        conversation_store.append(thread, user_msg, AIMessage(content=preview))
        logger.info("supervisor proposed intent={} (awaiting confirmation)", intent_result.intent.value)
        return {
            "intent": intent_result.intent.value,
            "result": {"message": preview},
            "status": "awaiting_confirmation",
            "error": None,
            "tokens_used": request_tokens.total,
        }

    # 4) Reads and unknowns run immediately, with history for context.
    result = await _execute(
        intent_result, history + [user_msg], user_id, org_slug, project_id, callbacks=callbacks
    )
    conversation_store.append(thread, user_msg, AIMessage(content=result["message"]))
    return {
        "intent": intent_result.intent.value,
        "result": result,
        "status": "executed",
        "error": None,
        "tokens_used": request_tokens.total,
    }
