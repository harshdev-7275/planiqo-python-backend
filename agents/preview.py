"""Write-confirmation preview + entity grounding.

When the supervisor proposes a mutating action it shows the user a plain-English
summary ("I'll create a bug titled 'X' with high priority…") so they confirm the
*exact* side-effects, never a hidden one. This module builds that string and, as
a pre-pass, grounds the LLM-extracted title/sprint name in words the user
actually typed (anti-fabrication, items 8/9).

Depends only on ``chains.title_clean`` and ``chains.extract_guard`` — no network,
no patched globals — so it stays unit-testable. The supervisor re-exports
``build_preview``/``effective_title``/``ground_create_issue_title``/
``ground_sprint_name`` under their old underscore names for compatibility.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from chains.extract_guard import extract_quoted, is_grounded, novel_words, strip_novel
from chains.title_clean import clean_title
from models.intents import Intent, IntentResult


def join_clauses(parts: list[str]) -> str:
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


def ground_create_issue_title(entities: dict[str, Any], message: str) -> None:
    """Items 8/9: if the LLM invented title content the user never typed,
    prune the fabricated words in place. If nothing grounded survives, the
    title is cleared and the missing-title gate asks the user — better than
    confirming a hallucinated name. Mutates ``entities``."""
    raw = entities.get("title") or entities.get("name")
    if not raw or is_grounded(str(raw), message):
        return
    grounded = strip_novel(str(raw), message)
    logger.info(
        "supervisor create_issue title not grounded (novel={}) — pruned to '{}'",
        sorted(novel_words(str(raw), message)), grounded,
    )
    entities["title"] = grounded
    entities.pop("name", None)


def ground_sprint_name(entities: dict[str, Any], message: str) -> None:
    """Item 9: ground a fabricated sprint name. Prefer the user's quoted name
    (sprint names are usually quoted, e.g. 'Q3 Hardening'); otherwise prune the
    invented words. If nothing grounded survives, keep the model's value and
    log — a sprint name is lower-stakes than an issue title and we don't want
    to block creation on a heuristic. Mutates ``entities``."""
    raw = entities.get("name") or entities.get("title")
    if not raw or is_grounded(str(raw), message):
        return
    grounded = extract_quoted(message) or strip_novel(str(raw), message)
    if grounded:
        logger.info(
            "supervisor create_sprint name not grounded (novel={}) — using '{}'",
            sorted(novel_words(str(raw), message)), grounded,
        )
        entities["name"] = grounded
        entities.pop("title", None)
    else:
        logger.warning(
            "supervisor create_sprint name '{}' not grounded in message and no "
            "quoted alternative — keeping as-is",
            raw,
        )


def effective_title(entities: dict[str, Any]) -> str:
    """The cleaned title we would actually use for a CREATE_ISSUE, or "" if the
    LLM extracted nothing usable. Single source of truth shared by the
    missing-title clarify gate (in ``run``) and the preview, so the two never
    disagree about whether a title exists."""
    return clean_title(entities.get("title") or entities.get("name"))


def build_preview(intent_result: IntentResult) -> str:
    """A human-readable confirmation prompt built from the extracted entities.
    All extracted mutations are surfaced so the user is never silently agreeing
    to a side-effect they didn't read (e.g. an assignee they didn't mention)."""
    entities = intent_result.entities
    if intent_result.intent == Intent.CREATE_ISSUE:
        # By the time we preview, the missing-title gate in ``run`` has already
        # short-circuited a titleless create — so this is non-empty. The
        # ``or "(untitled)"`` stays purely as defense-in-depth.
        title = effective_title(entities) or "(untitled)"
        # The LLM may return `null` for fields it can't fill — fall through
        # to defaults so the user never sees the literal Python ``None``
        # in their proposal.
        issue_type = entities.get("type") or "task"
        priority = entities.get("priority") or "medium"
        parts = [f"create a {issue_type} titled '{title}' with {priority} priority"]
        # The LLM may emit either 'assignee' or 'assignee_id' depending on the
        # prompt / model; accept both.
        assignee = entities.get("assignee") or entities.get("assignee_id")
        if assignee:
            parts.append(f"assign it to {assignee}")
        sprint = entities.get("sprint") or entities.get("sprint_id")
        if sprint:
            parts.append(f"put it in sprint {sprint}")
        return f"I'll {join_clauses(parts)}. Reply 'yes' to confirm or 'no' to cancel."
    if intent_result.intent == Intent.UPDATE_ISSUE:
        target = entities.get("issue") or entities.get("number") or "that issue"
        parts = [f"update {target}"]
        for key, label in (("status", "status"), ("priority", "priority"),
                           ("assignee", "assignee"), ("assignee_id", "assignee"),
                           ("title", "title"), ("sprint", "sprint"), ("sprint_id", "sprint")):
            if key in entities and key not in ("issue", "number"):
                parts.append(f"set {label} to {entities[key]}")
        return f"I'll {join_clauses(parts)}. Reply 'yes' to confirm or 'no' to cancel."
    if intent_result.intent == Intent.CREATE_SPRINT:
        name = entities.get("name") or entities.get("title") or "a new sprint"
        return f"I'll create sprint '{name}'. Reply 'yes' to confirm or 'no' to cancel."
    return "Please reply 'yes' to confirm or 'no' to cancel."
