"""PM-domain convention normalizer.

A small, auditable post-pass over the LLM's extracted entities. The LLM is
good at reading free text but inconsistent about applying team conventions
("a bug is high priority", "urgent means critical"). Rather than bloat the
prompt with policy the model may or may not honour, we apply those rules
deterministically here, *after* extraction, so they are testable and the
behaviour is the same every time.

The normalizer is intentionally conservative:
  * it only ever sets ``type`` and ``priority`` — never a title, assignee, or
    sprint (those are real user intent and must come from the LLM);
  * an explicit value the LLM already extracted is never overwritten;
  * it does not mutate its input.
"""

from __future__ import annotations

import re
from typing import Any

# Multi-word phrases are listed before single words so the more specific match
# is the one that fires (e.g. "prod is down" rather than a bare "down").
_CRITICAL_KEYWORDS = ("prod is down", "critical", "urgent", "blocker", "p0", "sev1")
_HIGH_KEYWORDS = ("asap", "important", "soon", "p1")

# A bug defaults to high priority unless the text/LLM say otherwise.
_BUG_DEFAULT_PRIORITY = "high"

# The four priority values the backend accepts.
_CANONICAL_PRIORITIES = frozenset({"low", "medium", "high", "critical"})

# When the LLM puts a *non-canonical* priority straight into the entities
# (e.g. it copies the user's "P0" / "Sev1" / "important" verbatim into the
# priority field rather than into the free text), map it to the canonical
# value here. Keyword scanning of the message only fires when priority is
# unset; this table also fixes an already-set-but-non-canonical value, which
# is the bug behind stress query #32 ("P0 priority" leaking into the preview).
_PRIORITY_ALIASES = {
    "p0": "critical", "sev0": "critical", "sev1": "critical",
    "blocker": "critical", "urgent": "critical",
    "p1": "high", "sev2": "high", "important": "high",
    "p2": "medium", "sev3": "medium", "normal": "medium", "moderate": "medium",
    "p3": "low", "p4": "low", "sev4": "low", "minor": "low", "trivial": "low",
}


def _canonical_priority(value: Any) -> str | None:
    """Map a raw priority value to one of ``_CANONICAL_PRIORITIES``.

    Returns the canonical string, or ``None`` if the value is not a
    recognised priority at all (so the caller can leave it untouched rather
    than guess)."""
    v = str(value).strip().lower()
    if v in _CANONICAL_PRIORITIES:
        return v
    return _PRIORITY_ALIASES.get(v)


def _matches(text: str, keyword: str) -> bool:
    """Whole-word (or whole-phrase) match, case-insensitive, so 'debugging'
    does not count as a 'bug' and 'p10' does not count as 'p1'."""
    return re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE) is not None


def _is_unset(value: Any) -> bool:
    """The LLM returns null or '' for fields it could not fill — both mean
    'the user did not specify this', so a convention is free to fill them."""
    return value is None or value == ""


def canonicalize_priority(value: Any) -> Any:
    """Return the canonical priority for *value* if it is a recognised alias,
    else the original value unchanged.

    Used for UPDATE_ISSUE, where running the full keyword ``normalize_entities``
    is wrong (it would invent a ``type``/``priority`` from the surrounding
    sentence) but a literal "P0" in the priority field still needs mapping to
    "critical" before it reaches the backend."""
    canonical = _canonical_priority(value)
    return canonical if canonical is not None else value


def normalize_entities(text: str | None, entities: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of *entities* with ``type`` and ``priority`` filled in
    according to PM conventions, leaving every other key untouched."""
    out: dict[str, Any] = dict(entities or {})
    text = text or ""

    # --- type: the word "bug" in the request implies an issue of type=bug ---
    if _is_unset(out.get("type")) and _matches(text, "bug"):
        out["type"] = "bug"

    # If priority is already set explicitly, no convention may overwrite it —
    # but DO canonicalize a non-standard literal the LLM may have copied in
    # ("P0" -> "critical", "important" -> "high"). A value that is already
    # canonical ("low") is preserved exactly; an unrecognised value is left
    # untouched rather than guessed at.
    existing = out.get("priority")
    if not _is_unset(existing):
        canonical = _canonical_priority(existing)
        if canonical is not None:
            out["priority"] = canonical
        return out

    # --- priority keywords: critical beats high; most specific phrase wins ---
    if any(_matches(text, kw) for kw in _CRITICAL_KEYWORDS):
        out["priority"] = "critical"
    elif any(_matches(text, kw) for kw in _HIGH_KEYWORDS):
        out["priority"] = "high"
    # --- a bug with no stronger signal defaults to high ---
    elif out.get("type") == "bug":
        out["priority"] = _BUG_DEFAULT_PRIORITY

    return out
