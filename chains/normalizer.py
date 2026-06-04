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


def _matches(text: str, keyword: str) -> bool:
    """Whole-word (or whole-phrase) match, case-insensitive, so 'debugging'
    does not count as a 'bug' and 'p10' does not count as 'p1'."""
    return re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE) is not None


def _is_unset(value: Any) -> bool:
    """The LLM returns null or '' for fields it could not fill — both mean
    'the user did not specify this', so a convention is free to fill them."""
    return value is None or value == ""


def normalize_entities(text: str | None, entities: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of *entities* with ``type`` and ``priority`` filled in
    according to PM conventions, leaving every other key untouched."""
    out: dict[str, Any] = dict(entities or {})
    text = text or ""

    # --- type: the word "bug" in the request implies an issue of type=bug ---
    if _is_unset(out.get("type")) and _matches(text, "bug"):
        out["type"] = "bug"

    # If priority is already set explicitly, no convention may overwrite it.
    if not _is_unset(out.get("priority")):
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
