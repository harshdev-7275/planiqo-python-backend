"""Regex/keyword pre-router (plan 4.1, item 12).

A cheap deterministic pass BEFORE the LLM classify. Many messages have an
unmistakable intent — "show me all issues", "summarize sprint 3", a bare
greeting — and don't need a model call. Routing those here saves tokens and
latency (the classify call is the most frequent paid request on the read path).

Design constraint — PRECISION OVER RECALL:
  * **Never pre-route a WRITE.** Writes need full entity extraction (title,
    assignee, priority, …) that only the LLM does; a half-extracted write would
    mis-execute. If the message contains any mutation verb we return ``None``
    and defer to the LLM.
  * **Only fire on unmistakable patterns.** Read pre-routes are anchored so a
    trailing "... assigned to Bob" (a member query) does not get swallowed as a
    plain issues list. A miss just costs one LLM call (the status quo); a wrong
    hit routes the user to the wrong agent, so we bias hard toward missing.
"""

from __future__ import annotations

import re

from models.intents import Intent, IntentResult

_PRE_ROUTED_CONFIDENCE = 0.95

# Any mutation verb/marker → defer to the LLM (we must not guess a write
# without extracting its entities). NOTE: "open", "closed", and "new" are
# deliberately absent — they double as read adjectives ("show me the OPEN
# issues") and would wrongly block a legitimate list query. Writes that use
# them ("open an issue") still fall through to None via the anchored positive
# patterns below, which only fire on show/list/view/display/summarize.
_WRITE_MARKERS = re.compile(
    r"\b(creat\w*|add|adds|adding|file|files|log|logs|raise|raises|submit\w*"
    r"|make|makes|making|track\w*|updat\w*|set|sets|setting|chang\w*|renam\w*"
    r"|reassign\w*|assign\w*|move\w*|delet\w*|remov\w*|edit\w*|mark\w*)\b",
    re.IGNORECASE,
)

# Bare social tokens — no PM intent. (A confirming "ok"/"yes" with a pending
# proposal is resolved earlier, before the pre-router ever runs.)
_GREETING = re.compile(
    r"^(hi|hey+|hello|yo|sup|good (morning|afternoon|evening)|thanks|thank you|"
    r"ty|thx|cheers|ok|okay|cool|great|nice|np)[!. ]*$",
    re.IGNORECASE,
)

# "show/list/view/display [me] [all|the|open|closed|high priority] issues" and
# nothing after the noun — the trailing anchor is what keeps a member query
# ("show me Bob's issues", "list issues assigned to Bob") OUT of this branch.
_QUERY_ISSUES = re.compile(
    r"^(show|list|view|display)\s+(me\s+)?"
    r"(all\s+|the\s+|open\s+|closed\s+|high[- ]?priority\s+)*"
    r"(issues?|bugs?|tickets?|tasks?|backlog)\s*[?.!]*$",
    re.IGNORECASE,
)

_SUMMARIZE = re.compile(
    r"^(summari[sz]e|recap|give me a (?:recap|summary))\b"
    r"|^how did .*\bsprint\b.* go\b",
    re.IGNORECASE,
)


def _routed(intent: Intent) -> IntentResult:
    return IntentResult(intent=intent, confidence=_PRE_ROUTED_CONFIDENCE, entities={})


def pre_route(message: str) -> IntentResult | None:
    """Return a confident IntentResult for an unambiguous message, else ``None``
    (defer to the LLM classifier). Never returns a WRITE intent."""
    text = message.strip()
    if not text:
        # Mirrors classify()'s empty-string short-circuit — no model needed.
        return IntentResult(intent=Intent.UNKNOWN, confidence=1.0, entities={})

    # A mutation verb means "extract entities first" — only the LLM does that.
    if _WRITE_MARKERS.search(text):
        return None

    if _GREETING.match(text):
        return IntentResult(intent=Intent.UNKNOWN, confidence=1.0, entities={})

    if _SUMMARIZE.search(text):
        return _routed(Intent.SUMMARIZE)

    if _QUERY_ISSUES.match(text):
        return _routed(Intent.QUERY_ISSUES)

    return None
