"""Shared helpers for the ReAct read agents (issue / member / sprint / …).

Two concerns the read agents share:

1. **Reasoning scratchpads.** Reasoning models (e.g. MiniMax-M2.7) wrap their
   final answer in ``<think>…</think>``. Strip it or the scratchpad leaks into
   the chat.

2. **Empty vs. unreachable.** A genuine "I looked and found none" must read
   differently from "I couldn't reach the server". The read tools already
   return a ``"Failed …"`` string on a fetch error (and a plain ``"No … found."``
   on a real empty), but the LLM that summarises the trace sometimes smooths a
   tool failure into a confident "none found". We detect the tool error
   deterministically from the message trace and make the failure explicit, so
   the user never reads an upstream outage as "nothing here".
"""

from __future__ import annotations

import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

# Every read tool returns a string starting with this token on a fetch failure
# (see tools/*.py — "Failed to get issues: …", "Failed to list members: …").
_TOOL_ERROR_PREFIX = "Failed"

_UNREACHABLE_CAVEAT = (
    "I couldn't reach the server for some of this, so this may be incomplete — "
    "please try again in a moment."
)


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks and trim."""
    return _THINK_RE.sub("", text).strip()


def tool_error_in_trace(messages: list[Any]) -> bool:
    """True if any tool call in the trace returned a fetch-failure string.

    Only the ``"Failed …"`` sentinel counts — a plain ``"No issues found."`` is
    a legitimate empty result, not an error, and must NOT trigger the caveat."""
    for m in messages:
        if getattr(m, "type", None) != "tool":
            continue
        content = getattr(m, "content", "")
        if isinstance(content, str) and content.lstrip().startswith(_TOOL_ERROR_PREFIX):
            return True
    return False


def _already_flags_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        k in lowered
        for k in ("couldn't reach", "could not reach", "try again", "unavailable")
    )


def finalize(messages: list[Any]) -> str:
    """Produce the user-facing string from a ReAct trace.

    Strips the reasoning scratchpad from the final message and, when a tool in
    the trace failed, appends an explicit unreachable-server caveat (unless the
    model already surfaced the failure) so an outage is never reported as an
    empty result."""
    last = messages[-1]
    raw = last.content if hasattr(last, "content") else str(last)
    content = strip_think(str(raw))
    if tool_error_in_trace(messages) and not _already_flags_failure(content):
        caveat = f"⚠️ {_UNREACHABLE_CAVEAT}"
        content = f"{content}\n\n{caveat}" if content else caveat
    return content
