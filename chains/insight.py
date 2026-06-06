"""Insight layer — proactive observations grounded in real data.

After ``classify`` returns an intent + validated entities, the insight
layer may run to surface observations the user didn't ask for but should
know. Examples from a real chat log:

  * 25 issues returned, 4 are critical → "Heads up — #14 is the only one
    in the 'critical' bucket that's blocking login, the other three are
    billing/admin. Want me to bump the priority on anything else?"
  * Two issues titled "login crash" → "Looks like #5 and #6 are
    duplicates. Want me to link them?"
  * User asked "what is Ranu working on?" → "Ranu has 6 issues, 2 of
    them are high-priority and one is critical. She's at capacity."

The insight is a small structured observation. The LLM does the
grounding (no fabrication), and the schema enforces safety rules.

Hard rules (from AIService.md § Insight layer):

  1. ``related_issue_numbers`` is an ENUM over the project's actual
     issues — the LLM cannot invent a number. Any out-of-set number is
     dropped silently.
  2. ``confidence < settings.INSIGHT_MIN_CONFIDENCE`` ⇒ drop the entire
     insight. Never half-show.
  3. The insight cannot change the user's intent. It can only observe.
  4. Insights that suggest a *write* action must set
     ``requires_confirmation: True`` on the corresponding Suggestion in
     the Presentation layer.
  5. The prompt-injection guard still applies — insight never echoes
     untrusted content.

The insight chain is gated by a deterministic ``SKIP_INTENTS`` set (data
queries don't need proactive observations; the user got what they asked
for). The skip is computed from the classified ``Intent`` enum, never
from the LLM.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from clients.llm_client import ChatRunnable, get_fast_resilient
from config.settings import settings
from models.intents import Intent

# Intents that already returned a complete answer — no proactive insight
# needed. Pure data queries (QUERY_ISSUES, QUERY_SPRINT, SUMMARIZE) and
# UNKNOWN skip the insight call entirely.
SKIP_INTENTS: frozenset[Intent] = frozenset({
    Intent.QUERY_ISSUES,
    Intent.QUERY_SPRINT,
    Intent.SUMMARIZE,
    Intent.UNKNOWN,
    Intent.TEAMS_CONTEXT,
})


class Insight(BaseModel):
    """One grounded observation the user might want to know about.

    The LLM is told to be specific, brief, and grounded — every claim
    must point at a real issue number from the data, never invented.
    """

    kind: Literal[
        "duplicate_suspect",
        "priority_audit",
        "pattern",
        "sprint_health",
        "team_load",
        "none",
    ] = "none"
    message: str = ""                  # 1-2 sentence observation, may be empty if kind="none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    related_issue_numbers: list[int] = Field(default_factory=list)
    suggested_intent: Intent | None = None   # the natural next action, if any


_SYSTEM = (
    "You are a senior project manager's read-only analyst. You surface "
    "ONE observation the user might want to know about, grounded in the "
    "data they just received. You NEVER invent issue numbers, names, or "
    "dates — every claim must point at a real item in the data.\n"
    "\n"
    "Output a single JSON object with these fields:\n"
    "  - kind: one of 'duplicate_suspect' (two+ issues look like the same "
    "          thing), 'priority_audit' (critical/high concentration), "
    "          'pattern' (recurring theme in titles), 'sprint_health' "
    "          (sprint behind/overloaded), 'team_load' (a person is at "
    "          capacity), or 'none' (nothing meaningful to say).\n"
    "  - message: 1-2 sentences, in the voice of a senior PM. No filler. "
    "             Empty string when kind='none'.\n"
    "  - confidence: 0.0-1.0. Use 0.7+ only when you're reasonably sure. "
    "                Below 0.5, return kind='none'.\n"
    "  - related_issue_numbers: list of integer issue numbers from the "
    "          data. NEVER invent. Empty list if the observation isn't "
    "          about specific issues.\n"
    "  - suggested_intent: the natural next action (Intent enum value) "
    "          if any — e.g. UPDATE_ISSUE if the user might want to "
    "          link duplicates, or null if there's nothing specific.\n"
    "\n"
    "If the data is unremarkable (a normal list, no patterns), return "
    "kind='none'. DO NOT manufacture observations. The best insight is "
    "sometimes no insight.\n"
    "Respond ONLY with the JSON object — no prose, no markdown, no code fences."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    (
        "human",
        "User intent: {intent}\n"
        "User question: {user_message}\n"
        "Issue numbers the user can see in this project: {known_numbers}\n"
        "Data returned to the user:\n{data}\n",
    ),
])


def _parse_response(message: AIMessage) -> Insight:
    """Parse the LLM's output. NEVER raises — bad output → kind='none'."""
    import re
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    code = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code:
        content = code.group(1).strip()
    if not content:
        return Insight()
    try:
        return Insight.model_validate_json(content)
    except ValidationError as e:
        logger.debug("insight: model output did not validate: {}", e)
        return Insight()


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], Insight]:
    return _PROMPT | llm | RunnableLambda(_parse_response)


def _filter_known_numbers(
    related: list[int], known: set[int]
) -> list[int]:
    """Defence in depth — drop any number the LLM invented. The LLM was
    told the known set, but a misbehaving model could ignore that."""
    return [n for n in related if n in known]


async def generate_insight(
    *,
    intent: Intent,
    user_message: str,
    data: str,
    known_issue_numbers: list[int] | None = None,
    llm: ChatRunnable | None = None,
    callbacks: Callbacks = None,
) -> Insight | None:
    """Return an Insight for this turn, or None if the LLM call was
    skipped (data query, UNKNOWN, or feature off) or the LLM returned
    kind='none' / low confidence.

    ``known_issue_numbers`` is the closed set the LLM is allowed to
    reference. Any out-of-set number in ``related_issue_numbers`` is
    silently dropped. Pass ``None`` to skip the filter (e.g. when the
    intent isn't about issues at all — the LLM will return no numbers
    naturally).
    """
    if not settings.INSIGHT_ENABLED:
        return None
    if intent in SKIP_INTENTS:
        return None

    known_set = set(known_issue_numbers) if known_issue_numbers is not None else set()
    known_list = sorted(known_set) if known_set else "(no issue data)"

    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    try:
        raw = await chain.ainvoke(
            {
                "intent": intent.value,
                "user_message": user_message,
                "known_numbers": ", ".join(str(n) for n in known_list),
                "data": data or "(no data)",
            },
            config=config,
        )
    except Exception as e:
        logger.warning("insight: LLM call failed: {}", e)
        return None

    if raw.kind == "none" or not raw.message:
        return None
    if raw.confidence < settings.INSIGHT_MIN_CONFIDENCE:
        logger.debug(
            "insight: low confidence {:.2f} < {:.2f}, dropping",
            raw.confidence, settings.INSIGHT_MIN_CONFIDENCE,
        )
        return None

    # Defence in depth: drop any number the LLM invented.
    if raw.related_issue_numbers and known_set:
        raw.related_issue_numbers = _filter_known_numbers(
            raw.related_issue_numbers, known_set
        )

    return raw


__all__ = ["Insight", "SKIP_INTENTS", "generate_insight"]
