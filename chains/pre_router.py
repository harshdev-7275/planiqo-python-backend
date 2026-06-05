"""LLM-based pre-router.

A fast classification pass BEFORE the full ``classify`` chain. Many messages
have an unmistakable read intent ("show me all issues", "summarize sprint 3",
"what is Alice working on") and don't need entity extraction — but the natural
language is too varied to anchor with regex ("show me", "list", "display",
"what are", "give me", "can you fetch", "what's on the backlog", …).

The LLM handles these variations without hardcoding. The pre-router is still
deliberately strict:

* **Never pre-route a WRITE.** Writes need full entity extraction (title,
  assignee, priority, …) that only ``classify`` does; a half-extracted write
  would mis-execute. If the LLM returns a WRITE intent we drop it and defer.
* **Confidence gate.** Only fire on high-confidence classifications. A miss
  just costs one ``classify`` call (the status quo); a wrong hit routes the
  user to the wrong agent, so we bias hard toward missing.
* **Deterministic short-circuits** for empty messages and bare greetings —
  no LLM needed, saves a call.

The LLM call uses the fast model (``get_fast_resilient()``) with the same
cross-tier fallback as ``classify``. Token budget: <200 in + <50 out.
"""

from __future__ import annotations

import re

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import ValidationError

from clients.llm_client import ChatRunnable, get_fast_resilient
from models.intents import Intent, IntentResult

# Minimum confidence to accept a pre-routed intent. Below this, defer to
# the full classify chain (which does entity extraction too).
_PRE_ROUTE_MIN_CONFIDENCE = 0.7

# WRITE intents — never pre-routable, even if the LLM returns them.
# ``classify`` is the only place that does entity extraction for writes.
_WRITE_INTENTS = frozenset({
    Intent.CREATE_ISSUE,
    Intent.UPDATE_ISSUE,
    Intent.CREATE_SPRINT,
})

# Bare social tokens — short-circuit, no LLM call. (A confirming "ok"/"yes"
# with a pending proposal is resolved earlier, before the pre-router runs.)
_GREETING = re.compile(
    r"^(hi|hey+|hello|yo|sup|good (morning|afternoon|evening)|thanks|thank you|"
    r"ty|thx|cheers|ok|okay|cool|great|nice|np)[!. ]*$",
    re.IGNORECASE,
)

_SYSTEM = (
    "You are a fast intent classifier for a project management assistant. "
    "Output a single JSON object with two fields: "
    "  - intent: one of QUERY_ISSUES, QUERY_SPRINT, QUERY_MEMBER, SUMMARIZE, UNKNOWN. "
    "  - confidence: 0.0-1.0 (use 0.5+ only if you're reasonably sure; 0.0-0.3 for UNKNOWN). "
    "WRITE intents (CREATE_ISSUE, UPDATE_ISSUE, CREATE_SPRINT) are FORBIDDEN at this "
    "stage — they need entity extraction that this pass does not do. If the user "
    "wants to create, update, or change anything, return UNKNOWN with low confidence "
    "and the next pass will handle it. "
    "Examples: "
    "  'show me all issues' → QUERY_ISSUES, 0.95. "
    "  'what\\'s on the backlog' → QUERY_ISSUES, 0.85. "
    "  'list bugs' → QUERY_ISSUES, 0.9. "
    "  'summarize sprint 2' → SUMMARIZE, 0.95. "
    "  'what is Alice working on' → QUERY_MEMBER, 0.95. "
    "  'create a bug for login' → UNKNOWN, 0.0. "
    "Respond ONLY with the JSON object — no prose, no markdown, no code fences."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "{message}"),
])


def _parse_response(message: AIMessage) -> IntentResult:
    """Parse the LLM's raw output into an IntentResult.

    NEVER raises on the happy path. Reasoning-model ``<think>`` wrappers and
    markdown code fences are stripped. A parse failure is raised as
    ``ValueError`` so the caller (pre_route) can catch it and defer.
    """
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code_match:
        content = code_match.group(1).strip()
    if not content:
        raise ValueError("empty LLM content")
    return IntentResult.model_validate_json(content)


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], IntentResult]:
    return _PROMPT | llm | RunnableLambda(_parse_response)


async def pre_route(
    message: str,
    llm: ChatRunnable | None = None,
    callbacks: Callbacks = None,
) -> IntentResult | None:
    """Return a confident IntentResult for an unambiguous read message, or
    ``None`` to defer to the full ``classify`` chain. Never returns a WRITE.

    Empty messages and bare greetings short-circuit without an LLM call.
    """
    text = message.strip()

    # Empty → UNKNOWN, no LLM call
    if not text:
        return IntentResult(intent=Intent.UNKNOWN, confidence=1.0, entities={})

    # Greeting → UNKNOWN, no LLM call
    if _GREETING.match(text):
        return IntentResult(intent=Intent.UNKNOWN, confidence=1.0, entities={})

    # LLM call for everything else
    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    try:
        result = await chain.ainvoke({"message": text}, config=config)
    except (ValidationError, ValueError, TypeError) as e:
        # Model returned garbage; defer to full classify.
        logger.debug("pre_route: parse failed: {}", e)
        return None
    except Exception as e:
        # LLM call itself failed (network, rate limit, etc.); defer.
        logger.warning("pre_route: LLM call failed: {}", e)
        return None

    # Safety: writes are NEVER pre-routed, even if the LLM thinks it's a write.
    if result.intent in _WRITE_INTENTS:
        logger.debug(
            "pre_route: LLM returned WRITE intent {}, deferring to classify",
            result.intent.value,
        )
        return None

    # Low confidence → defer
    if result.confidence < _PRE_ROUTE_MIN_CONFIDENCE:
        logger.debug(
            "pre_route: low confidence {:.2f} < {:.2f}, deferring to classify",
            result.confidence,
            _PRE_ROUTE_MIN_CONFIDENCE,
        )
        return None

    return result
