"""Context-aware follow-up resolver.

When a read turn ends by asking the user something (e.g. "Would you like me to
look up a specific issue, or list members first?"), the user's next reply is
often a short, context-dependent answer: "yes", "the second one", "list
members". Classifying that reply *cold* (the supervisor's normal path) turns it
into UNKNOWN → "I didn't understand that" — the conversation loses context.

This chain rewrites such a reply into a **standalone request** using the
assistant's question as context, so the existing classify → dispatch pipeline
can handle it normally. It is deliberately small (fast model, temperature 0,
tiny token budget) and only runs when a clarification is actually pending.

Design notes:
* It NEVER raises — on any failure it returns the reply unchanged, so the worst
  case is the status quo (the reply is classified as-is), never a 500.
* If the reply is already a complete, standalone request (the user ignored the
  question and asked something new), it is returned unchanged.
* For a bare affirmation ("yes"/"sure") to a multi-option question, the FIRST
  option is chosen — a reasonable default that beats a dead-end. Clean
  per-option disambiguation is the job of the structured-choices follow-up
  (the "complete" version), not this minimal resolver.
"""

from __future__ import annotations

import re

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger

from clients.llm_client import ChatRunnable, get_fast_resilient

_SYSTEM = (
    "You rewrite a user's short reply into a single standalone request for a "
    "project management assistant, using the question the assistant just asked "
    "as context. "
    "Rules: "
    "  - If the reply is already a complete, standalone request (it makes sense "
    "    on its own), return it UNCHANGED. "
    "  - If the reply is a short answer to the assistant's question (e.g. 'the "
    "    second one', 'list members', 'the login one'), rewrite it into a full "
    "    standalone request that captures what the user wants. "
    "  - If the assistant offered multiple options and the user simply affirms "
    "    ('yes', 'sure', 'ok', 'please do'), choose the FIRST option offered. "
    "  - Keep it short and in the user's language. "
    "Respond with ONLY the rewritten request text — no quotes, no prose, no "
    "explanation, no markdown."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Assistant asked: {question}\nUser replied: {reply}"),
])


def _extract_text(message: AIMessage) -> str:
    """Pull the plain rewritten request out of the model's reply.

    Strips reasoning-model ``<think>`` wrappers and markdown code fences (the
    same failure modes intent.py / pre_router.py guard against), then trims.
    """
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    code_match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", content)
    if code_match:
        content = code_match.group(1).strip()
    return content.strip().strip('"').strip()


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], str]:
    return _PROMPT | llm | RunnableLambda(_extract_text)


async def resolve_followup(
    question: str,
    reply: str,
    llm: ChatRunnable | None = None,
    callbacks: Callbacks = None,
) -> str:
    """Return ``reply`` rewritten as a standalone request, or ``reply``
    unchanged when it cannot help (empty reply, LLM failure, empty output).
    Never raises."""
    if not reply.strip():
        return reply

    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    try:
        out = await chain.ainvoke({"question": question, "reply": reply}, config=config)
    except Exception as e:
        logger.warning("followup_resolver: LLM call failed, using reply as-is: {}", e)
        return reply
    return out or reply
