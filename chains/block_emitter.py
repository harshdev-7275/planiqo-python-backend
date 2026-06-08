"""LLM block emitter — the generative, PM-scoped path of the render contract.

Given a read answer, the LLM re-expresses it as typed render-contract blocks,
bounded by ``chains/render_policy.RENDER_POLICY_PROMPT`` and validated by
``chains/render_contract.validate_blocks`` (drop-invalid; never trust the
model's shape). This is the "presentation pass" from the render-contract
design: it runs for read intents with structure, behind the
``RENDER_BLOCKS_LLM_ENABLED`` flag (opt-in — it adds an LLM call).

Returns the rich blocks on success, or ``None`` when the model produced
nothing better than prose / the call was skipped or failed. The caller then
relies on the service-edge prose carrier
(``chains/render_contract.attach_render_blocks``), so the chat always renders.

The inline single-pass fast-path (a render-tool the read agent calls directly)
is a future cost optimization; correctness lives here in the presentation pass.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger

from chains.render_contract import validate_blocks
from chains.render_policy import RENDER_POLICY_PROMPT
from clients.llm_client import ChatRunnable, get_fast_resilient
from config.settings import settings
from models.blocks import Block
from models.intents import Intent

# Read intents whose answers tend to carry visual structure worth a block
# pass. SUMMARIZE is excluded — it already emits deterministic blocks
# (zero-LLM), and UNKNOWN is plain prose.
EMIT_INTENTS: frozenset[Intent] = frozenset(
    {Intent.QUERY_ISSUES, Intent.QUERY_SPRINT, Intent.QUERY_MEMBER}
)


def _parse_blocks_message(message: AIMessage) -> list[Block]:
    """LLM message -> validated blocks. Strips think-tags and code fences,
    parses JSON, then runs the boundary validator. Returns [] when nothing
    valid survives (the caller treats that as 'no rich blocks')."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if fence:
        content = fence.group(1).strip()
    if not content:
        return []
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    result = validate_blocks(raw)
    return [] if result.used_fallback else result.blocks


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], list[Block]]:
    # The policy goes in a concrete SystemMessage (not a template) so its many
    # literal JSON braces are not parsed as prompt variables.
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=RENDER_POLICY_PROMPT),
            (
                "human",
                "User asked: {user_message}\n"
                "Intent: {intent}\n"
                "Answer to render as blocks:\n{answer}\n"
                "Return the JSON object now.",
            ),
        ]
    )
    return prompt | llm | RunnableLambda(_parse_blocks_message)


async def emit_blocks(
    *,
    intent: Intent,
    user_message: str,
    answer: str,
    data: Any = None,
    llm: ChatRunnable | None = None,
    callbacks: Callbacks = None,
) -> list[Block] | None:
    """Return rich render-contract blocks for a read answer, or None.

    None means: skipped (flag off / intent not in EMIT_INTENTS / empty
    answer), the LLM call failed, or the model produced nothing better than
    prose. The caller then relies on the service-edge prose carrier.
    """
    if not settings.RENDER_BLOCKS_LLM_ENABLED:
        return None
    if intent not in EMIT_INTENTS:
        return None
    if not answer.strip():
        return None

    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    try:
        blocks = await chain.ainvoke(
            {"user_message": user_message, "intent": intent.value, "answer": answer},
            config=config,
        )
    except Exception as e:
        logger.warning("block_emitter: LLM call failed, falling back to prose: {}", e)
        return None

    return blocks or None


__all__ = ["EMIT_INTENTS", "emit_blocks"]
