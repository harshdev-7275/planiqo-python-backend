"""Presentation layer — shape how the AI reads.

The presentation layer takes the validated reasoning output (intent +
entities + insight + data) and a Persona, and produces the user-facing
reply. The LLM is *encouraged* to be expressive here — the bound is the
Pydantic ``Presentation`` schema, not the JSON-only prompt suffix.

Two outputs:

  * ``Presentation`` — the structured reply. ``narrative`` is free-form
    prose; everything else is a closed field. This is what the AI sees
    when reading the data + insight and choosing how to phrase it.

  * ``render_presentation(...)`` — pure function that converts a
    ``Presentation`` to a single string for the chat response. The
    rendering is deterministic and does NOT call the LLM — it's how
    the structured fields become the message the user actually reads.

Why the split: reasoning is strict (Intent, entities). Presentation is
free-form (narrative, opener, closer). Rendering is pure (no LLM). Three
layers, three different contracts.

Voice without drift — the 6 invariants from AIService.md:

  1. The validated entities the executor receives are UNCHANGED.
  2. The tool the executor calls is UNCHANGED.
  3. The confirmation requirement for writes is UNCHANGED.
  4. The authz the executor enforces is UNCHANGED.
  5. The prompt-injection guard is UNCHANGED.
  6. The user's language is matched.

The presentation can shape how the AI speaks; it cannot change what the
system does.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from chains.insight import Insight
from chains.persona_resolver import Persona
from clients.llm_client import ChatRunnable, get_fast_resilient
from config.settings import settings
from models.intents import Intent

# Intents that use the existing templated response (no LLM call) — same
# set as the insight layer's skip set. Pure data queries don't need
# persona-shaped prose; the table template is already the right reply.
SKIP_INTENTS: frozenset[Intent] = frozenset({
    Intent.QUERY_ISSUES,
    Intent.QUERY_SPRINT,
    Intent.SUMMARIZE,
    Intent.UNKNOWN,
    Intent.TEAMS_CONTEXT,
})


class Suggestion(BaseModel):
    """A clickable next-step the user can take. The UI renders these as
    quick-reply buttons (or as plain text in the Teams adapter)."""

    label: str
    intent_preview: Intent | None = None     # the Intent that would fire
    requires_confirmation: bool = False      # writes = True, reads = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Presentation(BaseModel):
    """The user-facing reply as structured fields. ``narrative`` is the
    free-form part — the rest are closed fields the renderer can use to
    build a consistent UI."""

    opener:   str = ""        # "Heads up — ", "Nice catch: ", "" (none)
    narrative: str            # 1-3 sentences in the persona's voice
    data_block: dict[str, Any] | None = None  # structured data (table) if any
    suggestions: list[Suggestion] = Field(default_factory=list)
    closer:   str = ""        # "— Vortex", "Want me to … ?", "" (none)
    tone_markers: list[str] = Field(default_factory=list)  # telemetry only


def _parse_response(message: AIMessage) -> Presentation:
    import re
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    code = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code:
        content = code.group(1).strip()
    if not content:
        raise ValueError("empty content")
    return Presentation.model_validate_json(content)


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], Presentation]:
    _PROMPT = ChatPromptTemplate.from_messages([
        ("system", "{persona_directive}"),
        (
            "human",
            "User said: {user_message}\n"
            "Classified intent: {intent}\n"
            "{insight_block}"
            "Data returned:\n{data}\n\n"
            "Compose a reply. Output JSON with these fields:\n"
            "  - opener: a 1-3 word lead-in matching the persona's voice "
            "           (or empty string for none)\n"
            "  - narrative: 1-3 sentences. The persona shapes the wording, "
            "               the data is the substance. If there's an insight, "
            "               work it in naturally — don't paste it verbatim.\n"
            "  - data_block: the structured data unchanged (or null if none)\n"
            "  - suggestions: 0-3 next-steps the user might want. Each is a "
            "                 JSON object with: label (string), "
            "                 intent_preview (Intent enum value or null), "
            "                 requires_confirmation (bool), confidence (0-1). "
            "                 Reads = no confirmation. Writes = yes.\n"
            "  - closer: 1 short line. Optional.\n"
            "  - tone_markers: list of strings for telemetry, e.g. "
            "                  ['concise', 'warm']. Empty list if unsure.\n"
            "Rules:\n"
            "  - NEVER invent a name, number, date, or assignee the data "
            "    doesn't support.\n"
            "  - NEVER re-derive entities. Use the validated intent and data "
            "    as-is.\n"
            "  - Match the user's language.\n"
            "  - If the data is a list of issues and the user asked for it, "
            "    just present the data. Don't add filler.\n"
            "Respond ONLY with the JSON object — no prose wrapper.",
        ),
    ])
    return _PROMPT | llm | RunnableLambda(_parse_response)


def _insight_block(insight: Insight | None) -> str:
    if insight is None:
        return ""
    return (
        f"Proactive insight (use naturally, don't paste verbatim): "
        f"{insight.kind} — {insight.message}\n"
    )


async def generate_presentation(
    *,
    persona: Persona,
    intent: Intent,
    user_message: str,
    data: Any = None,
    insight: Insight | None = None,
    data_block: dict[str, Any] | None = None,
    llm: ChatRunnable | None = None,
    callbacks: Callbacks = None,
) -> Presentation | None:
    """Return a Presentation for this turn, or None if the LLM call was
    skipped (data query or feature off) or the LLM call failed.

    The caller (supervisor) decides what to do with None — typically
    fall back to a templated response.
    """
    if not settings.PRESENTATION_ENABLED:
        return None
    if intent in SKIP_INTENTS:
        return None

    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    try:
        raw = await chain.ainvoke(
            {
                "persona_directive": persona.voice_directive,
                "user_message": user_message,
                "intent": intent.value,
                "insight_block": _insight_block(insight),
                "data": str(data) if data is not None else "(no data)",
            },
            config=config,
        )
    except (ValidationError, ValueError, TypeError) as e:
        logger.debug("presentation: parse failed: {}", e)
        return None
    except Exception as e:
        logger.warning("presentation: LLM call failed: {}", e)
        return None

    # Surface the data block the caller passed in (the LLM might or might
    # not echo it; we always carry the structured data through).
    if data_block is not None and raw.data_block is None:
        raw.data_block = data_block

    return raw


# =============================================================================
# RENDERING — pure functions, no LLM. Convert Presentation → display string.
# =============================================================================


_EMOJI_HINTS = {
    "warning": "⚠ ",
    "success": "✓ ",
    "info":    "",
    "none":    "",
}


def render_presentation(p: Presentation, *, persona: Persona | None = None) -> str:
    """Convert a Presentation to the single string the user sees.

    Pure function. Same input → same output. No LLM, no I/O, no exceptions.
    The output is the concatenation of: opener + narrative + data_block
    + suggestions + closer, with the persona's emoji policy applied.
    """
    parts: list[str] = []

    # Opener — short, capitalized, ends with a separator if non-empty.
    if p.opener:
        opener_text = _format_opener(p.opener, persona)
        parts.append(opener_text)

    # Narrative — the main message.
    if p.narrative:
        parts.append(p.narrative)

    # Data block — render as a small table if it's a list of issues, else
    # as a JSON-like block. The actual table rendering happens in the
    # frontend (which has the proper UI for it); the backend just emits
    # the structured data as a hint to the renderer.
    if p.data_block:
        parts.append(_render_data_block(p.data_block))

    # Suggestions — one per line, prefixed with an arrow.
    if p.suggestions:
        for s in p.suggestions:
            parts.append(f"  → {s.label}")

    # Closer — short tail.
    if p.closer:
        parts.append(p.closer)

    out = "\n".join(parts).strip()
    return out or "(no message)"


def _format_opener(opener: str, persona: Persona | None) -> str:
    """Apply the persona's emoji policy to the opener.

    The opener text comes from the LLM, but the leading emoji can be
    auto-stripped if the persona's emoji_policy is 'none' or 'rare' and
    the LLM overused emoji.
    """
    if persona is None or persona.emoji_policy == "expressive":
        return opener
    # Strip leading emoji if persona is conservative.
    if persona.emoji_policy in ("none", "rare"):
        cleaned = re.sub(r"^[\W_]+", "", opener).strip()
        return cleaned or opener
    return opener


def _render_data_block(block: dict[str, Any]) -> str:
    """A simple inline representation. The frontend has the rich UI;
    this is the backend's best-effort hint."""
    if "table" in block and isinstance(block["table"], list):
        rows = block["table"]
        if not rows:
            return ""
        # Render as a markdown-ish table.
        keys = list(rows[0].keys())
        out = ["| " + " | ".join(keys) + " |"]
        out.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        return "\n".join(out)
    if "summary" in block:
        return str(block["summary"])
    return ""


__all__ = [
    "Presentation",
    "SKIP_INTENTS",
    "Suggestion",
    "generate_presentation",
    "render_presentation",
]
