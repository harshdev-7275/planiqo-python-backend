"""Persona resolution — the AI's voice, not its brain.

A ``Persona`` describes how the AI talks (opener style, verbosity,
suggestion aggressiveness, emoji policy) — never what it does. Persona is
data, not code: phrases live in the persona config, not in Python
``if/else``.

The resolver follows a 4-step chain (first match wins):

  1. Per-conversation override — ``memory.store.pending["persona_override"]``
     set when the user says "be more concise" or similar mid-thread.
  2. Per-user preference      — ``users.preferences.persona`` (DB column,
     not yet wired — falls through to step 3).
  3. Per-org default          — ``organizations.settings.persona``
     (not yet wired — falls through to step 4).
  4. Global default           — ``settings.PERSONA_DEFAULT`` (env var).

The result is cached per (org_slug, user_id) for ``PERSONA_CACHE_TTL_SECONDS``
(default 300s) so the resolver adds zero LLM cost to the steady state.

Three built-in personas ship by default:

  * ``senior_pm``  — warm, proactive, suggests next steps. The default.
                    "Heads up — you've got 4 critical issues; #14 is the
                    one most users are blocked on right now."
  * ``auditor``    — terse, formal, no emoji, no suggestions. "4 critical
                    issues: #14, #18, #22, #25."
  * ``assistant``  — neutral, factual, balanced. Answers the question,
                    no flair, no push.

The persona does NOT affect:
  - which tool the executor calls
  - the entities the executor mutates
  - whether a write requires confirmation
  - the authz the executor enforces
  - the prompt-injection guard
  - the user's language

See AIService.md § Voice without drift for the 6 invariants.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings

# Persona is data, not code. If you find yourself writing phrases in Python,
# STOP — put them in the persona's voice_directive. The LLM will phrase them
# per the directive.

Verbosity = Literal["terse", "balanced", "detailed"]
EmojiPolicy = Literal["none", "rare", "moderate", "expressive"]


class Persona(BaseModel):
    """The AI's voice on a single turn. See module docstring."""

    name: str
    voice_directive: str
    verbosity: Verbosity = "balanced"
    opinion_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    emoji_policy: EmojiPolicy = "rare"
    suggestion_aggressiveness: float = Field(default=0.3, ge=0.0, le=1.0)
    signature: str | None = None
    rules: list[str] = Field(default_factory=list)


# Built-in personas. The default is ``senior_pm`` because that's what a
# project-management tool's AI assistant should sound like: proactive,
# context-aware, suggests next steps.

_PERSONAS: dict[str, Persona] = {
    "senior_pm": Persona(
        name="senior_pm",
        voice_directive=(
            "You are a senior project manager who knows the team and the "
            "backlog by heart. You speak like a real PM in a standup: warm, "
            "direct, and context-aware. You notice patterns (duplicates, "
            "critical issues piling up, who's overloaded, what's been idle "
            "too long) and surface them. You suggest the next concrete step "
            "when it would help, but you don't over-suggest. You use 0-1 "
            "emojis per reply, and only when it actually helps (✓ for "
            "confirmations, ⚠ for warnings, never decorative). You don't "
            "echo the user's question back at them. You don't add filler "
            "like 'Sure!' or 'Of course!' — you just answer."
        ),
        verbosity="balanced",
        opinion_strength=0.7,
        emoji_policy="rare",
        suggestion_aggressiveness=0.5,
        signature=None,
        rules=[
            "Surface duplicate-looking issues and critical-priority patterns",
            "Suggest the single most useful next action, not a menu",
            "Mention who's overloaded if the user is asking about a person",
            "Never invent a name, number, or date the data doesn't support",
        ],
    ),
    "auditor": Persona(
        name="auditor",
        voice_directive=(
            "You are a precise, terse assistant. You answer in the minimum "
            "number of words that conveys the full answer. You do not "
            "suggest, recommend, or editorialize. You use no emoji. You do "
            "not add pleasantries. Your output is what a senior engineer "
            "would paste into a ticket comment."
        ),
        verbosity="terse",
        opinion_strength=0.2,
        emoji_policy="none",
        suggestion_aggressiveness=0.0,
        signature=None,
        rules=[
            "Never suggest, recommend, or push back",
            "Tables and lists, not prose",
            "No emoji, no exclamation marks",
        ],
    ),
    "assistant": Persona(
        name="assistant",
        voice_directive=(
            "You are a neutral, factual project management assistant. You "
            "answer the question, add a sentence of useful context, and "
            "stop. You do not editorialize, do not suggest next steps, "
            "and do not address the user by name. Your tone is friendly "
            "but not chatty — closer to a competent colleague than a "
            "cheerleader."
        ),
        verbosity="balanced",
        opinion_strength=0.3,
        emoji_policy="rare",
        suggestion_aggressiveness=0.2,
        signature=None,
        rules=[
            "Answer first, context second",
            "No suggestions unless the user asked",
            "Match the user's language",
        ],
    ),
}


def _builtin(name: str) -> Persona:
    """Look up a built-in persona by name. Falls back to the global default
    if the name is unknown — never raises (callers can't recover from a
    missing persona)."""
    if name in _PERSONAS:
        return _PERSONAS[name]
    fallback = settings.PERSONA_DEFAULT
    if fallback in _PERSONAS:
        logger.warning("persona: unknown persona {}, falling back to {}", name, fallback)
        return _PERSONAS[fallback]
    # Last resort: the first built-in we know about.
    logger.warning("persona: unknown persona {} and bad default {}, using first builtin", name, fallback)
    return next(iter(_PERSONAS.values()))


# Per-(org_slug, user_id) cache. Keeps the resolver off the critical path
# after the first call. Set TTL to 0 in tests to disable.
_cache: dict[tuple[str, str], tuple[Persona, float]] = {}


def _cache_key(org_slug: str, user_id: str) -> tuple[str, str]:
    return (org_slug, user_id)


def resolve_persona(
    org_slug: str,
    user_id: str,
    *,
    override: str | None = None,
) -> Persona:
    """Return the active Persona for this turn.

    Resolution order:
      1. ``override`` arg (used by supervisor when the user said "be brief"
         mid-thread; sourced from memory.store.pending["persona_override"]).
      2. Cache hit for (org_slug, user_id) within the TTL.
      3. Per-user preference (not yet wired — falls through).
      4. Per-org default     (not yet wired — falls through).
      5. ``settings.PERSONA_DEFAULT`` → built-in persona.

    Cached per (org_slug, user_id) for ``settings.PERSONA_CACHE_TTL_SECONDS``
    (default 300s). Tests pass ``settings.PERSONA_CACHE_TTL_SECONDS=0`` to
    disable caching.
    """
    # 1) Conversation override — always wins, never cached.
    if override:
        return _builtin(override)

    key = _cache_key(org_slug, user_id)
    ttl = settings.PERSONA_CACHE_TTL_SECONDS
    cached = _cache.get(key)
    if cached is not None and ttl > 0:
        persona, ts = cached
        if time.time() - ts < ttl:
            return persona

    # 4 + 5 today. (2 + 3 are wired in the future when the user/org
    # preference columns land.)
    persona = _builtin(settings.PERSONA_DEFAULT)

    if ttl > 0:
        _cache[key] = (persona, time.time())
    return persona


def reset_cache() -> None:
    """Clear the resolver cache. Used by tests and by the ``/admin/reset``
    endpoint to force a re-resolve after a persona change."""
    _cache.clear()


__all__ = [
    "EmojiPolicy",
    "Persona",
    "Verbosity",
    "reset_cache",
    "resolve_persona",
]


# Quiet the type-checker about Any — settings is a Pydantic model, but we
# access the cache directly. The Any import keeps the type-checker happy if
# this module is later extended to store arbitrary cache metadata.
_ = Any
