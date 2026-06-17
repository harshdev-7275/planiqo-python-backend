"""Agent registry — the single place agents are declared and looked up.

Adding a new agent is three lines: write a prompt, declare its tools, and
register an ``AgentSpec`` here. Nothing in the graph engine or the chat
endpoint needs to change.

    from ai_service.agents.base import AgentSpec
    from ai_service.agents.registry import register_agent

    register_agent(AgentSpec(
        name="triage",
        prompt=TRIAGE_PROMPT,
        select_tools=make_triage_tools,   # or build_all_tools
        description="Triages incoming issues.",
    ))
"""

from __future__ import annotations

from ai_service.agents.base import AgentSpec
from ai_service.agents.prompts import PM_PERSONA_PROMPT
from ai_service.agents.tools import build_all_tools

# The default agent — the general-purpose PM assistant.
DEFAULT_AGENT = "pm"

PM_SPEC = AgentSpec(
    name=DEFAULT_AGENT,
    prompt=PM_PERSONA_PROMPT,
    select_tools=build_all_tools,
    description="Planiqo Assistant — general-purpose project-management assistant.",
)


_REGISTRY: dict[str, AgentSpec] = {}


def register_agent(spec: AgentSpec) -> None:
    """Add (or replace) an agent in the registry, keyed by ``spec.name``."""
    _REGISTRY[spec.name] = spec


def get_agent_spec(name: str | None = None) -> AgentSpec:
    """Resolve an agent by name, defaulting to the PM agent.

    Raises:
        KeyError: if ``name`` is not a registered agent. The chat endpoint
            translates this into a 400 so callers get a clear message.
    """
    resolved = name or DEFAULT_AGENT
    try:
        return _REGISTRY[resolved]
    except KeyError as exc:
        raise KeyError(
            f"Unknown agent {resolved!r}. Available: {', '.join(available_agents())}."
        ) from exc


def available_agents() -> list[str]:
    """Sorted list of registered agent names (for errors and discovery)."""
    return sorted(_REGISTRY)


# Register the built-in agents at import time.
register_agent(PM_SPEC)


__all__ = [
    "DEFAULT_AGENT",
    "PM_SPEC",
    "available_agents",
    "get_agent_spec",
    "register_agent",
]
