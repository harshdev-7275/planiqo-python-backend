"""AI agents — the agent abstraction, registry, PM assistant, and tools.

To add a new agent: declare an ``AgentSpec`` and ``register_agent`` it in
``ai_service.agents.registry``. The graph engine and chat endpoint are generic.
"""

from ai_service.agents.base import AgentSpec, GraphFactory, ToolSelector
from ai_service.agents.deps import AgentDeps
from ai_service.agents.pm_agent import (
    AgentState,
    build_graph,
    build_llm,
    get_or_build_graph,
)
from ai_service.agents.prompts import PM_PERSONA_PROMPT
from ai_service.agents.registry import (
    DEFAULT_AGENT,
    PM_SPEC,
    available_agents,
    get_agent_spec,
    register_agent,
)
from ai_service.agents.tools import build_all_tools

__all__ = [
    "DEFAULT_AGENT",
    "PM_PERSONA_PROMPT",
    "PM_SPEC",
    "AgentDeps",
    "AgentSpec",
    "AgentState",
    "GraphFactory",
    "ToolSelector",
    "available_agents",
    "build_all_tools",
    "build_graph",
    "build_llm",
    "get_agent_spec",
    "get_or_build_graph",
    "register_agent",
]
