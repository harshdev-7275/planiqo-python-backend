"""Agent abstraction — the contract every agent must satisfy.

An ``AgentSpec`` is the *only* thing you write to add a new agent. It bundles
the three things that vary between agents:

1. ``prompt``       — the system persona.
2. ``select_tools`` — which tools the agent is given (same signature as
   ``build_all_tools``: ``(client, org_slug, *, neo4j_client, org_id,
   scoped_project_id)``).
3. ``graph_factory`` — *optional*. Override the graph topology. ``None`` means
   "use the default model<->tools react loop in ``pm_agent.build_graph``".

The graph engine, LLM-provider selection, the chat endpoint, auth, the audit
trail, and error handling are all generic and never need to change. Register a
new spec in ``ai_service.agents.registry`` and it is immediately reachable via
the ``agent`` field on ``POST /v1/chat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ai_service.clients.node_backend import NodeBackendClient
    from ai_service.config import Settings
    from ai_service.neo4j import Neo4jClient


class ToolSelector(Protocol):
    """Builds the tool list for an agent, bound to the request's context.

    Matches ``ai_service.agents.tools.build_all_tools`` so that function can be
    used directly as a selector, while bespoke agents can narrow the set.
    """

    def __call__(
        self,
        client: NodeBackendClient,
        org_slug: str,
        *,
        neo4j_client: Neo4jClient | None = ...,
        org_id: str = ...,
        scoped_project_id: str | None = ...,
    ) -> list[Any]: ...


class GraphFactory(Protocol):
    """Compiles a runnable graph from tools + settings + system prompt."""

    def __call__(
        self,
        tools: list[Any],
        settings: Settings,
        *,
        prompt: str,
    ) -> Any: ...


@dataclass(frozen=True)
class AgentSpec:
    """Declarative definition of one agent.

    Attributes:
        name: Stable identifier used by callers (the ``agent`` field on the
            chat request) and as part of the compiled-graph cache key.
        prompt: System persona injected at the top of every model call.
        select_tools: Builds the agent's tools for a given request context.
        description: Human-readable summary (surfaced in errors / docs).
        graph_factory: Optional custom topology. ``None`` uses the default
            react loop (``pm_agent.build_graph``).
    """

    name: str
    prompt: str
    select_tools: ToolSelector
    description: str = ""
    graph_factory: GraphFactory | None = None


__all__ = ["AgentSpec", "GraphFactory", "ToolSelector"]
