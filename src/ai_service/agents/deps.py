"""Agent dependencies — passed to the LangGraph graph at invoke time.

The ai-service does NOT connect to databases directly. AgentDeps carries
the HTTP client to node-backend, plus the org/project/user context. The
LangGraph graph state holds these fields and reads from them in tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from ai_service.clients.node_backend import NodeBackendClient


@dataclass
class AgentDeps:
    """Dependencies injected into the agent state at invoke time.

    Attributes:
        node_backend: HTTP client for the node-backend. Tools call this.
        org_id: Organization the user belongs to (tenant isolation).
        org_slug: URL slug of the org (used in node-backend paths).
        project_id: Optional project scope. If set, tools prefer this project.
        user_id: The user asking the question.
    """

    node_backend: NodeBackendClient
    org_id: UUID
    org_slug: str
    project_id: UUID | None
    user_id: UUID


__all__ = ["AgentDeps"]
