from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from agents.react_utils import finalize, strip_think
from agents.state import SupervisorState
from chains.sanitize import INJECTION_GUARD
from clients.embeddings import embedding_client
from clients.llm_client import get_tool
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from tools.issue_tools import (
    CreateIssueTool,
    FindSimilarIssuesTool,
    FindUserByNameTool,
    GetIssuesTool,
    GetStatusesTool,
    SuggestAssigneeTool,
    UpdateIssueStatusTool,
    UpdateIssueTool,
)
from tools.sprint_tools import AddIssueToSprintTool

_SYSTEM = (
    "You are a project management assistant. "
    "Use the provided tools to list, create, or update issues. "
    "Tools already know the project context — do not ask for org or project details. "
    "Before calling create_issue OR update_issue with an assignee the user "
    "mentioned by NAME (not a user ID), call find_user_by_name first to resolve "
    "the name to a user ID. Never pass a display name as assignee_id — the "
    "backend stores it as a literal string and the assignment silently breaks. "
    "To change an issue's priority, title, or assignee use update_issue with the "
    "issue number (e.g. 5); use update_issue_status only for status changes. "
    "If a tool result begins with 'Failed' or reports an error, tell the user you "
    "couldn't reach the server — do NOT say there are no issues or that none were "
    "found, because you don't actually know. "
    "Be concise. Respond in plain text after tool calls complete. "
    + INJECTION_GUARD
)

# Back-compat alias — the reasoning-scratchpad stripper now lives in
# agents/react_utils so the read agents share one implementation.
_strip_think = strip_think


def _build_agent(
    org_slug: str, project_id: str, user_id: str | None
) -> CompiledStateGraph[Any, None, Any, Any]:
    api = node_api_client
    tools = [
        GetIssuesTool(api=api, org_slug=org_slug, project_id=project_id),
        GetStatusesTool(api=api, org_slug=org_slug, project_id=project_id),
        CreateIssueTool(api=api, org_slug=org_slug, project_id=project_id,
                        user_id=user_id, neo4j_client=neo4j_client,
                        embedding_client=embedding_client),
        # change priority/title/assignee by issue number
        UpdateIssueTool(api=api, org_slug=org_slug, project_id=project_id,
                        user_id=user_id, neo4j_client=neo4j_client,
                        embedding_client=embedding_client),
        UpdateIssueStatusTool(api=api, org_slug=org_slug, project_id=project_id, user_id=user_id),
        # lets the model fulfill the "in Sprint X" preview
        AddIssueToSprintTool(api=api, org_slug=org_slug, project_id=project_id),
        # resolve "Alice" → user id before create_issue
        FindUserByNameTool(neo4j_client=neo4j_client, org_slug=org_slug, project_id=project_id),
        SuggestAssigneeTool(neo4j_client=neo4j_client, org_slug=org_slug, project_id=project_id),
        FindSimilarIssuesTool(neo4j_client=neo4j_client, org_slug=org_slug,
                              project_id=project_id, embedding_client=embedding_client),
    ]
    return create_react_agent(get_tool(), tools, prompt=_SYSTEM)


async def run(state: SupervisorState, callbacks: Callbacks = None) -> dict[str, Any]:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id")

    logger.info("issue_agent org={} project={} user={}", org_slug, project_id, user_id)

    agent = _build_agent(org_slug, project_id, user_id)
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    response = await agent.ainvoke({"messages": state["messages"]}, config=config)

    # finalize strips the reasoning scratchpad AND, if a tool in the trace
    # failed, makes the unreachable-server failure explicit so the user never
    # reads an outage as "no issues found".
    content = finalize(response["messages"])

    logger.info("issue_agent response='{}'", content[:120])
    return {"result": {"message": content}}
