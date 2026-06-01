from langgraph.prebuilt import create_react_agent
from loguru import logger

from agents.state import SupervisorState
from clients.llm_client import get_tool
from clients.neo4j_client import neo4j_client
from clients.node_api import node_api_client
from tools.issue_tools import (
    CreateIssueTool,
    FindSimilarIssuesTool,
    GetIssuesTool,
    GetStatusesTool,
    SuggestAssigneeTool,
    UpdateIssueStatusTool,
)

_SYSTEM = (
    "You are a project management assistant. "
    "Use the provided tools to list, create, or update issues. "
    "Tools already know the project context — do not ask for org or project details. "
    "Be concise. Respond in plain text after tool calls complete."
)


def _build_agent(org_slug: str, project_id: str, user_id: str | None):
    api_ctx   = {"api": node_api_client, "org_slug": org_slug, "project_id": project_id, "user_id": user_id}
    graph_ctx = {"neo4j_client": neo4j_client, "org_slug": org_slug, "project_id": project_id}
    tools = [
        GetIssuesTool(**api_ctx),
        GetStatusesTool(**api_ctx),
        CreateIssueTool(**api_ctx, neo4j_client=neo4j_client),
        UpdateIssueStatusTool(**api_ctx),
        SuggestAssigneeTool(**graph_ctx),
        FindSimilarIssuesTool(**graph_ctx),
    ]
    return create_react_agent(get_tool(), tools, prompt=_SYSTEM)


async def run(state: SupervisorState) -> dict:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id")

    logger.info("issue_agent org={} project={} user={}", org_slug, project_id, user_id)

    agent = _build_agent(org_slug, project_id, user_id)
    response = await agent.ainvoke({"messages": state["messages"]})

    last_message = response["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info("issue_agent response='{}'", content[:120])
    return {"result": {"message": content}}
