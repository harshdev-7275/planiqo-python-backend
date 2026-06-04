import re

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from loguru import logger

from agents.state import SupervisorState
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
)
from tools.sprint_tools import AddIssueToSprintTool

_SYSTEM = (
    "You are a project management assistant. "
    "Use the provided tools to list, create, or update issues. "
    "Tools already know the project context — do not ask for org or project details. "
    "Before calling create_issue with an assignee the user mentioned by NAME "
    "(not a user ID), call find_user_by_name first to resolve the name to a "
    "user ID. Never pass a display name as assignee_id — the backend stores it "
    "as a literal string and the assignment silently breaks. "
    "Be concise. Respond in plain text after tool calls complete."
)

# Reasoning models (e.g. MiniMax-M2.7) wrap their final answer in
# <think>…</think> before the user-visible content. Strip it everywhere an
# agent's text surfaces to the user, or the scratchpad leaks into the chat.
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _build_agent(org_slug: str, project_id: str, user_id: str | None):
    api_ctx   = {"api": node_api_client, "org_slug": org_slug, "project_id": project_id, "user_id": user_id}
    graph_ctx = {"neo4j_client": neo4j_client, "org_slug": org_slug, "project_id": project_id}
    tools = [
        GetIssuesTool(**api_ctx),
        GetStatusesTool(**api_ctx),
        CreateIssueTool(**api_ctx, neo4j_client=neo4j_client),
        UpdateIssueStatusTool(**api_ctx),
        AddIssueToSprintTool(**api_ctx),  # lets the model fulfill the "in Sprint X" preview
        FindUserByNameTool(**graph_ctx),  # resolve "Alice" → user id before create_issue
        SuggestAssigneeTool(**graph_ctx),
        FindSimilarIssuesTool(**graph_ctx),
    ]
    return create_react_agent(get_tool(), tools, prompt=_SYSTEM)


async def run(state: SupervisorState, callbacks: Callbacks = None) -> dict:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id")

    logger.info("issue_agent org={} project={} user={}", org_slug, project_id, user_id)

    agent = _build_agent(org_slug, project_id, user_id)
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    response = await agent.ainvoke({"messages": state["messages"]}, config=config)

    last_message = response["messages"][-1]
    raw = last_message.content if hasattr(last_message, "content") else str(last_message)
    content = _strip_think(str(raw))

    logger.info("issue_agent response='{}'", content[:120])
    return {"result": {"message": content}}
