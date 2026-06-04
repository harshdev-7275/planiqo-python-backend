import re
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from agents.state import SupervisorState
from clients.llm_client import get_tool
from clients.node_api import node_api_client
from tools.member_tools import ListMembersTool, MemberIssuesTool

_SYSTEM = (
    "You are a project management assistant. "
    "Use tools to answer questions about team members: list_members for who is "
    "on the team, member_issues to see what a specific person is working on. "
    "Tools already know the project context — do not ask for org or project details. "
    "Be concise. Respond in plain text after tool calls complete."
)

# Reasoning models (e.g. MiniMax-M2.7) wrap their final answer in
# <think>…</think> before the user-visible content. Strip it everywhere an
# agent's text surfaces to the user, or the scratchpad leaks into the chat.
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _build_agent(org_slug: str, project_id: str) -> CompiledStateGraph[Any, None, Any, Any]:
    api = node_api_client
    tools = [
        ListMembersTool(api=api, org_slug=org_slug, project_id=project_id),
        MemberIssuesTool(api=api, org_slug=org_slug, project_id=project_id),
    ]
    return create_react_agent(get_tool(), tools, prompt=_SYSTEM)


async def run(state: SupervisorState, callbacks: Callbacks = None) -> dict[str, Any]:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""

    logger.info("member_agent org={} project={}", org_slug, project_id)

    agent = _build_agent(org_slug, project_id)
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    response = await agent.ainvoke({"messages": state["messages"]}, config=config)

    last_message = response["messages"][-1]
    raw = last_message.content if hasattr(last_message, "content") else str(last_message)
    content = _strip_think(str(raw))

    logger.info("member_agent response='{}'", content[:120])
    return {"result": {"message": content}}
