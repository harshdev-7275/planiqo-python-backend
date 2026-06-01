from langgraph.prebuilt import create_react_agent
from loguru import logger

from agents.state import SupervisorState
from clients.llm_client import get_tool
from clients.node_api import node_api_client
from tools.sprint_tools import AddIssueToSprintTool, GetActiveSprintTool, GetSprintsTool

_SYSTEM = (
    "You are a project management assistant. "
    "Use tools to query sprints or add issues to sprints. "
    "Tools already know the project context — do not ask for org or project details. "
    "Be concise. Respond in plain text after tool calls complete."
)


def _build_agent(org_slug: str, project_id: str):
    ctx = {"api": node_api_client, "org_slug": org_slug, "project_id": project_id}
    tools = [
        GetSprintsTool(**ctx),
        GetActiveSprintTool(**ctx),
        AddIssueToSprintTool(**ctx),
    ]
    return create_react_agent(get_tool(), tools, prompt=_SYSTEM)


async def run(state: SupervisorState) -> dict:
    org_slug = state["org_slug"]
    project_id = state.get("project_id") or ""

    logger.info("sprint_agent org={} project={}", org_slug, project_id)

    agent = _build_agent(org_slug, project_id)
    response = await agent.ainvoke({"messages": state["messages"]})

    last_message = response["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info("sprint_agent response='{}'", content[:120])
    return {"result": {"message": content}}
