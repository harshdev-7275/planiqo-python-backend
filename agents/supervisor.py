from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from agents.state import SupervisorState
from chains.intent import classify
from models.intents import Intent, IntentResult
import agents.issue_agent as _issue_agent


# --- nodes ---

async def classify_intent(state: SupervisorState) -> dict:
    last = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    intent_result = await classify(str(last))
    logger.info("supervisor intent={} confidence={:.2f}", intent_result.intent.value, intent_result.confidence)
    return {"intent": intent_result, "next": _route_intent(intent_result)}


def handle_unknown(state: SupervisorState) -> dict:
    return {
        "result": {"message": "I didn't understand that. Try asking about issues, sprints, or team members."},
        "next": END,
    }


def _placeholder(agent_name: str):
    def _node(state: SupervisorState) -> dict:
        return {"result": {"message": f"{agent_name} not yet implemented"}}
    _node.__name__ = agent_name
    return _node


# --- routing ---

def _route_intent(intent_result: IntentResult) -> str:
    match intent_result.intent:
        case Intent.CREATE_ISSUE:  return "issue_agent"
        case Intent.QUERY_ISSUES:  return "issue_agent"
        case Intent.UPDATE_ISSUE:  return "issue_agent"
        case Intent.QUERY_SPRINT:  return "sprint_agent"
        case Intent.CREATE_SPRINT: return "sprint_agent"
        case Intent.QUERY_MEMBER:  return "member_agent"
        case Intent.SUMMARIZE:     return "summarize_agent"
        case Intent.TEAMS_CONTEXT: return "teams_agent"
        case _:                    return "handle_unknown"


def route(state: SupervisorState) -> str:
    return state["next"]


# --- graph ---

_PLACEHOLDER_AGENTS = [
    "sprint_agent",
    "member_agent",
    "summarize_agent",
    "teams_agent",
]

_builder = StateGraph(SupervisorState)
_builder.add_node("classify_intent", classify_intent)
_builder.add_node("handle_unknown", handle_unknown)
_builder.add_node("issue_agent", _issue_agent.run)

for _name in _PLACEHOLDER_AGENTS:
    _builder.add_node(_name, _placeholder(_name))

_builder.add_edge(START, "classify_intent")
_builder.add_conditional_edges("classify_intent", route)

_builder.add_edge("issue_agent", END)
for _name in _PLACEHOLDER_AGENTS:
    _builder.add_edge(_name, END)

_builder.add_edge("handle_unknown", END)

graph = _builder.compile()


# --- entry point ---

async def run(
    message: str,
    user_id: str,
    org_slug: str,
    project_id: str | None = None,
) -> dict:
    initial: SupervisorState = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "intent": None,
        "result": None,
        "error": None,
        "next": "",
    }
    state = await graph.ainvoke(initial)
    intent = state.get("intent")
    return {
        "intent": intent.intent.value if intent else None,
        "result": state.get("result"),
        "error": state.get("error"),
    }
