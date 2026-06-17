"""PM agent — LangGraph StateGraph with PM persona + tool calling.

The agent is a custom StateGraph (not prebuilt create_react_agent) so we
own the loop: model -> tool -> model -> ... -> end. State is a TypedDict
that extends MessagesState with our org/project/user context.

LLM providers supported:
- "groq"   — langchain_groq.ChatGroq
- "minimax" — langchain_openai.ChatOpenAI pointed at MiniMax's OpenAI-compatible endpoint
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ai_service.agents.prompts import PM_PERSONA_PROMPT
from ai_service.config import Settings, get_settings

# Max model<->tool loops before LangGraph aborts. The default (25) is easily
# hit when a tool returns nothing and the model keeps probing other tools
# (e.g. an empty knowledge graph). 40 gives genuine multi-project questions
# room while still bounding runaway loops.
_RECURSION_LIMIT = 40


class AgentState(TypedDict, total=False):
    """State carried through the agent's graph execution.

    `messages` is accumulated by the `add_messages` reducer (new messages are
    appended, not replaced). The other fields are static for the duration of
    a single run.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    org_id: str
    org_slug: str
    project_id: str | None
    user_id: str


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------


def build_llm(settings: Settings) -> BaseChatModel:
    """Construct the LLM client for the configured provider.

    Multi-provider via a single function. Each provider's library is
    imported lazily so unused deps don't bloat the import graph.
    """
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return ChatGroq(
            model_name=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_retries=2,
        )
    if settings.llm_provider == "minimax":
        from langchain_openai import ChatOpenAI

        if not settings.minimax_api_key:
            raise ValueError("MINIMAX_API_KEY is required when LLM_PROVIDER=minimax")
        return ChatOpenAI(
            model=settings.minimax_model,
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_base_url,
            temperature=settings.llm_temperature,
            max_retries=2,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Decide whether to call tools again or finish.

    If the last AI message has tool_calls, route to the tools node. Otherwise
    we're done — return the final answer.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


def call_model(state: AgentState, llm_with_tools: BaseChatModel) -> dict[str, Any]:
    """Run the LLM. Returns the new message(s) to merge into state."""
    system = SystemMessage(content=PM_PERSONA_PROMPT)
    response = llm_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}


def build_graph(tools: list[Any], settings: Settings) -> Any:
    """Compile a LangGraph StateGraph that loops between model and tools.

    Returns a compiled graph with `.ainvoke(state)` for async execution.
    """
    llm = build_llm(settings)
    # Bind tools to the LLM so it knows what to call.
    llm_with_tools = cast("BaseChatModel", llm.bind_tools(tools))

    def _call(state: AgentState) -> dict[str, Any]:
        return call_model(state, llm_with_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _call)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def run_agent(
    graph: Any,
    message: str,
    *,
    org_id: str,
    org_slug: str,
    user_id: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the agent graph and return the final state.

    Returns the full state dict so callers can extract the final message,
    tool calls, and any other state they care about.
    """
    state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "org_id": org_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "user_id": user_id,
    }
    return cast(
        "dict[str, Any]",
        graph.invoke(state, config={"recursion_limit": _RECURSION_LIMIT}),
    )


async def arun_agent(
    graph: Any,
    message: str,
    *,
    org_id: str,
    org_slug: str,
    user_id: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Async version of run_agent."""
    state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "org_id": org_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "user_id": user_id,
    }
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT}),
    )


# Module-level graph placeholder; tests should call build_graph with
# a controlled list of tools and settings.
_graph_cache: dict[Any, Any] = {}


def get_or_build_graph(tools: list[Any], settings: Settings | None = None) -> Any:
    """Get the cached graph or build one. Keyed by tool id to avoid stale tools."""
    settings = settings or get_settings()
    key: tuple[Any, ...] = (
        settings.llm_provider,
        settings.groq_model if settings.llm_provider == "groq" else settings.minimax_model,
        id(tuple(tools)),
    )
    if key not in _graph_cache:
        _graph_cache[key] = build_graph(tools, settings)
    return _graph_cache[key]


__all__ = [
    "AgentState",
    "arun_agent",
    "build_graph",
    "build_llm",
    "get_or_build_graph",
    "run_agent",
]
