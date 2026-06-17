"""PM agent — LangGraph StateGraph with PM persona + tool calling.

The agent is a custom StateGraph (not prebuilt create_react_agent) so we
own the loop: model -> tool -> model -> ... -> end. State is a TypedDict
that extends MessagesState with our org/project/user context.

LLM providers supported:
- "groq"   — langchain_groq.ChatGroq
- "minimax" — langchain_openai.ChatOpenAI pointed at MiniMax's OpenAI-compatible endpoint
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ai_service.agents.prompts import PM_PERSONA_PROMPT
from ai_service.config import Settings, get_settings

if TYPE_CHECKING:
    from ai_service.agents.base import AgentSpec

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
    # Human-readable label of the scoped project (e.g. "New Project (NP)"),
    # injected into the system prompt so the model knows the active scope. None
    # when the conversation is not scoped to a single project.
    project_label: str | None
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


def call_model(
    state: AgentState,
    llm_with_tools: BaseChatModel,
    prompt: str = PM_PERSONA_PROMPT,
) -> dict[str, Any]:
    """Run the LLM. Returns the new message(s) to merge into state.

    `prompt` is the agent's system persona — supplied by the agent's
    `AgentSpec` so the same loop serves any agent.

    When the run is scoped to a single project (`project_label` is set in
    state), a scope notice is appended so the model knows which project it is
    working in and does not ask the user for a project id it already has.
    """
    content = prompt
    label = state.get("project_label")
    if label:
        content += (
            "\n\n## Current scope\n"
            f"This conversation is scoped to the project {label}. Every tool is "
            "already restricted to this project, so never ask the user for a "
            "project id or project name — assume all questions are about this "
            "project, and do not reference or compare against any other project."
        )
    system = SystemMessage(content=content)
    response = llm_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}


def build_graph(tools: list[Any], settings: Settings, *, prompt: str = PM_PERSONA_PROMPT) -> Any:
    """Compile a LangGraph StateGraph that loops between model and tools.

    `prompt` is the agent persona injected on every model call. Defaults to the
    PM persona so the function is usable standalone, but the registry passes
    each agent's own prompt.

    Returns a compiled graph with `.ainvoke(state)` for async execution.
    """
    llm = build_llm(settings)
    # Bind tools to the LLM so it knows what to call.
    llm_with_tools = cast("BaseChatModel", llm.bind_tools(tools))

    def _call(state: AgentState) -> dict[str, Any]:
        return call_model(state, llm_with_tools, prompt)

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


def _seed_messages(
    message: str, history: list[BaseMessage] | None
) -> list[BaseMessage]:
    """Prior turns (oldest first) followed by the new user message."""
    return [*(history or []), HumanMessage(content=message)]


def run_agent(
    graph: Any,
    message: str,
    *,
    org_id: str,
    org_slug: str,
    user_id: str,
    project_id: str | None = None,
    project_label: str | None = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Invoke the agent graph and return the final state.

    `history` is the prior conversation (already converted to LangChain
    messages, oldest first). It seeds the run so the agent has multi-turn
    context — the endpoint itself stays stateless.

    `project_label` is the human-readable name of the scoped project; when set
    it is surfaced to the model via the system prompt.

    Returns the full state dict so callers can extract the final message,
    tool calls, and any other state they care about.
    """
    state: AgentState = {
        "messages": _seed_messages(message, history),
        "org_id": org_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "project_label": project_label,
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
    project_label: str | None = None,
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Async version of run_agent."""
    state: AgentState = {
        "messages": _seed_messages(message, history),
        "org_id": org_id,
        "org_slug": org_slug,
        "project_id": project_id,
        "project_label": project_label,
        "user_id": user_id,
    }
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT}),
    )


# Module-level graph placeholder; tests should call build_graph with
# a controlled list of tools and settings.
_graph_cache: dict[Any, Any] = {}


def get_or_build_graph(
    tools: list[Any],
    settings: Settings | None = None,
    *,
    spec: AgentSpec | None = None,
) -> Any:
    """Get the cached graph or build one for the given agent.

    Keyed by provider, model, agent name, and tool identity so different
    agents (and stale tool lists) never collide in the cache.

    `spec` selects the agent's prompt and (optionally) a custom graph topology.
    Defaults to the PM agent so existing callers keep working.
    """
    settings = settings or get_settings()
    if spec is None:
        from ai_service.agents.registry import get_agent_spec

        spec = get_agent_spec()
    key: tuple[Any, ...] = (
        settings.llm_provider,
        settings.groq_model if settings.llm_provider == "groq" else settings.minimax_model,
        spec.name,
        id(tuple(tools)),
    )
    if key not in _graph_cache:
        if spec.graph_factory is not None:
            _graph_cache[key] = spec.graph_factory(tools, settings, prompt=spec.prompt)
        else:
            _graph_cache[key] = build_graph(tools, settings, prompt=spec.prompt)
    return _graph_cache[key]


__all__ = [
    "AgentState",
    "arun_agent",
    "build_graph",
    "build_llm",
    "get_or_build_graph",
    "run_agent",
]
