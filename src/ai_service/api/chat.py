"""Chat endpoint — POST /v1/chat.

Takes a user message, runs the PM agent (LangGraph), returns the answer
+ audit trail. Stateless per request in v1.

In `app_env == "development"` a single pretty-printed block is written to the
console summarising the user query, every tool call (with args + a short
result preview), and the AI's final answer. This is dev-only — production keeps
the structured JSON logs so log pipelines can ingest them.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from ai_service.agents import AgentDeps, build_all_tools, get_or_build_graph
from ai_service.config import get_settings
from ai_service.core.errors import ConfigurationError
from ai_service.logging import get_logger
from ai_service.schemas import ChatRequest, ChatResponse, ToolCallRecord

# Width of the dev-mode console block (chars between the border rules).
_DEV_LOG_WIDTH = 72
# Tool result previews in the dev log are clipped so one huge payload can't
# flood the console. The schema field is already documented as "~200 chars".
_TOOL_RESULT_PREVIEW_CHARS = 200


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Base64-decode a JWT payload section without signature verification."""
    try:
        part = token.split(".")[1]
        padding = 4 - len(part) % 4
        if padding != 4:
            part += "=" * padding
        return json.loads(base64.urlsafe_b64decode(part))  # type: ignore[no-any-return]
    except (IndexError, ValueError, json.JSONDecodeError, binascii.Error):
        return {}

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


_FALLBACK_UUID = "00000000-0000-0000-0000-000000000000"


def get_agent_deps(request: Request) -> AgentDeps:
    """Build AgentDeps from the incoming request JWT.

    The frontend sends the user's JWT in Authorization: Bearer <token>.
    That JWT now contains orgId + orgSlug (embedded at login by node-backend).
    We decode it here (no signature verification — ai-service trusts its own
    frontend; auth enforcement happens inside node-backend via AI_SERVICE_TOKEN).
    """
    client = getattr(request.app.state, "node_backend", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "node-backend client is not initialized. "
                "Set NODE_BACKEND_URL and NODE_BACKEND_SERVICE_TOKEN in env."
            ),
        )

    # Decode the user JWT from the Authorization header.
    auth_header = request.headers.get("authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    payload = _decode_jwt_payload(token) if token else {}

    org_slug   = payload.get("orgSlug")   or getattr(request.app.state, "service_org_slug", "") or "unknown"
    org_id_str = payload.get("orgId")     or getattr(request.app.state, "service_org_id", "")  or _FALLBACK_UUID
    user_id_str = payload.get("userId")   or getattr(request.app.state, "service_user_id", "") or _FALLBACK_UUID

    try:
        org_id  = UUID(org_id_str)
        user_id = UUID(user_id_str)
    except ValueError:
        org_id  = UUID(_FALLBACK_UUID)
        user_id = UUID(_FALLBACK_UUID)

    return AgentDeps(
        node_backend=client,
        org_id=org_id,
        org_slug=org_slug,
        project_id=None,
        user_id=user_id,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the PM agent a question",
    description="Stateless chat with the PM agent. The agent uses tools to query the project database via node-backend.",
)
async def chat(
    request: Request,
    body: ChatRequest,
    deps: AgentDeps = Depends(get_agent_deps),
) -> ChatResponse:
    """Run a single turn of the PM agent and return the answer."""
    settings = get_settings()

    if not settings.llm_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM provider {settings.llm_provider!r} is not configured. "
                f"Set the corresponding API key in env."
            ),
        )

    # Scope to the requested project if provided.
    project_id_str: str | None = None
    if body.project_id is not None:
        deps.project_id = body.project_id
        project_id_str = str(body.project_id)

    # Build tools per-request (they capture org_slug from deps).
    # Graph tools are included when Neo4j is configured on app.state.
    neo4j_client = getattr(request.app.state, "neo4j", None)
    tools = build_all_tools(
        deps.node_backend,
        deps.org_slug,
        neo4j_client=neo4j_client,
        org_id=str(deps.org_id),
        scoped_project_id=project_id_str,
    )
    graph = get_or_build_graph(tools, settings)

    model_id = (
        settings.llm_provider
        + ":"
        + (settings.groq_model if settings.llm_provider == "groq" else settings.minimax_model)
    )

    try:
        result: dict[str, Any] = await _invoke_async(graph, body.message, deps, project_id_str)
    except ConfigurationError as exc:
        logger.exception("chat.config_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except GraphRecursionError:
        # The model looped without converging (often an empty knowledge graph
        # or a question too broad to answer with the available tools). Return a
        # graceful message instead of a 502 so the user can rephrase.
        logger.warning("chat.recursion_limit", extra={"message_preview": body.message[:80]})
        return ChatResponse(
            message=(
                "I wasn't able to finish that one — I kept gathering data without "
                "reaching an answer. Try narrowing it down (name a specific project "
                "or issue), and make sure the knowledge graph has been synced for "
                "graph-based questions."
            ),
            tool_calls=[],
            model=model_id,
            steps=0,
        )
    except Exception:
        logger.exception("chat.agent_error", extra={"message_preview": body.message[:80]})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The agent failed to process the request. Try again or rephrase.",
        ) from None

    # Pull the final answer (last message in the run).
    final = result["messages"][-1]
    answer = _content_to_text(getattr(final, "content", ""))

    # Build a tool_call_id -> result preview map so each audit row can show
    # what the tool returned (clipped to keep the response payload small).
    tool_results: dict[str, str] = {}
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            preview = _content_to_text(getattr(msg, "content", ""))
            if len(preview) > _TOOL_RESULT_PREVIEW_CHARS:
                preview = preview[: _TOOL_RESULT_PREVIEW_CHARS - 1] + "…"
            tool_results[str(msg.tool_call_id)] = preview

    # Extract the tool-call audit trail from all AI messages.
    tool_calls: list[ToolCallRecord] = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCallRecord(
                        tool=tc.get("name", "unknown"),
                        args=dict(tc.get("args", {}) or {}),
                        result_preview=tool_results.get(str(tc.get("id", ""))),
                    )
                )

    # Dev-only: write a single pretty-printed block summarising the turn.
    # Production keeps the structured JSON logs for ingestion.
    if settings.app_env == "development":
        logger.info(
            _format_chat_log(
                user_query=body.message,
                tool_calls=tool_calls,
                answer=answer,
                model=model_id,
                steps=len(result["messages"]),
                scope=project_id_str,
            )
        )

    return ChatResponse(
        message=answer,
        tool_calls=tool_calls,
        model=model_id,
        steps=len(result["messages"]),
    )


async def _invoke_async(
    graph: Any,
    message: str,
    deps: AgentDeps,
    project_id_str: str | None,
) -> dict[str, Any]:
    from ai_service.agents.pm_agent import arun_agent

    return await arun_agent(
        graph,
        message,
        org_id=str(deps.org_id),
        org_slug=deps.org_slug,
        user_id=str(deps.user_id),
        project_id=project_id_str,
    )


def _content_to_text(content: Any) -> str:
    """LangChain message content can be a string or a list of blocks. Coerce to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Dev-mode console formatter
# ---------------------------------------------------------------------------


def _format_chat_log(
    *,
    user_query: str,
    tool_calls: list[ToolCallRecord],
    answer: str,
    model: str,
    steps: int,
    scope: str | None = None,
) -> str:
    """Build a single multi-line, pretty-printed summary of one chat turn.

    Used by the dev console. Production skips this entirely and relies on the
    structured JSON logs configured in `ai_service.logging`.

    The output is intentionally one string so it goes through the standard
    logger (preserving level, handler, propagation) rather than `print()`.
    """
    rule = "─" * _DEV_LOG_WIDTH
    indent = "  "
    lines: list[str] = [rule, "💬 CHAT REQUEST"]

    query = user_query.replace("\n", " ").strip()
    lines.append(f"{indent}Query : {query}")
    lines.append(f"{indent}Scope : {scope if scope else 'all projects'}")
    lines.append(f"{indent}Model : {model}")
    lines.append(f"{indent}Steps : {steps}")
    lines.append("")

    if tool_calls:
        lines.append(f"🔧 TOOL CALLS ({len(tool_calls)})")
        for index, tc in enumerate(tool_calls, start=1):
            lines.append(f"{indent}[{index}] {tc.tool}")
            if tc.args:
                args_repr = ", ".join(f"{k}={v!r}" for k, v in tc.args.items())
                lines.append(f"{indent}    args   : {args_repr}")
            if tc.result_preview:
                preview_one_line = tc.result_preview.replace("\n", " ⏎ ")
                lines.append(f"{indent}    result : {preview_one_line}")
            lines.append("")
    else:
        lines.append("🔧 TOOL CALLS: (none)")
        lines.append("")

    lines.append("🤖 AI RESPONSE")
    answer_text = answer.strip() or "(empty)"
    for chunk in answer_text.split("\n"):
        lines.append(f"{indent}{chunk}")

    lines.append(rule)
    return "\n".join(lines)


__all__ = ["_format_chat_log", "get_agent_deps", "router"]
