"""Chat endpoint — POST /v1/chat (and POST /v1/chat/stream).

Takes a user message, runs the PM agent (LangGraph), returns the answer
+ audit trail. Stateless per request in v1.

The streaming variant emits Server-Sent Events (SSE) — `data: {...}\\n\\n`
frames carrying token deltas, tool calls, the final answer, and any error.
A streaming-safe filter strips ``<think>...</think>`` reasoning blocks
mid-stream so they never reach the user.

In `app_env == "development"` a single pretty-printed block is written to the
console summarising the user query, every tool call (with args + a short
result preview), and the AI's final answer. This is dev-only — production keeps
the structured JSON logs so log pipelines can ingest them.
"""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from ai_service.agents import AgentDeps, get_or_build_graph
from ai_service.agents.callbacks import ToolLoggingCallbackHandler
from ai_service.agents.pm_agent import astream_agent
from ai_service.agents.registry import get_agent_spec
from ai_service.config import get_settings
from ai_service.core.errors import ConfigurationError
from ai_service.logging import get_logger
from ai_service.schemas import ChatRequest, ChatResponse, ChatTurn, ToolCallRecord

# Cap how many prior turns we replay into the agent. The client may send up to
# 50 (schema limit); we keep the most recent ones to bound prompt size and cost.
_MAX_HISTORY_TURNS = 20
# Tool result previews in the audit trail are clipped so one huge payload can't
# bloat the response. The schema field is already documented as "~200 chars".
_TOOL_RESULT_PREVIEW_CHARS = 200

# Reasoning models (e.g. MiniMax-M2) emit their chain-of-thought wrapped in
# <think>...</think> inside the message content. We strip it before returning so
# the user never sees raw reasoning. Matches complete blocks, a trailing
# unclosed block (truncated output), and any stray tags.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TRAILING_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


_FALLBACK_UUID = "00000000-0000-0000-0000-000000000000"


def get_agent_deps(request: Request) -> AgentDeps:
    """Build AgentDeps from the trusted service-to-service request.

    ai-service is a PRIVATE service. Its only caller is node-backend (the BFF),
    which authenticates with a static shared secret in ``X-Service-Token`` and
    passes the *already-verified* caller identity in trusted headers:
    ``X-Org-Id``, ``X-Org-Slug``, ``X-User-Id``. ai-service never sees, nor
    trusts, the end user's JWT — node-backend has already verified it.

    When no service token is configured (local dev), the token check is skipped.
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

    # --- Service-token auth (enforced whenever a token is configured) ---
    expected_token = getattr(request.app.state, "service_token", "") or ""
    if expected_token:
        presented = request.headers.get("x-service-token", "")
        if not hmac.compare_digest(presented, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing service token.",
            )

    # --- Trusted identity headers (set by node-backend from the verified JWT) ---
    org_slug    = request.headers.get("x-org-slug")  or getattr(request.app.state, "service_org_slug", "") or "unknown"
    org_id_str  = request.headers.get("x-org-id")    or getattr(request.app.state, "service_org_id", "")  or _FALLBACK_UUID
    user_id_str = request.headers.get("x-user-id")   or getattr(request.app.state, "service_user_id", "") or _FALLBACK_UUID

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

    # Resolve which agent to run (defaults to the PM agent). Unknown names are
    # a client error, not a server error.
    try:
        spec = get_agent_spec(body.agent)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # Scope to the requested project if provided.
    project_id_str: str | None = None
    project_label: str | None = None
    if body.project_id is not None:
        deps.project_id = body.project_id
        project_id_str = str(body.project_id)
        project_label = await _resolve_project_label(deps, body.project_id)

    logger.info(
        "chat.request",
        extra={
            "agent": body.agent or "pm",
            "message_chars": len(body.message),
            "project_id": project_id_str,
            "project_label": project_label,
            "org_slug": deps.org_slug,
            "org_id": str(deps.org_id),
            "user_id": str(deps.user_id),
            "history_turns": len(body.history),
        },
    )

    # Build tools per-request (they capture org_slug from deps). The agent's
    # spec decides which tools it gets. Graph tools are included when Neo4j is
    # configured on app.state.
    neo4j_client = getattr(request.app.state, "neo4j", None)
    tools = spec.select_tools(
        deps.node_backend,
        deps.org_slug,
        neo4j_client=neo4j_client,
        org_id=str(deps.org_id),
        scoped_project_id=project_id_str,
    )
    graph = get_or_build_graph(tools, settings, spec=spec, scoped_project_id=project_id_str)

    model_id = (
        settings.llm_provider
        + ":"
        + (settings.groq_model if settings.llm_provider == "groq" else settings.minimax_model)
    )

    history = _history_to_messages(body.history)
    # Per-turn observability: logs each tool call (incl. Neo4j graph tools) and
    # model round-trip, correlated by request_id.
    callbacks = [ToolLoggingCallbackHandler()]

    try:
        result: dict[str, Any] = await _invoke_async(
            graph, body.message, deps, project_id_str, project_label, history, callbacks
        )
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

    # Pull the final answer (last message in the run), stripped of any reasoning.
    final = result["messages"][-1]
    answer = _strip_reasoning(_content_to_text(getattr(final, "content", "")))

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

    logger.info(
        "chat.completed",
        extra={
            "agent": spec.name,
            "model": model_id,
            "steps": len(result["messages"]),
            "tool_call_count": len(tool_calls),
            "tools_used": [tc.tool for tc in tool_calls],
            "scoped": project_id_str is not None,
            "message_chars": len(body.message),
            "history_turns": len(body.history),
        },
    )

    if settings.app_env == "development":
        _dev_summary(body.message, project_id_str, project_label, tool_calls, answer, model_id)

    return ChatResponse(
        message=answer,
        tool_calls=tool_calls,
        model=model_id,
        steps=len(result["messages"]),
    )


def _history_to_messages(history: list[ChatTurn]) -> list[BaseMessage]:
    """Map the caller-supplied turns to LangChain messages (most recent kept).

    Truncated to the last `_MAX_HISTORY_TURNS` so a long client history can't
    blow up the prompt.
    """
    recent = history[-_MAX_HISTORY_TURNS:]
    messages: list[BaseMessage] = []
    for turn in recent:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


async def _resolve_project_label(deps: AgentDeps, project_id: UUID) -> str:
    """Look up a readable label ("Name (KEY)") for the scoped project.

    Resolved server-side from node-backend so the model sees the real project
    name, not a UUID. Falls back to the bare id if the lookup fails or the
    project can't be found — scoping still works either way.
    """
    try:
        project = await deps.node_backend.get_project(deps.org_slug, project_id)
    except Exception:  # noqa: BLE001 — scoping must not fail the chat; fall back to the id
        logger.warning("chat.project_lookup_failed", extra={"project_id": str(project_id)})
        project = None
    if not project:
        return str(project_id)
    name = project.get("name") or str(project_id)
    key = project.get("key")
    return f"{name} ({key})" if key else str(name)


async def _invoke_async(
    graph: Any,
    message: str,
    deps: AgentDeps,
    project_id_str: str | None,
    project_label: str | None = None,
    history: list[BaseMessage] | None = None,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    from ai_service.agents.pm_agent import arun_agent

    return await arun_agent(
        graph,
        message,
        org_id=str(deps.org_id),
        org_slug=deps.org_slug,
        user_id=str(deps.user_id),
        project_id=project_id_str,
        project_label=project_label,
        history=history,
        callbacks=callbacks,
    )


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model answer.

    Handles complete blocks, a trailing unclosed block (truncated generation),
    and stray tags. If the message was *only* reasoning, returns that text with
    the tags removed rather than an empty string, so the user never sees a blank
    reply.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_TRAILING_RE.sub("", cleaned)
    cleaned = _THINK_TAG_RE.sub("", cleaned).strip()
    if cleaned:
        return cleaned
    return _THINK_TAG_RE.sub("", text).strip()


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


def _dev_summary(
    user_msg: str,
    project_id: str | None,
    project_label: str | None,
    tool_calls: list[ToolCallRecord],
    answer: str,
    model_id: str,
) -> None:
    """Pretty-print a request summary to the console (development only)."""
    import sys

    lines = [
        "",
        "=" * 72,
        f"  USER: {user_msg[:120]}",
        f"  SCOPE: project_id={project_id}  label={project_label}",
        f"  MODEL: {model_id}",
        "-" * 72,
    ]
    for i, tc in enumerate(tool_calls, 1):
        args_str = json.dumps(tc.args, default=str)
        lines.append(f"  [{i}] {tc.tool}({args_str})")
        if tc.result_preview:
            preview = tc.result_preview[:300]
            lines.append(f"      → {preview}")
    lines.append("-" * 72)
    lines.append(f"  ANSWER: {answer[:300]}")
    lines.append("=" * 72)
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Streaming — Server-Sent Events
# ---------------------------------------------------------------------------

_THINK_OPEN_PREFIXES: tuple[str, ...] = (
    "<", "<t", "<th", "<thi", "<thin", "<think",
)


class StreamingReasoningFilter:
    """Stateful filter that strips ``<think>...</think>`` blocks from a token stream.

    Tokens arrive one at a time and a single ``<think>`` tag may straddle two
    consecutive tokens (the model emits ``"<"`` then ``"think>..."``). The
    filter maintains a small pending buffer; while outside a think block, any
    text that could still become a ``<think>`` opener is held back so a stray
    ``<`` is never emitted before the rest of the tag arrives.

    Usage::

        f = StreamingReasoningFilter()
        for delta in stream:
            safe = f.feed(delta)
            if safe:
                emit(safe)
        tail = f.flush()           # any leftover after end-of-stream
    """

    def __init__(self) -> None:
        self._in_think: bool = False
        self._pending: str = ""

    def feed(self, text: str) -> str:
        """Consume one token chunk; return the text that's safe to emit now.

        Empty string is a valid return — the caller should not emit anything
        when the buffer is being held back. Multiple ``feed`` calls may be
        needed to fully resolve a single think block.
        """
        self._pending += text
        return self._drain()

    def flush(self) -> str:
        """Drain any remaining buffered text at end-of-stream.

        If the model never closed a think block, the leading ``<think>`` tag
        is stripped from the remaining text so the user sees the intended
        answer, not raw reasoning — same fallback as ``_strip_reasoning``'s
        "reasoning-only" behaviour.
        """
        out = self._pending
        self._pending = ""
        self._in_think = False
        if out.lower().startswith("<think>"):
            out = out[len("<think>"):]
        return out

    def _drain(self) -> str:
        out_parts: list[str] = []
        while self._pending:
            if self._in_think:
                idx = self._pending.lower().find("</think>")
                if idx == -1:
                    # Inside a think block, nothing to emit — keep waiting.
                    break
                # Drop the think block (and its closing tag) and continue
                # draining whatever's after it.
                self._pending = self._pending[idx + len("</think>"):]
                self._in_think = False
                continue

            idx = self._pending.lower().find("<think>")
            if idx == -1:
                # No think block found. Hold back text that could still
                # become a ``<think>`` opener at a token boundary.
                if self._could_be_partial_open_tag():
                    break
                out_parts.append(self._pending)
                self._pending = ""
                break

            # Emit everything before the think tag, then enter think mode.
            out_parts.append(self._pending[:idx])
            self._pending = self._pending[idx + len("<think>"):]
            self._in_think = True
            # Loop to keep draining the rest of the buffer.

        return "".join(out_parts)

    def _could_be_partial_open_tag(self) -> bool:
        lower = self._pending.lower()
        return any(lower.endswith(p) for p in _THINK_OPEN_PREFIXES)


def _sse_frame(event_type: str, data: dict[str, Any]) -> bytes:
    """Build an SSE frame: ``data: {json}\\n\\n``."""
    payload: dict[str, Any] = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream chat with the PM agent (SSE)",
    description=(
        "Server-Sent Events stream of the agent's response. Each frame is a "
        "JSON object on a `data:` line. Event types: `token` (incremental "
        "answer text), `tool_start` (model invoked a tool), `tool_end` (tool "
        "returned), `done` (final message + audit summary), `error` (the "
        "run failed). `<think>...</think>` reasoning blocks are stripped "
        "mid-stream and never reach the client."
    ),
)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    deps: AgentDeps = Depends(get_agent_deps),
) -> StreamingResponse:
    """Stream the PM agent's response as SSE.

    Same auth, agent selection, and project scoping as ``POST /v1/chat``.
    The body of the response is a sequence of ``data: {...}\\n\\n`` frames;
    see the OpenAPI docstring for the event types.
    """
    settings = get_settings()

    if not settings.llm_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM provider {settings.llm_provider!r} is not configured. "
                f"Set the corresponding API key in env."
            ),
        )

    try:
        spec = get_agent_spec(body.agent)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    project_id_str: str | None = None
    project_label: str | None = None
    if body.project_id is not None:
        deps.project_id = body.project_id
        project_id_str = str(body.project_id)
        project_label = await _resolve_project_label(deps, body.project_id)

    logger.info(
        "chat.stream.request",
        extra={
            "agent": body.agent or "pm",
            "message_chars": len(body.message),
            "project_id": project_id_str,
            "project_label": project_label,
            "org_slug": deps.org_slug,
            "org_id": str(deps.org_id),
            "user_id": str(deps.user_id),
            "history_turns": len(body.history),
        },
    )

    neo4j_client = getattr(request.app.state, "neo4j", None)
    tools = spec.select_tools(
        deps.node_backend,
        deps.org_slug,
        neo4j_client=neo4j_client,
        org_id=str(deps.org_id),
        scoped_project_id=project_id_str,
    )
    graph = get_or_build_graph(tools, settings, spec=spec, scoped_project_id=project_id_str)

    model_id = (
        settings.llm_provider
        + ":"
        + (settings.groq_model if settings.llm_provider == "groq" else settings.minimax_model)
    )

    history = _history_to_messages(body.history)

    async def event_generator() -> AsyncIterator[bytes]:
        reasoning_filter = StreamingReasoningFilter()
        # Track the safe (filter-stripped) accumulated text so the final `done`
        # event carries the cleaned message — `astream_agent`'s `done.message`
        # is the raw concatenation of every token delta, which includes any
        # <think>...</think> content the model emitted.
        accumulated_safe: str = ""
        try:
            async for event in astream_agent(
                graph,
                body.message,
                org_id=str(deps.org_id),
                org_slug=deps.org_slug,
                user_id=str(deps.user_id),
                project_id=project_id_str,
                project_label=project_label,
                history=history,
                model=model_id,
            ):
                kind = event.get("type")
                if kind == "token":
                    safe = reasoning_filter.feed(str(event.get("delta", "")))
                    if safe:
                        accumulated_safe += safe
                        yield _sse_frame("token", {"delta": safe})
                elif kind == "tool_start":
                    yield _sse_frame(
                        "tool_start",
                        {
                            "tool": event.get("tool"),
                            "tool_call_id": event.get("tool_call_id"),
                            "args": event.get("args", {}),
                        },
                    )
                elif kind == "tool_end":
                    yield _sse_frame(
                        "tool_end",
                        {
                            "tool_call_id": event.get("tool_call_id"),
                            "result_preview": event.get("result_preview"),
                        },
                    )
                elif kind == "done":
                    # Flush any pending text the filter was holding back.
                    tail = reasoning_filter.flush()
                    if tail:
                        accumulated_safe += tail
                        yield _sse_frame("token", {"delta": tail})
                    yield _sse_frame(
                        "done",
                        {
                            "message": accumulated_safe,
                            "tool_calls": event.get("tool_calls", []),
                            "model": event.get("model", ""),
                            "steps": event.get("steps", 0),
                        },
                    )
        except GraphRecursionError:
            logger.warning(
                "chat.stream.recursion_limit",
                extra={"message_preview": body.message[:80]},
            )
            yield _sse_frame(
                "error",
                {
                    "code": "RECURSION_LIMIT",
                    "message": (
                        "I wasn't able to finish that one — try narrowing it down "
                        "(name a specific project or issue)."
                    ),
                },
            )
        except Exception:
            logger.exception("chat.stream.error", extra={"message_preview": body.message[:80]})
            yield _sse_frame(
                "error",
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The agent failed to process the request. Try again or rephrase.",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["StreamingReasoningFilter", "get_agent_deps", "router"]
