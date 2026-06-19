"""Agent observability — a LangChain callback handler that logs every step.

Attached to the agent run (see ``arun_agent``), this emits one structured log
line per tool call and per model round-trip, so a chat turn can be debugged end
to end: which tools fired (REST *and* Neo4j ``graph_*`` tools), with what args,
how long they took, and whether they failed. Every line inherits the request's
``request_id`` (set by the request-logging middleware), so all the steps of one
chat turn share a correlation id.

Tool arguments and a result preview are logged so that debugging scoping and
data-flow issues requires nothing beyond the server console.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from ai_service.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

logger = get_logger("ai_service.agent")

_MAX_LOG_CHARS = 500
_MAX_RESULT_PREVIEW = 500


def _clip(value: Any, limit: int = _MAX_LOG_CHARS) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ToolLoggingCallbackHandler(AsyncCallbackHandler):
    """Logs the agent's tool and model calls for debugging/observability."""

    def __init__(self) -> None:
        self._runs: dict[UUID, tuple[float, str]] = {}

    # --- tools ---------------------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._runs[run_id] = (time.perf_counter(), name)
        args = inputs if inputs is not None else input_str
        logger.info(
            "tool.start",
            extra={"tool": name, "tool_args": _clip(args, _MAX_LOG_CHARS)},
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        duration_ms, name = self._finish(run_id)
        logger.info(
            "tool.end",
            extra={
                "tool": name,
                "duration_ms": duration_ms,
                "result_chars": len(str(output)),
                "result_preview": _clip(output, _MAX_RESULT_PREVIEW),
            },
        )

    async def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        duration_ms, name = self._finish(run_id)
        logger.warning(
            "tool.error",
            extra={
                "tool": name,
                "duration_ms": duration_ms,
                "error": type(error).__name__,
                "error_message": _clip(error),
            },
        )

    # --- model ---------------------------------------------------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._runs[run_id] = (time.perf_counter(), "model")
        message_count = sum(len(batch) for batch in messages)
        logger.info("model.start", extra={"message_count": message_count})

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        duration_ms, _ = self._finish(run_id)
        logger.info("model.end", extra={"duration_ms": duration_ms})

    # --- helpers -------------------------------------------------------------

    def _finish(self, run_id: UUID) -> tuple[float | None, str]:
        started = self._runs.pop(run_id, None)
        if started is None:
            return None, "unknown"
        start, name = started
        return round((time.perf_counter() - start) * 1000, 2), name


__all__ = ["ToolLoggingCallbackHandler"]
