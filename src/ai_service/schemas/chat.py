"""Chat endpoint request/response schemas."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """User's question to the PM agent."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or instruction for the agent.",
    )
    project_id: UUID | None = Field(
        default=None,
        description="Optional project scope. If set, the agent prefers this project.",
    )
    agent: str | None = Field(
        default=None,
        max_length=64,
        description="Which agent to run. Defaults to the general-purpose PM agent.",
    )


class ToolCallRecord(BaseModel):
    """Record of a single tool the agent invoked during a turn."""

    tool: str = Field(..., description="Tool function name.")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed.")
    result_preview: str | None = Field(
        default=None, description="First ~200 chars of the tool's result for the audit log."
    )


class ChatResponse(BaseModel):
    """Agent's reply to a chat request."""

    message: str = Field(..., description="The PM agent's natural-language answer.")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Tools the agent invoked while answering (audit trail).",
    )
    model: str = Field(..., description="LLM model used.")
    steps: int = Field(..., description="Number of agent steps (LLM turns + tool calls).")


__all__ = ["ChatRequest", "ChatResponse", "ToolCallRecord"]
