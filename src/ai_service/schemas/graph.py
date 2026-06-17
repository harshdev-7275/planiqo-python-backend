"""Pydantic schemas for graph sync and stats endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class GraphSyncRequest(BaseModel):
    """Body for POST /v1/graph/sync — trigger a full org sync."""

    org_slug: str = Field(..., min_length=1, description="Organisation URL slug.")
    org_id: str = Field(..., min_length=1, description="Organisation UUID.")


class GraphProjectSyncRequest(BaseModel):
    """Body for POST /v1/graph/sync/project/:projectId — incremental project sync."""

    org_slug: str = Field(..., min_length=1, description="Organisation URL slug.")
    org_id: str = Field(..., min_length=1, description="Organisation UUID.")
    project_id: UUID = Field(..., description="Project UUID to sync.")


class GraphSyncResponse(BaseModel):
    """Stats returned after a sync run."""

    projects: int
    categories: int
    sprints: int
    issues: int
    users: int
    relationships: int
    errors: list[str] = Field(default_factory=list)


class GraphStatsResponse(BaseModel):
    """Node and relationship counts for each label/type."""

    nodes: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)
    configured: bool = True


__all__ = [
    "GraphProjectSyncRequest",
    "GraphStatsResponse",
    "GraphSyncRequest",
    "GraphSyncResponse",
]
