"""Knowledge graph — schema bootstrap, sync, and query helpers."""

from ai_service.graph.queries import (
    get_issue_neighbors,
    get_similar_issues,
    get_smart_assignee,
    get_sprint_issues,
    get_subtask_tree,
    get_user_workload,
)
from ai_service.graph.schema import bootstrap_schema
from ai_service.graph.sync import GraphSyncService, SyncStats

__all__ = [
    "GraphSyncService",
    "SyncStats",
    "bootstrap_schema",
    "get_issue_neighbors",
    "get_similar_issues",
    "get_smart_assignee",
    "get_sprint_issues",
    "get_subtask_tree",
    "get_user_workload",
]
