"""Graph-based agent tools — query the Neo4j knowledge graph.

These tools are only included in the agent's tool list when Neo4j is
configured (i.e., `app.state.neo4j` is not None). They provide relationship-
aware answers that REST-based tools cannot: "who should own this?", "what
else is in this sprint?", "show me the subtask tree".
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ai_service.neo4j import Neo4jClient


def make_graph_tools(
    neo4j_client: Neo4jClient,
    org_id: str,
    scoped_project_id: str | None = None,
) -> list[Any]:
    """Return knowledge-graph tools bound to the given Neo4j client and org.

    When `scoped_project_id` is set, the org-wide tools (similar issues, user
    workload) are filtered to that project so nothing leaks across projects.
    """

    @tool
    async def graph_suggest_assignee(
        issue_id: str,
        issue_title: str,
        issue_description: str = "",
    ) -> list[dict[str, Any]]:
        """Suggest the best assignees for an issue using the knowledge graph.

        Ranks users by how many similar issues they have resolved in the same
        category (x3 weight) and how often they have commented on related issues
        (x1 weight). Applies a workload penalty for users with many open issues.

        Args:
            issue_id: UUID of the issue to find assignees for.
            issue_title: Title of the issue (used for context, not directly queried).
            issue_description: Optional description (for context).

        Returns:
            List of dicts with user_id, name, resolved_count, comment_count,
            open_issues, score. Ordered by score descending. Empty if no graph
            data exists.
        """
        from ai_service.graph.queries import get_smart_assignee

        return await get_smart_assignee(neo4j_client, issue_id, org_id)

    @tool
    async def graph_similar_issues(
        keywords: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find issues whose title or description contains any of the keywords.

        Useful for detecting duplicate issues or finding related prior work.
        Case-insensitive substring matching.

        Args:
            keywords: List of keywords to search for (e.g. ["auth", "login"]).
            limit: Max results to return (1-50, default 10).

        Returns:
            List of dicts with issue_id, title, type, priority, completed_at,
            category_name. Ordered by creation date descending.
        """
        from ai_service.graph.queries import get_similar_issues

        if limit < 1 or limit > 50:
            limit = 10
        return await get_similar_issues(
            neo4j_client, keywords, org_id, limit=limit, project_id=scoped_project_id
        )

    @tool
    async def graph_related_issues(issue_id: str) -> list[dict[str, Any]]:
        """Find issues in the same category as the given issue.

        Also shows whether each neighbor is in the same sprint. Useful for
        understanding the scope of work around a specific issue.

        Args:
            issue_id: UUID of the issue.

        Returns:
            List of neighbor issues with in_same_sprint flag.
        """
        from ai_service.graph.queries import get_issue_neighbors

        return await get_issue_neighbors(neo4j_client, issue_id)

    @tool
    async def graph_sprint_progress(sprint_id: str) -> list[dict[str, Any]]:
        """Return all issues in a sprint with their assignee and completion status.

        Args:
            sprint_id: UUID of the sprint.

        Returns:
            List of issues with issue_id, title, type, priority, status_id,
            completed_at, assignee_id, assignee_name.
        """
        from ai_service.graph.queries import get_sprint_issues

        return await get_sprint_issues(neo4j_client, sprint_id)

    @tool
    async def graph_user_workload(user_id: str = "") -> list[dict[str, Any]]:
        """Show open-issue counts for users in the org.

        If user_id is provided, returns only that user's count. Otherwise
        returns all members sorted by workload descending.

        Args:
            user_id: Optional UUID of a specific user. Empty string returns all.

        Returns:
            List of dicts with user_id, name, open_issues.
        """
        from ai_service.graph.queries import get_user_workload

        return await get_user_workload(
            neo4j_client,
            org_id,
            user_id=user_id if user_id else None,
            project_id=scoped_project_id,
        )

    @tool
    async def graph_user_activity(limit: int = 10) -> list[dict[str, Any]]:
        """Rank org members by how much they've changed or commented on issues.

        Uses the knowledge graph's CHANGED (edit history) and COMMENTED_ON
        activity to answer "who's been most active?" or "who has made the most
        changes recently?". When the chat is scoped to a project, only that
        project's issues are counted.

        Args:
            limit: Max users to return (1-50, default 10).

        Returns:
            List of dicts with user_id, name, change_count, comment_count,
            total_activity. Ordered by total_activity descending. Empty if no
            activity data exists in the graph.
        """
        from ai_service.graph.queries import get_user_activity

        if limit < 1 or limit > 50:
            limit = 10
        return await get_user_activity(
            neo4j_client, org_id, limit=limit, project_id=scoped_project_id
        )

    @tool
    async def graph_subtask_tree(issue_id: str) -> list[dict[str, Any]]:
        """Return all subtasks reachable from a parent issue (recursive up to 10 levels).

        Args:
            issue_id: UUID of the parent issue.

        Returns:
            List of subtask dicts with depth indicating nesting level (1 = direct child).
        """
        from ai_service.graph.queries import get_subtask_tree

        return await get_subtask_tree(neo4j_client, issue_id)

    return [
        graph_suggest_assignee,
        graph_similar_issues,
        graph_related_issues,
        graph_sprint_progress,
        graph_user_workload,
        graph_user_activity,
        graph_subtask_tree,
    ]


__all__ = ["make_graph_tools"]
