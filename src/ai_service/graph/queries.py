"""Structured Cypher query helpers for the knowledge graph.

All functions take a `Neo4jClient` and return plain dicts so they can be
called from both agent tools and HTTP endpoints without coupling.

Smart-assignee scoring formula (per user):
    raw_score = resolved_same_type * 3.0 + comment_count * 1.0
    score     = raw_score / (1.0 + open_issues * 0.5)

The workload penalty (open_issues * 0.5) discounts users who are already
heavily loaded without completely excluding them.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_service.neo4j import Neo4jClient

logger = logging.getLogger(__name__)


async def get_smart_assignee(
    client: Neo4jClient,
    issue_id: str,
    org_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return ranked assignee suggestions for an issue.

    Scores users by:
      - How many similar (same category + type) issues they have resolved  (x3)
      - How many times they have commented on issues in the same category  (x1)
    Penalised by current open-issue count.

    Returns an empty list if the issue or its category is not in the graph.
    """
    cypher = """
MATCH (target:Issue {id: $issue_id})-[:IN_CATEGORY]->(cat:Category)
WITH target, cat, target.type AS issue_type

MATCH (u:User)-[:MEMBER_OF]->(org:Org {id: $org_id})

OPTIONAL MATCH (u)-[:RESOLVED]->(r:Issue)-[:IN_CATEGORY]->(cat)
WHERE r.type = issue_type
WITH u, cat, count(r) AS resolved_count

OPTIONAL MATCH (u)-[c:COMMENTED_ON]->(cm:Issue)-[:IN_CATEGORY]->(cat)
WITH u, resolved_count, sum(coalesce(c.count, 0)) AS comment_count

OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(open:Issue)
WHERE open.completedAt IS NULL
WITH u, resolved_count, comment_count, count(open) AS open_issues

WITH u, resolved_count, comment_count, open_issues,
     (toFloat(resolved_count) * 3.0 + toFloat(comment_count)) /
     (1.0 + toFloat(open_issues) * 0.5) AS score
WHERE score > 0

RETURN u.id AS user_id, u.name AS name,
       toInteger(resolved_count) AS resolved_count,
       toInteger(comment_count)  AS comment_count,
       toInteger(open_issues)    AS open_issues,
       score
ORDER BY score DESC
LIMIT $limit
"""
    async with client.session() as sess:
        result = await sess.run(cypher, issue_id=issue_id, org_id=org_id, limit=limit)
        return [dict(r) for r in await result.data()]


async def get_similar_issues(
    client: Neo4jClient,
    keywords: list[str],
    org_id: str,
    limit: int = 10,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return issues whose title or description contains any of the keywords.

    All matching is case-insensitive substring search — no embeddings required.
    Scoped to the org via the Org→Project→Category→Issue path. When
    `project_id` is given, results are further restricted to that one project.
    """
    if not keywords:
        return []

    project_filter = "AND p.id = $project_id" if project_id else ""
    cypher = f"""
MATCH (i:Issue)-[:IN_CATEGORY]->(cat:Category)-[:IN_PROJECT]->(p:Project)
    -[:BELONGS_TO]->(o:Org {{id: $org_id}})
WHERE (ANY(kw IN $keywords WHERE toLower(i.title) CONTAINS toLower(kw))
   OR ANY(kw IN $keywords
          WHERE i.description IS NOT NULL
            AND toLower(i.description) CONTAINS toLower(kw)))
   {project_filter}
RETURN i.id          AS issue_id,
       i.title       AS title,
       i.type        AS type,
       i.priority    AS priority,
       i.completedAt AS completed_at,
       cat.name      AS category_name
ORDER BY i.createdAt DESC
LIMIT $limit
"""
    params: dict[str, Any] = {"keywords": keywords, "org_id": org_id, "limit": limit}
    if project_id:
        params["project_id"] = project_id
    async with client.session() as sess:
        result = await sess.run(cypher, **params)
        return [dict(r) for r in await result.data()]


async def get_issue_neighbors(
    client: Neo4jClient,
    issue_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return issues in the same category as the given issue.

    Also indicates whether the neighbor shares the same sprint.
    """
    cypher = """
MATCH (i:Issue {id: $issue_id})-[:IN_CATEGORY]->(cat:Category)
MATCH (neighbor:Issue)-[:IN_CATEGORY]->(cat)
WHERE neighbor.id <> $issue_id

OPTIONAL MATCH (i)-[:IN_SPRINT]->(s:Sprint)
OPTIONAL MATCH (neighbor)-[:IN_SPRINT]->(ns:Sprint)

RETURN neighbor.id          AS issue_id,
       neighbor.title       AS title,
       neighbor.type        AS type,
       neighbor.priority    AS priority,
       neighbor.completedAt AS completed_at,
       cat.name             AS category_name,
       (s IS NOT NULL AND ns IS NOT NULL AND s.id = ns.id) AS in_same_sprint
LIMIT $limit
"""
    async with client.session() as sess:
        result = await sess.run(cypher, issue_id=issue_id, limit=limit)
        return [dict(r) for r in await result.data()]


async def get_sprint_issues(
    client: Neo4jClient,
    sprint_id: str,
) -> list[dict[str, Any]]:
    """Return all issues in a sprint with their assignee and status."""
    cypher = """
MATCH (i:Issue)-[:IN_SPRINT]->(s:Sprint {id: $sprint_id})
OPTIONAL MATCH (u:User)-[:ASSIGNED_TO]->(i)
RETURN i.id          AS issue_id,
       i.title       AS title,
       i.type        AS type,
       i.priority    AS priority,
       i.statusId    AS status_id,
       i.completedAt AS completed_at,
       u.id          AS assignee_id,
       u.name        AS assignee_name
ORDER BY i.number ASC
"""
    async with client.session() as sess:
        result = await sess.run(cypher, sprint_id=sprint_id)
        return [dict(r) for r in await result.data()]


async def get_user_workload(
    client: Neo4jClient,
    org_id: str,
    user_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return open-issue counts per user in the org.

    If `user_id` is given, returns only that user's workload. When `project_id`
    is given, only issues belonging to that project are counted.
    """
    # When scoped to a project, the open-issue match must traverse to it.
    if project_id:
        open_match = (
            "OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(i:Issue)"
            "-[:IN_CATEGORY]->(:Category)-[:IN_PROJECT]->(p:Project {id: $project_id})\n"
            "WHERE i.completedAt IS NULL"
        )
    else:
        open_match = (
            "OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(i:Issue)\nWHERE i.completedAt IS NULL"
        )

    params: dict[str, Any] = {"org_id": org_id}
    if project_id:
        params["project_id"] = project_id

    if user_id:
        params["user_id"] = user_id
        cypher = f"""
MATCH (u:User {{id: $user_id}})-[:MEMBER_OF]->(o:Org {{id: $org_id}})
{open_match}
RETURN u.id AS user_id, u.name AS name, count(i) AS open_issues
"""
    else:
        cypher = f"""
MATCH (u:User)-[:MEMBER_OF]->(o:Org {{id: $org_id}})
{open_match}
RETURN u.id AS user_id, u.name AS name, count(i) AS open_issues
ORDER BY open_issues DESC
"""
    async with client.session() as sess:
        result = await sess.run(cypher, **params)
        return [dict(r) for r in await result.data()]


async def get_user_activity(
    client: Neo4jClient,
    org_id: str,
    limit: int = 10,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Rank org members by issue activity (edits + comments).

    Sums each user's CHANGED relationship counts (edit history) and
    COMMENTED_ON counts to answer "who's been most active?". When `project_id`
    is given, only activity on issues in that project is counted.

    Returns an empty list if no behavioral data exists in the graph.
    """
    # Anchor on the activity itself (not MEMBER_OF) so contributors are ranked
    # even if they are no longer org members. Scope via the issue→project path:
    # to one project when scoped, otherwise to any project in the org.
    issue_path = (
        "-[:IN_CATEGORY]->(:Category)-[:IN_PROJECT]->(:Project {id: $project_id})"
        if project_id
        else "-[:IN_CATEGORY]->(:Category)-[:IN_PROJECT]->(:Project)"
        "-[:BELONGS_TO]->(:Org {id: $org_id})"
    )
    cypher = f"""
MATCH (u:User)-[r:CHANGED|COMMENTED_ON]->(i:Issue){issue_path}
WITH u,
     sum(CASE type(r) WHEN 'CHANGED' THEN coalesce(r.count, 0) ELSE 0 END)
         AS change_count,
     sum(CASE type(r) WHEN 'COMMENTED_ON' THEN coalesce(r.count, 0) ELSE 0 END)
         AS comment_count
WITH u, change_count, comment_count, change_count + comment_count AS total_activity
WHERE total_activity > 0
RETURN u.id AS user_id, u.name AS name,
       toInteger(change_count)   AS change_count,
       toInteger(comment_count)  AS comment_count,
       toInteger(total_activity) AS total_activity
ORDER BY total_activity DESC
LIMIT $limit
"""
    params: dict[str, Any] = {"org_id": org_id, "limit": limit}
    if project_id:
        params["project_id"] = project_id
    async with client.session() as sess:
        result = await sess.run(cypher, **params)
        return [dict(r) for r in await result.data()]


async def get_subtask_tree(
    client: Neo4jClient,
    parent_id: str,
) -> list[dict[str, Any]]:
    """Return all subtasks reachable from a parent issue (recursive)."""
    cypher = """
MATCH path = (root:Issue {id: $parent_id})<-[:SUBTASK_OF*1..10]-(subtask:Issue)
RETURN subtask.id          AS issue_id,
       subtask.title       AS title,
       subtask.type        AS type,
       subtask.priority    AS priority,
       subtask.completedAt AS completed_at,
       length(path)        AS depth
ORDER BY depth ASC, subtask.number ASC
"""
    async with client.session() as sess:
        result = await sess.run(cypher, parent_id=parent_id)
        return [dict(r) for r in await result.data()]


async def get_graph_stats(client: Neo4jClient) -> dict[str, Any]:
    """Return node and relationship counts by label/type for observability."""
    count_cypher = """
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
"""
    rel_cypher = """
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count
"""
    async with client.session() as sess:
        node_result = await sess.run(count_cypher)
        nodes = {r["label"]: r["count"] for r in await node_result.data() if r["label"]}

        rel_result = await sess.run(rel_cypher)
        rels = {r["rel_type"]: r["count"] for r in await rel_result.data() if r["rel_type"]}

    return {"nodes": nodes, "relationships": rels}


__all__ = [
    "get_graph_stats",
    "get_issue_neighbors",
    "get_similar_issues",
    "get_smart_assignee",
    "get_sprint_issues",
    "get_subtask_tree",
    "get_user_activity",
    "get_user_workload",
]
