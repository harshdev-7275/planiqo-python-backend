"""Knowledge graph sync — fetch PM data from node-backend and upsert into Neo4j.

All writes use MERGE + SET so the sync is idempotent (safe to re-run). The
service fetches all HTTP data before opening the Neo4j session to avoid
holding a connection while waiting on the node-backend.

Sync order within a project:
    1. Org + Project nodes
    2. Categories  → IN_PROJECT rels
    3. Sprints     → IN_PROJECT rels
    4. Members     → MEMBER_OF rels
    5. Issues      → IN_CATEGORY / IN_SPRINT / ASSIGNED_TO / SUBTASK_OF rels
    6. RESOLVED    (completed issues → their assignee)
    7. COMMENTED_ON (per-issue comment fetch → unique author counts)
    8. CHANGED     (per-issue history fetch → unique changer counts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from ai_service.clients.node_backend import NodeBackendClient
from ai_service.neo4j import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Counts of nodes and relationships touched during a sync run."""

    projects: int = 0
    categories: int = 0
    sprints: int = 0
    issues: int = 0
    users: int = 0
    relationships: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: SyncStats) -> SyncStats:
        return SyncStats(
            projects=self.projects + other.projects,
            categories=self.categories + other.categories,
            sprints=self.sprints + other.sprints,
            issues=self.issues + other.issues,
            users=self.users + other.users,
            relationships=self.relationships + other.relationships,
            errors=self.errors + other.errors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "projects": self.projects,
            "categories": self.categories,
            "sprints": self.sprints,
            "issues": self.issues,
            "users": self.users,
            "relationships": self.relationships,
            "errors": self.errors,
        }


class GraphSyncService:
    """Fetches PM data from node-backend and upserts it into the Neo4j graph.

    Does NOT connect to Postgres — all data flows through the node-backend's
    HTTP API.
    """

    def __init__(self, neo4j: Neo4jClient, backend: NodeBackendClient) -> None:
        self._neo4j = neo4j
        self._backend = backend

    async def full_sync(self, org_slug: str, org_id: str) -> SyncStats:
        """Sync every project in the org. Returns aggregated stats."""
        projects = await self._backend.list_projects(org_slug, limit=200)
        totals = SyncStats()
        for project in projects:
            stats = await self.sync_project(org_slug, org_id, project)
            totals = totals.merge(stats)
        logger.info(
            "graph.sync.full_complete",
            extra={"org_id": org_id, **totals.to_dict()},
        )
        return totals

    async def sync_project(
        self,
        org_slug: str,
        org_id: str,
        project: dict[str, Any],
    ) -> SyncStats:
        """Sync a single project (all entities + behavioral signals) into Neo4j."""
        project_id = project["id"]
        stats = SyncStats()

        # ------------------------------------------------------------------
        # 1. Fetch all data from node-backend before touching Neo4j
        # ------------------------------------------------------------------
        categories = await self._backend.list_categories(org_slug, UUID(project_id))
        sprints = await self._backend.list_sprints(org_slug, UUID(project_id))
        issues = await self._backend.list_issues(org_slug, UUID(project_id), limit=1000)
        members = await self._backend.list_org_members(org_slug, limit=500)

        # Behavioral signals: comments + history per issue (one HTTP call each)
        comment_signals: dict[str, list[dict[str, Any]]] = {}
        history_signals: dict[str, list[dict[str, Any]]] = {}
        for issue in issues:
            iid = issue["id"]
            try:
                comment_signals[iid] = await self._backend.list_issue_comments(
                    org_slug, UUID(project_id), UUID(iid)
                )
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"comments:{iid}:{exc}")
                comment_signals[iid] = []
            try:
                history_signals[iid] = await self._backend.list_issue_history(
                    org_slug, UUID(project_id), UUID(iid)
                )
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"history:{iid}:{exc}")
                history_signals[iid] = []

        # ------------------------------------------------------------------
        # 2. Write everything to Neo4j
        # ------------------------------------------------------------------
        async with self._neo4j.session() as sess:
            # Org node
            await sess.run(
                "MERGE (o:Org {id: $id}) SET o.slug = $slug",
                id=org_id,
                slug=org_slug,
            )

            # Project node + BELONGS_TO
            await sess.run(
                """
                MERGE (p:Project {id: $id})
                SET p.name = $name, p.orgId = $org_id
                WITH p
                MATCH (o:Org {id: $org_id})
                MERGE (p)-[:BELONGS_TO]->(o)
                """,
                id=project_id,
                name=project.get("name", ""),
                org_id=org_id,
            )
            stats.projects += 1

            # Categories + IN_PROJECT
            if categories:
                await sess.run(
                    """
                    UNWIND $cats AS c
                    MERGE (cat:Category {id: c.id})
                    SET cat.name = c.name, cat.color = coalesce(c.color, '#6366f1'),
                        cat.projectId = c.projectId
                    WITH cat, c
                    MATCH (p:Project {id: c.projectId})
                    MERGE (cat)-[:IN_PROJECT]->(p)
                    """,
                    cats=categories,
                )
                stats.categories += len(categories)
                stats.relationships += len(categories)

            # Sprints + IN_PROJECT
            if sprints:
                await sess.run(
                    """
                    UNWIND $sprints AS s
                    MERGE (sprint:Sprint {id: s.id})
                    SET sprint.name = s.name, sprint.status = s.status,
                        sprint.startDate = s.startDate, sprint.endDate = s.endDate,
                        sprint.projectId = s.projectId
                    WITH sprint, s
                    MATCH (p:Project {id: s.projectId})
                    MERGE (sprint)-[:IN_PROJECT]->(p)
                    """,
                    sprints=sprints,
                )
                stats.sprints += len(sprints)
                stats.relationships += len(sprints)

            # Members + MEMBER_OF
            if members:
                await sess.run(
                    """
                    UNWIND $members AS m
                    MERGE (u:User {id: m.userId})
                    SET u.name = m.name, u.email = coalesce(m.email, '')
                    WITH u
                    MATCH (o:Org {id: $org_id})
                    MERGE (u)-[:MEMBER_OF]->(o)
                    """,
                    members=members,
                    org_id=org_id,
                )
                stats.users += len(members)
                stats.relationships += len(members)

            # Issue nodes (batch upsert)
            if issues:
                issue_props = [_issue_props(i) for i in issues]
                await sess.run(
                    """
                    UNWIND $issues AS props
                    MERGE (i:Issue {id: props.id})
                    SET i += {
                        number: props.number,
                        title: props.title,
                        description: props.description,
                        type: props.type,
                        priority: props.priority,
                        statusId: props.statusId,
                        completedAt: props.completedAt,
                        createdAt: props.createdAt
                    }
                    """,
                    issues=issue_props,
                )
                stats.issues += len(issues)

                # IN_CATEGORY
                cat_rels = [
                    {"issue_id": i["id"], "cat_id": i["categoryId"]}
                    for i in issues
                    if i.get("categoryId")
                ]
                if cat_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (c:Category {id: r.cat_id})
                        MERGE (i)-[:IN_CATEGORY]->(c)
                        """,
                        rels=cat_rels,
                    )
                    stats.relationships += len(cat_rels)

                # IN_SPRINT
                sprint_rels = [
                    {"issue_id": i["id"], "sprint_id": i["sprintId"]}
                    for i in issues
                    if i.get("sprintId")
                ]
                if sprint_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (s:Sprint {id: r.sprint_id})
                        MERGE (i)-[:IN_SPRINT]->(s)
                        """,
                        rels=sprint_rels,
                    )
                    stats.relationships += len(sprint_rels)

                # ASSIGNED_TO
                assign_rels = [
                    {"issue_id": i["id"], "user_id": i["assigneeId"]}
                    for i in issues
                    if i.get("assigneeId")
                ]
                if assign_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (u:User {id: r.user_id})
                        MERGE (u)-[:ASSIGNED_TO]->(i)
                        """,
                        rels=assign_rels,
                    )
                    stats.relationships += len(assign_rels)

                # SUBTASK_OF
                parent_rels = [
                    {"issue_id": i["id"], "parent_id": i["parentId"]}
                    for i in issues
                    if i.get("parentId")
                ]
                if parent_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (p:Issue {id: r.parent_id})
                        MERGE (i)-[:SUBTASK_OF]->(p)
                        """,
                        rels=parent_rels,
                    )
                    stats.relationships += len(parent_rels)

                # RESOLVED — assignee of a completed issue gets credit
                resolved_rels = [
                    {
                        "issue_id": i["id"],
                        "user_id": i["assigneeId"],
                        "at": i.get("completedAt"),
                    }
                    for i in issues
                    if i.get("completedAt") and i.get("assigneeId")
                ]
                if resolved_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (u:User {id: r.user_id})
                        MERGE (u)-[rel:RESOLVED]->(i)
                        SET rel.at = r.at
                        """,
                        rels=resolved_rels,
                    )
                    stats.relationships += len(resolved_rels)

                # COMMENTED_ON — group per (issue, author)
                comment_rels = _build_comment_rels(comment_signals)
                if comment_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MERGE (u:User {id: r.user_id})
                        ON CREATE SET u.name = r.user_name, u.email = ''
                        MERGE (u)-[rel:COMMENTED_ON]->(i)
                        SET rel.count = r.count, rel.lastAt = r.lastAt
                        """,
                        rels=comment_rels,
                    )
                    stats.relationships += len(comment_rels)

                # CHANGED — group per (issue, changer)
                change_rels = _build_change_rels(history_signals)
                if change_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MERGE (u:User {id: r.user_id})
                        ON CREATE SET u.name = r.user_name, u.email = ''
                        MERGE (u)-[rel:CHANGED]->(i)
                        SET rel.count = r.count, rel.lastAt = r.lastAt
                        """,
                        rels=change_rels,
                    )
                    stats.relationships += len(change_rels)

        logger.info(
            "graph.sync.project_complete",
            extra={"project_id": project_id, **stats.to_dict()},
        )
        return stats

    async def sync_single_issue(
        self,
        org_slug: str,
        org_id: str,
        project_id: str,
    ) -> SyncStats:
        """Fetch all issues for a project and sync them (lightweight incremental path).

        Used by the incremental endpoint — re-syncs the whole project's issues
        but skips categories/sprints/members (already in the graph from full sync).
        """
        stats = SyncStats()
        issues = await self._backend.list_issues(org_slug, UUID(project_id), limit=1000)

        comment_signals: dict[str, list[dict[str, Any]]] = {}
        history_signals: dict[str, list[dict[str, Any]]] = {}
        for issue in issues:
            iid = issue["id"]
            try:
                comment_signals[iid] = await self._backend.list_issue_comments(
                    org_slug, UUID(project_id), UUID(iid)
                )
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"comments:{iid}:{exc}")
                comment_signals[iid] = []
            try:
                history_signals[iid] = await self._backend.list_issue_history(
                    org_slug, UUID(project_id), UUID(iid)
                )
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"history:{iid}:{exc}")
                history_signals[iid] = []

        project_obj: dict[str, Any] = {"id": project_id, "name": ""}
        async with self._neo4j.session() as sess:
            # Ensure Org + Project exist (no-op if already there)
            await sess.run(
                "MERGE (o:Org {id: $id}) SET o.slug = $slug",
                id=org_id,
                slug=org_slug,
            )
            await sess.run(
                """
                MERGE (p:Project {id: $id})
                SET p.orgId = $org_id
                WITH p
                MATCH (o:Org {id: $org_id})
                MERGE (p)-[:BELONGS_TO]->(o)
                """,
                id=project_id,
                org_id=org_id,
            )

            if issues:
                issue_props = [_issue_props(i) for i in issues]
                await sess.run(
                    """
                    UNWIND $issues AS props
                    MERGE (i:Issue {id: props.id})
                    SET i += {
                        number: props.number,
                        title: props.title,
                        description: props.description,
                        type: props.type,
                        priority: props.priority,
                        statusId: props.statusId,
                        completedAt: props.completedAt,
                        createdAt: props.createdAt
                    }
                    """,
                    issues=issue_props,
                )
                stats.issues += len(issues)

                cat_rels = [
                    {"issue_id": i["id"], "cat_id": i["categoryId"]}
                    for i in issues
                    if i.get("categoryId")
                ]
                if cat_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (c:Category {id: r.cat_id})
                        MERGE (i)-[:IN_CATEGORY]->(c)
                        """,
                        rels=cat_rels,
                    )
                    stats.relationships += len(cat_rels)

                sprint_rels = [
                    {"issue_id": i["id"], "sprint_id": i["sprintId"]}
                    for i in issues
                    if i.get("sprintId")
                ]
                if sprint_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (s:Sprint {id: r.sprint_id})
                        MERGE (i)-[:IN_SPRINT]->(s)
                        """,
                        rels=sprint_rels,
                    )
                    stats.relationships += len(sprint_rels)

                assign_rels = [
                    {"issue_id": i["id"], "user_id": i["assigneeId"]}
                    for i in issues
                    if i.get("assigneeId")
                ]
                if assign_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (u:User {id: r.user_id})
                        MERGE (u)-[:ASSIGNED_TO]->(i)
                        """,
                        rels=assign_rels,
                    )
                    stats.relationships += len(assign_rels)

                resolved_rels = [
                    {"issue_id": i["id"], "user_id": i["assigneeId"], "at": i.get("completedAt")}
                    for i in issues
                    if i.get("completedAt") and i.get("assigneeId")
                ]
                if resolved_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MATCH (u:User {id: r.user_id})
                        MERGE (u)-[rel:RESOLVED]->(i)
                        SET rel.at = r.at
                        """,
                        rels=resolved_rels,
                    )
                    stats.relationships += len(resolved_rels)

                comment_rels = _build_comment_rels(comment_signals)
                if comment_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MERGE (u:User {id: r.user_id})
                        ON CREATE SET u.name = r.user_name, u.email = ''
                        MERGE (u)-[rel:COMMENTED_ON]->(i)
                        SET rel.count = r.count, rel.lastAt = r.lastAt
                        """,
                        rels=comment_rels,
                    )
                    stats.relationships += len(comment_rels)

                change_rels = _build_change_rels(history_signals)
                if change_rels:
                    await sess.run(
                        """
                        UNWIND $rels AS r
                        MATCH (i:Issue {id: r.issue_id})
                        MERGE (u:User {id: r.user_id})
                        ON CREATE SET u.name = r.user_name, u.email = ''
                        MERGE (u)-[rel:CHANGED]->(i)
                        SET rel.count = r.count, rel.lastAt = r.lastAt
                        """,
                        rels=change_rels,
                    )
                    stats.relationships += len(change_rels)

        _ = project_obj  # suppress unused variable
        return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _issue_props(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue["id"],
        "number": issue.get("number", 0),
        "title": issue.get("title", ""),
        "description": issue.get("description") or "",
        "type": issue.get("type", "task"),
        "priority": issue.get("priority", "medium"),
        "statusId": issue.get("statusId", ""),
        "completedAt": issue.get("completedAt"),
        "createdAt": issue.get("createdAt", ""),
    }


def _build_comment_rels(
    comment_signals: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Group comments by (issue_id, author_id) and return relationship props."""
    rels: list[dict[str, Any]] = []
    for issue_id, comments in comment_signals.items():
        counts: dict[str, dict[str, Any]] = {}
        for c in comments:
            author = c.get("author") or {}
            uid = author.get("id")
            if not uid:
                continue
            if uid not in counts:
                counts[uid] = {
                    "user_id": uid,
                    "user_name": author.get("name", ""),
                    "issue_id": issue_id,
                    "count": 0,
                    "lastAt": c.get("createdAt") or "",
                }
            counts[uid]["count"] += 1
            ts = c.get("createdAt") or ""
            if ts > counts[uid]["lastAt"]:
                counts[uid]["lastAt"] = ts
        rels.extend(counts.values())
    return rels


def _build_change_rels(
    history_signals: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Group history entries by (issue_id, changer_id) and return relationship props."""
    rels: list[dict[str, Any]] = []
    for issue_id, entries in history_signals.items():
        counts: dict[str, dict[str, Any]] = {}
        for e in entries:
            changer = e.get("changedBy") or {}
            uid = changer.get("id")
            if not uid:
                continue
            if uid not in counts:
                counts[uid] = {
                    "user_id": uid,
                    "user_name": changer.get("name", ""),
                    "issue_id": issue_id,
                    "count": 0,
                    "lastAt": e.get("changedAt") or "",
                }
            counts[uid]["count"] += 1
            ts = e.get("changedAt") or ""
            if ts > counts[uid]["lastAt"]:
                counts[uid]["lastAt"] = ts
        rels.extend(counts.values())
    return rels


__all__ = ["GraphSyncService", "SyncStats"]
