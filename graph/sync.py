from typing import Any

from loguru import logger

from clients.neo4j_client import Neo4jClient
from clients.node_api import NodeAPIClient


async def upsert_user(neo4j_client: Neo4jClient, user: dict[str, Any]) -> None:
    await neo4j_client.run(
        """
        MERGE (u:User {id: $id})
        SET u.name = $name, u.email = $email, u.avatarUrl = $avatarUrl
        """,
        {
            "id":        user["userId"],
            "name":      user["name"],
            "email":     user["email"],
            "avatarUrl": user.get("avatarUrl"),
        },
    )


async def upsert_project(neo4j_client: Neo4jClient, project: dict[str, Any]) -> None:
    await neo4j_client.run(
        """
        MERGE (p:Project {id: $id})
        SET p.name = $name, p.key = $key
        """,
        {
            "id":   project["id"],
            "name": project["name"],
            "key":  project["key"],
        },
    )


async def upsert_issue(
    neo4j_client: Neo4jClient,
    issue: dict[str, Any],
    embedding: list[float] | None = None,
) -> None:
    """Upsert the Issue node and its project/reporter/assignee/sprint edges.

    ``embedding`` is an optional 768-dim vector (matching the
    ``issue_embedding`` vector index). When provided, it is stored on
    the node so the vector index has data to query on subsequent
    ``hybrid_similar_issues`` calls. Omitting it is a no-op for the
    embedding field — backwards compatible with the full_sync path that
    does not (yet) compute embeddings for every issue.
    """
    params: dict[str, Any] = {
        "id":          issue["id"],
        "number":      issue["number"],
        "title":       issue["title"],
        "type":        issue["type"],
        "priority":    issue["priority"],
        "createdAt":   issue["createdAt"],
        "completedAt": issue.get("completedAt"),
    }
    if embedding is not None:
        params["embedding"] = embedding

    await neo4j_client.run(
        """
        MERGE (i:Issue {id: $id})
        SET i.number = $number, i.title = $title,
            i.type = $type, i.priority = $priority,
            i.createdAt = $createdAt, i.completedAt = $completedAt
        """,
        params,
    )

    await neo4j_client.run(
        """
        MERGE (i:Issue {id: $issue_id})
        MERGE (p:Project {id: $project_id})
        MERGE (i)-[:IN_PROJECT]->(p)
        """,
        {"issue_id": issue["id"], "project_id": issue["projectId"]},
    )

    await neo4j_client.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (i:Issue {id: $issue_id})
        MERGE (u)-[:REPORTED]->(i)
        """,
        {"user_id": issue["reporterId"], "issue_id": issue["id"]},
    )

    if issue.get("assigneeId"):
        await neo4j_client.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (i:Issue {id: $issue_id})
            MERGE (u)-[:ASSIGNED_TO]->(i)
            """,
            {"user_id": issue["assigneeId"], "issue_id": issue["id"]},
        )

    if issue.get("sprintId"):
        await neo4j_client.run(
            """
            MERGE (i:Issue {id: $issue_id})
            MERGE (s:Sprint {id: $sprint_id})
            MERGE (i)-[:IN_SPRINT]->(s)
            """,
            {"issue_id": issue["id"], "sprint_id": issue["sprintId"]},
        )

    # --- Labels (item 15): HAS_LABEL edges. Gated on presence so the full_sync
    # path (which doesn't carry labels yet) and existing callers are unchanged.
    for label in issue.get("labels") or []:
        if not (isinstance(label, dict) and label.get("id")):
            continue
        await upsert_label(neo4j_client, label)
        await neo4j_client.run(
            """
            MERGE (i:Issue {id: $issue_id})
            MERGE (l:Label {id: $label_id})
            MERGE (i)-[:HAS_LABEL]->(l)
            """,
            {"issue_id": issue["id"], "label_id": label["id"]},
        )

    # --- Dependencies (item 14): (this)-[:BLOCKS]->(other). ``blocks`` lists
    # the issues this one blocks; ``blockedBy`` lists the issues blocking it.
    # Both gated on presence — no-op for callers that don't supply them.
    for blocked_id in issue.get("blocks") or []:
        await neo4j_client.run(
            """
            MERGE (i:Issue {id: $issue_id})
            MERGE (b:Issue {id: $blocked_id})
            MERGE (i)-[:BLOCKS]->(b)
            """,
            {"issue_id": issue["id"], "blocked_id": blocked_id},
        )
    for blocker_id in issue.get("blockedBy") or []:
        await neo4j_client.run(
            """
            MERGE (i:Issue {id: $issue_id})
            MERGE (b:Issue {id: $blocker_id})
            MERGE (b)-[:BLOCKS]->(i)
            """,
            {"issue_id": issue["id"], "blocker_id": blocker_id},
        )


async def upsert_label(neo4j_client: Neo4jClient, label: dict[str, Any]) -> None:
    """Upsert a Label node (Sprint 3.2 / item 15). ``label`` is ``{"id", "name"}``."""
    await neo4j_client.run(
        """
        MERGE (l:Label {id: $id})
        SET l.name = $name
        """,
        {"id": label["id"], "name": label.get("name")},
    )


async def upsert_sprint(neo4j_client: Neo4jClient, sprint: dict[str, Any]) -> None:
    """Upsert a Sprint node with real metadata (Sprint 3.2 / item 15).

    Before this, Sprint nodes were created bare (id only) as a side effect of
    an issue's IN_SPRINT edge. Carrying name/status/dates unlocks sprint-health
    and "what's in sprint X" queries that need more than the id."""
    await neo4j_client.run(
        """
        MERGE (s:Sprint {id: $id})
        SET s.name = $name, s.status = $status,
            s.startDate = $startDate, s.endDate = $endDate
        """,
        {
            "id":        sprint["id"],
            "name":      sprint.get("name"),
            "status":    sprint.get("status"),
            "startDate": sprint.get("startDate"),
            "endDate":   sprint.get("endDate"),
        },
    )


async def upsert_member(
    neo4j_client: Neo4jClient, user_id: str, project_id: str, role: str
) -> None:
    """Upsert a User-(MEMBER_OF)->Project edge.

    Reactivating a re-joined user clears the tombstone (``removed_at``) so
    smart queries immediately start treating them as active again.
    """
    await neo4j_client.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (p:Project {id: $project_id})
        MERGE (u)-[r:MEMBER_OF]->(p)
        SET r.role = $role, r.removed_at = null
        """,
        {"user_id": user_id, "project_id": project_id, "role": role},
    )


async def tombstone_missing_members(
    neo4j_client: Neo4jClient,
    project_id: str,
    active_user_ids: set[str],
) -> int:
    """Mark every MEMBER_OF relationship in ``project_id`` as removed when
    the user is not in ``active_user_ids``.

    Why this is needed: full_sync is all-MERGE — it never deletes. An
    ex-member who left the project would still be reachable from
    ``MEMBER_OF`` and ``ASSIGNED_TO`` edges, so ``smart-assignee`` would
    still surface them. Tombstoning preserves the history (re-join resets
    the flag) while keeping active-state queries clean.

    Returns the number of relationships marked. The ``active_user_ids`` set
    may be empty (a project whose entire roster was removed) — in that case
    we tombstone every still-active edge for the project.
    """
    if active_user_ids:
        query = """
            MATCH (u:User)-[r:MEMBER_OF]->(p:Project {id: $project_id})
            WHERE r.removed_at IS NULL
              AND NOT u.id IN $active_user_ids
            SET r.removed_at = datetime()
            RETURN count(r) AS tombstoned
        """
        params: dict[str, Any] = {
            "project_id": project_id,
            "active_user_ids": list(active_user_ids),
        }
    else:
        query = """
            MATCH (u:User)-[r:MEMBER_OF]->(p:Project {id: $project_id})
            WHERE r.removed_at IS NULL
            SET r.removed_at = datetime()
            RETURN count(r) AS tombstoned
        """
        params = {"project_id": project_id}

    result = await neo4j_client.run(query, params)
    return int(result[0]["tombstoned"]) if result else 0


async def upsert_comment_activity(
    neo4j_client: Neo4jClient, user_id: str, issue_id: str
) -> None:
    await neo4j_client.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (i:Issue {id: $issue_id})
        MERGE (u)-[r:COMMENTED_ON]->(i)
        ON CREATE SET r.count = 1
        ON MATCH  SET r.count = r.count + 1
        """,
        {"user_id": user_id, "issue_id": issue_id},
    )


async def upsert_change_activity(
    neo4j_client: Neo4jClient, user_id: str, issue_id: str, field: str
) -> None:
    await neo4j_client.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (i:Issue {id: $issue_id})
        MERGE (u)-[r:CHANGED {field: $field}]->(i)
        ON CREATE SET r.count = 1
        ON MATCH  SET r.count = r.count + 1
        """,
        {"user_id": user_id, "issue_id": issue_id, "field": field},
    )


async def full_sync(
    neo4j_client: Neo4jClient,
    node_api_client: NodeAPIClient,
    org_slug: str,
) -> dict[str, int]:
    nodes_created = 0
    relationships_created = 0
    tombstones_created = 0

    project_list: list[dict[str, Any]] = await node_api_client.get_projects(org_slug)
    logger.info("full_sync org={} projects_found={}", org_slug, len(project_list))

    for project in project_list:
        await upsert_project(neo4j_client, project)
        nodes_created += 1

        members: list[dict[str, Any]] = await node_api_client.get_project_members(org_slug, project["id"])
        active_ids: set[str] = set()
        for member in members:
            await upsert_user(neo4j_client, member)
            await upsert_member(neo4j_client, member["userId"], project["id"], member["role"])
            active_ids.add(member["userId"])
            nodes_created += 1
            relationships_created += 1  # MEMBER_OF

        # Tombstone members who were in the project last sync but aren't in
        # Node's response now (they left, or were removed from the project).
        # Runs AFTER the upsert loop so re-joined users have removed_at reset
        # in the same pass.
        tombstoned = await tombstone_missing_members(
            neo4j_client, project["id"], active_ids
        )
        tombstones_created += tombstoned

        # Sprint metadata (item 15): upsert real Sprint nodes (name/status/dates)
        # so sprint-health queries have more than the bare id that an issue's
        # IN_SPRINT edge would create.
        sprint_list: list[dict[str, Any]] = await node_api_client.get_sprints(
            org_slug, project["id"]
        )
        for sprint in sprint_list:
            await upsert_sprint(neo4j_client, sprint)
            nodes_created += 1

        issues: list[dict[str, Any]] = await node_api_client.get_issues(org_slug, project["id"])
        for issue in issues:
            await upsert_issue(neo4j_client, issue)
            nodes_created += 1
            relationships_created += 2  # IN_PROJECT + REPORTED
            if issue.get("assigneeId"):
                relationships_created += 1
            if issue.get("sprintId"):
                relationships_created += 1

        logger.info(
            "full_sync project='{}' members={} issues={} (assigned={} in_sprint={}) tombstoned={}",
            project.get("name"),
            len(members),
            len(issues),
            sum(1 for i in issues if i.get("assigneeId")),
            sum(1 for i in issues if i.get("sprintId")),
            tombstoned,
        )

    logger.info(
        "full_sync org={} done | nodes={} rels={} tombstones={}",
        org_slug, nodes_created, relationships_created, tombstones_created,
    )
    return {
        "nodes_created": nodes_created,
        "relationships_created": relationships_created,
        "tombstones_created": tombstones_created,
    }
