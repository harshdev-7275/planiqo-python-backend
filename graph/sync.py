from loguru import logger


async def upsert_user(neo4j_client, user: dict) -> None:
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


async def upsert_project(neo4j_client, project: dict) -> None:
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


async def upsert_issue(neo4j_client, issue: dict) -> None:
    await neo4j_client.run(
        """
        MERGE (i:Issue {id: $id})
        SET i.number = $number, i.title = $title,
            i.type = $type, i.priority = $priority,
            i.createdAt = $createdAt, i.completedAt = $completedAt
        """,
        {
            "id":          issue["id"],
            "number":      issue["number"],
            "title":       issue["title"],
            "type":        issue["type"],
            "priority":    issue["priority"],
            "createdAt":   issue["createdAt"],
            "completedAt": issue.get("completedAt"),
        },
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


async def upsert_member(
    neo4j_client, user_id: str, project_id: str, role: str
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
    neo4j_client,
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
        params: dict = {
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
    neo4j_client, user_id: str, issue_id: str
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
    neo4j_client, user_id: str, issue_id: str, field: str
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
    neo4j_client,
    node_api_client,
    org_slug: str,
) -> dict[str, int]:
    nodes_created = 0
    relationships_created = 0
    tombstones_created = 0

    project_list: list = await node_api_client.get_projects(org_slug)
    logger.info("full_sync org={} projects_found={}", org_slug, len(project_list))

    for project in project_list:
        await upsert_project(neo4j_client, project)
        nodes_created += 1

        members: list = await node_api_client.get_project_members(org_slug, project["id"])
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

        issues: list = await node_api_client.get_issues(org_slug, project["id"])
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
