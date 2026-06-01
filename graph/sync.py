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
            i.createdAt = $createdAt
        """,
        {
            "id":        issue["id"],
            "number":    issue["number"],
            "title":     issue["title"],
            "type":      issue["type"],
            "priority":  issue["priority"],
            "createdAt": issue["createdAt"],
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
    await neo4j_client.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (p:Project {id: $project_id})
        MERGE (u)-[r:MEMBER_OF]->(p)
        SET r.role = $role
        """,
        {"user_id": user_id, "project_id": project_id, "role": role},
    )


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

    project_list: list = await node_api_client.get_projects(org_slug)

    for project in project_list:
        await upsert_project(neo4j_client, project)
        nodes_created += 1

        members: list = await node_api_client.get_project_members(org_slug, project["id"])
        for member in members:
            await upsert_user(neo4j_client, member)
            await upsert_member(neo4j_client, member["userId"], project["id"], member["role"])
            nodes_created += 1
            relationships_created += 1  # MEMBER_OF

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
        "full_sync org={} nodes={} rels={}",
        org_slug,
        nodes_created,
        relationships_created,
    )
    return {"nodes_created": nodes_created, "relationships_created": relationships_created}
