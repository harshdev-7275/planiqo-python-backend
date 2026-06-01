from loguru import logger


async def suggest_assignee(
    neo4j_client,
    project_id: str,
    issue_title: str,
    issue_type: str,
) -> list[dict]:
    """
    Score each project member as a candidate assignee:
      +3 per resolved same-type issue they were assigned to
      +1 per comment on same-type issues
      -1 per currently open issue (load penalty)
    Returns list of {userId, score} sorted by score descending.
    """
    resolved = await neo4j_client.run(
        """
        MATCH (u:User)-[:ASSIGNED_TO]->(i:Issue)-[:IN_PROJECT]->(p:Project {id: $project_id})
        WHERE i.type = $issue_type AND i.completedAt IS NOT NULL
        RETURN u.id AS userId, count(i) AS cnt
        """,
        {"project_id": project_id, "issue_type": issue_type},
    )

    commented = await neo4j_client.run(
        """
        MATCH (u:User)-[r:COMMENTED_ON]->(i:Issue)-[:IN_PROJECT]->(p:Project {id: $project_id})
        WHERE i.type = $issue_type
        RETURN u.id AS userId, sum(r.count) AS cnt
        """,
        {"project_id": project_id, "issue_type": issue_type},
    )

    load = await neo4j_client.run(
        """
        MATCH (u:User)-[:MEMBER_OF]->(p:Project {id: $project_id})
        OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(open:Issue)-[:IN_PROJECT]->(p)
        WHERE open.completedAt IS NULL
        RETURN u.id AS userId, count(open) AS open_count
        """,
        {"project_id": project_id},
    )

    scores: dict[str, float] = {}

    for row in resolved:
        scores[row["userId"]] = scores.get(row["userId"], 0.0) + row["cnt"] * 3

    for row in commented:
        scores[row["userId"]] = scores.get(row["userId"], 0.0) + row["cnt"]

    for row in load:
        user_id = row["userId"]
        if user_id not in scores:
            scores[user_id] = 0.0
        scores[user_id] -= row["open_count"]

    result = sorted(
        [{"userId": uid, "score": score} for uid, score in scores.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
    logger.info("suggest_assignee project={} candidates={}", project_id, len(result))
    return result


async def find_similar_issues(
    neo4j_client,
    project_id: str,
    title: str,
) -> list[dict]:
    """
    Find issues in the project whose title shares meaningful words with the given title.
    Words of 3 chars or fewer are ignored as noise. Returns up to 5 results sorted by
    match count descending.
    """
    words = [w.lower() for w in title.split() if len(w) > 3]
    if not words:
        return []

    results = await neo4j_client.run(
        """
        MATCH (i:Issue)-[:IN_PROJECT]->(p:Project {id: $project_id})
        WHERE any(word IN $words WHERE toLower(i.title) CONTAINS word)
          AND toLower(i.title) <> toLower($title)
        RETURN i.id AS id, i.number AS number, i.title AS title,
               size([word IN $words WHERE toLower(i.title) CONTAINS word]) AS match_count
        ORDER BY match_count DESC
        LIMIT 5
        """,
        {"project_id": project_id, "words": words, "title": title},
    )
    logger.info("find_similar_issues project={} matches={}", project_id, len(results))
    return results


async def get_expertise_map(
    neo4j_client,
    project_id: str,
) -> list[dict]:
    """
    For each project member, return their activity counts:
    assigned issues, comments made, and field changes made.
    """
    results = await neo4j_client.run(
        """
        MATCH (u:User)-[:MEMBER_OF]->(p:Project {id: $project_id})
        OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(ai:Issue)-[:IN_PROJECT]->(p)
        OPTIONAL MATCH (u)-[co:COMMENTED_ON]->(ci:Issue)-[:IN_PROJECT]->(p)
        OPTIONAL MATCH (u)-[ch:CHANGED]->(chi:Issue)-[:IN_PROJECT]->(p)
        RETURN u.id AS userId, u.name AS name,
               count(DISTINCT ai)  AS assigned,
               sum(co.count)       AS comments,
               sum(ch.count)       AS changes
        """,
        {"project_id": project_id},
    )
    logger.info("get_expertise_map project={} users={}", project_id, len(results))
    return results
