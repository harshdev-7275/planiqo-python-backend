from typing import Any

from loguru import logger

from clients.neo4j_client import Neo4jClient

# Recency decay (item 16): a same-type issue resolved within this many days
# earns a bonus on top of the all-time weight, so recent expertise outranks
# stale expertise without ignoring it entirely.
_RECENCY_WINDOW_DAYS = 30
_RESOLVED_WEIGHT = 3
_RECENCY_BONUS = 2


async def suggest_assignee(
    neo4j_client: Neo4jClient,
    project_id: str,
    issue_title: str,
    issue_type: str,
) -> list[dict[str, Any]]:
    """
    Score each project member as a candidate assignee:
      +3 per resolved same-type issue they were assigned to
      +2 extra per such issue resolved in the last 30 days (recency decay)
      +1 per comment on same-type issues
      -1 per currently open issue (load penalty)
    Returns list of {userId, score} sorted by score descending.

    Tombstoned (removed) members are excluded: the load query already filters
    on ``MEMBER_OF.removed_at IS NULL``, and we intersect the final result
    with the set of active member IDs from that query. An ex-member's old
    resolved-issue signal is therefore not surfaced as a suggestion.
    """
    resolved = await neo4j_client.run(
        """
        MATCH (u:User)-[:ASSIGNED_TO]->(i:Issue)-[:IN_PROJECT]->(p:Project {id: $project_id})
        WHERE i.type = $issue_type AND i.completedAt IS NOT NULL
        RETURN u.id AS userId,
               count(i) AS cnt,
               sum(CASE
                     WHEN datetime(i.completedAt) >= datetime() - duration({days: $recent_days})
                     THEN 1 ELSE 0
                   END) AS recent_cnt
        """,
        {
            "project_id": project_id,
            "issue_type": issue_type,
            "recent_days": _RECENCY_WINDOW_DAYS,
        },
    )

    commented = await neo4j_client.run(
        """
        MATCH (u:User)-[r:COMMENTED_ON]->(i:Issue)-[:IN_PROJECT]->(p:Project {id: $project_id})
        WHERE i.type = $issue_type
        RETURN u.id AS userId, sum(r.count) AS cnt
        """,
        {"project_id": project_id, "issue_type": issue_type},
    )

    # Load query: project members ONLY (not tombstones). The active set built
    # from this row set is the final filter for the suggestions.
    load = await neo4j_client.run(
        """
        MATCH (u:User)-[r:MEMBER_OF]->(p:Project {id: $project_id})
        WHERE r.removed_at IS NULL
        OPTIONAL MATCH (u)-[:ASSIGNED_TO]->(open:Issue)-[:IN_PROJECT]->(p)
        WHERE open.completedAt IS NULL
        RETURN u.id AS userId, u.name AS userName, count(open) AS open_count
        """,
        {"project_id": project_id},
    )

    logger.debug(
        "suggest_assignee project={} type={} | resolved={} commented={} active_members={}",
        project_id, issue_type, len(resolved), len(commented), len(load),
    )

    active_member_ids: set[str] = {row["userId"] for row in load}

    scores: dict[str, float] = {}
    names:  dict[str, str]   = {}

    for row in resolved:
        # All-time weight + a recency bonus. ``recent_cnt`` is absent from older
        # callers/mocks, so default to 0 — backward compatible (the score then
        # equals the previous all-time-only formula).
        weight = row["cnt"] * _RESOLVED_WEIGHT + row.get("recent_cnt", 0) * _RECENCY_BONUS
        scores[row["userId"]] = scores.get(row["userId"], 0.0) + weight

    for row in commented:
        scores[row["userId"]] = scores.get(row["userId"], 0.0) + row["cnt"]

    for row in load:
        user_id = row["userId"]
        names[user_id] = row.get("userName") or user_id
        if user_id not in scores:
            scores[user_id] = 0.0
        scores[user_id] -= row["open_count"]

    # Final filter: only active members. An ex-member who left the project
    # but has historical resolved-issue signal will have a positive score
    # above — drop them here.
    candidates: list[dict[str, Any]] = [
        {"userId": uid, "userName": names.get(uid, uid), "score": score}
        for uid, score in scores.items()
        if uid in active_member_ids
    ]
    result = sorted(candidates, key=lambda x: x["score"], reverse=True)

    if result:
        logger.debug(
            "suggest_assignee top candidates: {}",
            [{"user": r["userName"], "score": r["score"]} for r in result[:3]],
        )
    else:
        logger.info(
            "suggest_assignee project={} type={}: no candidates — "
            "no active MEMBER_OF relationships in graph (run /graph/sync first)",
            project_id, issue_type,
        )

    logger.info("suggest_assignee project={} candidates={}", project_id, len(result))
    return result


async def find_similar_issues(
    neo4j_client: Neo4jClient,
    project_id: str,
    title: str,
) -> list[dict[str, Any]]:
    """
    Find issues in the project whose title shares meaningful words with the given title.
    Words of 3 chars or fewer are ignored as noise. Returns up to 5 results sorted by
    match count descending.

    This is the LEGACY word-overlap path. New callers should use
    ``hybrid_similar_issues`` (Sprint 2.2) which combines this pre-filter
    with a vector kNN rerank for semantic match ("login fails" vs
    "can't sign in").
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


async def hybrid_similar_issues(
    neo4j_client: Neo4jClient,
    project_id: str,
    title: str,
    embedding: list[float],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Find similar issues using a hybrid retrieval pattern (Sprint 2.2).

    Step 1 (cheap): word-overlap pre-filter narrows the candidate set.
    Step 2 (semantic): a ``db.index.vector.queryNodes`` call ranks the
    candidates by cosine similarity to ``embedding``.

    The pre-filter is what makes this practical on Aura Free: a naive
    full-graph vector kNN would scan every issue; the word-overlap
    narrows to O(handful) candidates first. If the pre-filter returns
    nothing, we return the empty list — the vector kNN would also
    return nothing, and we avoid the network round-trip.

    Falls back to the word-overlap result on a vector-index error
    (e.g. the index hasn't been created yet on a fresh deploy) — the
    smart feature is best-effort, not blocking.
    """
    pre = await find_similar_issues(neo4j_client, project_id, title)
    if not pre:
        return []
    candidate_ids = [row["id"] for row in pre]
    try:
        ranked = await neo4j_client.run(
            """
            CALL db.index.vector.queryNodes(
                'issue_embedding', $k, $embedding
            ) YIELD node AS i, score
            WHERE i.id IN $candidate_ids
              AND i.id IS NOT NULL
            RETURN i.id AS id, i.number AS number, i.title AS title, score
            ORDER BY score DESC
            LIMIT $k
            """,
            {
                "k": k,
                "embedding": embedding,
                "candidate_ids": candidate_ids,
            },
        )
    except Exception as e:  # index missing or transient neo4j error
        logger.warning(
            "hybrid_similar: vector index query failed, returning pre-filter: {}",
            e,
        )
        return pre
    logger.info(
        "hybrid_similar_issues project={} pre_filter={} vector_ranked={}",
        project_id, len(pre), len(ranked),
    )
    return ranked


async def get_expertise_map(
    neo4j_client: Neo4jClient,
    project_id: str,
) -> list[dict[str, Any]]:
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


async def blocked_issues_in_sprint(
    neo4j_client: Neo4jClient,
    sprint_id: str,
) -> list[dict[str, Any]]:
    """Issues in a sprint that are blocked by a still-incomplete blocker
    (item 14). Answers "what's blocking sprint X". An issue blocked only by
    already-completed work is NOT returned — that dependency is satisfied.

    Returns ``[{number, title, blockers: [{number, title}, ...]}]`` ordered by
    issue number.
    """
    results = await neo4j_client.run(
        """
        MATCH (blocker:Issue)-[:BLOCKS]->(i:Issue)-[:IN_SPRINT]->(s:Sprint {id: $sprint_id})
        WHERE blocker.completedAt IS NULL
        WITH i, collect({number: blocker.number, title: blocker.title}) AS blockers
        RETURN i.number AS number, i.title AS title, blockers
        ORDER BY i.number
        """,
        {"sprint_id": sprint_id},
    )
    logger.info("blocked_issues_in_sprint sprint={} blocked={}", sprint_id, len(results))
    return results


async def find_dependency_cycles(
    neo4j_client: Neo4jClient,
    project_id: str,
) -> list[dict[str, Any]]:
    """Detect circular BLOCKS dependencies in a project (item 14).

    A cycle (A blocks B blocks … blocks A) is a planning error — nothing in it
    can start. Returns one row per detected cycle: ``{cycle: [number, ...]}``.
    The ``size(nodes(path)) <= 8`` bound keeps the variable-length match cheap
    on Aura Free; deeper cycles are rare and still partially surfaced.
    """
    results = await neo4j_client.run(
        """
        MATCH path = (i:Issue)-[:BLOCKS*1..8]->(i)
        WHERE (i)-[:IN_PROJECT]->(:Project {id: $project_id})
        RETURN [n IN nodes(path) | n.number] AS cycle
        LIMIT 25
        """,
        {"project_id": project_id},
    )
    logger.info("find_dependency_cycles project={} cycles={}", project_id, len(results))
    return results


async def find_user_by_name(
    neo4j_client: Neo4jClient,
    project_id: str,
    name: str,
) -> list[dict[str, Any]]:
    """Resolve a human name to project member candidates.

    Case-insensitive substring match on name OR email — handles "Alice"
    matching "Alice Smith", and "alice@acme.com" matching "alice". Filters
    out tombstoned (removed) members. Up to 5 results, ordered by name.
    """
    query = (name or "").strip()
    if not query:
        return []

    results = await neo4j_client.run(
        """
        MATCH (u:User)-[r:MEMBER_OF]->(p:Project {id: $project_id})
        WHERE r.removed_at IS NULL
          AND (toLower(u.name)  CONTAINS toLower($name)
            OR toLower(u.email) CONTAINS toLower($name))
        RETURN u.id AS id, u.name AS name, u.email AS email
        ORDER BY u.name
        LIMIT 5
        """,
        {"project_id": project_id, "name": query},
    )
    logger.info(
        "find_user_by_name project={} query='{}' matches={}",
        project_id, query, len(results),
    )
    return results
