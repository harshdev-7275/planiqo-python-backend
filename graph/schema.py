from loguru import logger

from clients.neo4j_client import Neo4jClient


class NodeLabel:
    USER    = "User"
    ISSUE   = "Issue"
    PROJECT = "Project"
    SPRINT  = "Sprint"
    STATUS  = "Status"
    LABEL   = "Label"


class RelType:
    MEMBER_OF    = "MEMBER_OF"
    ASSIGNED_TO  = "ASSIGNED_TO"
    REPORTED     = "REPORTED"
    COMMENTED_ON = "COMMENTED_ON"
    CHANGED      = "CHANGED"
    IN_PROJECT   = "IN_PROJECT"
    IN_SPRINT    = "IN_SPRINT"
    HAS_STATUS   = "HAS_STATUS"
    HAS_LABEL    = "HAS_LABEL"
    BLOCKS       = "BLOCKS"


CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User)    REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue)   REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sprint)  REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (st:Status) REQUIRE st.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label)   REQUIRE l.id IS UNIQUE",
]

# Native Cypher 5+ vector index on Issue.embedding. Dim matches the
# text-embedding-004 model (768). Idempotent (IF NOT EXISTS) so re-runs
# on startup are safe. Runs on Aura Free — see Sprint 2.2 plan.
# Backticks on the indexConfig keys are required by Cypher (they're
# reserved otherwise).
VECTOR_INDEXES: list[str] = [
    (
        "CREATE VECTOR INDEX issue_embedding IF NOT EXISTS "
        "FOR (i:Issue) ON i.embedding "
        "OPTIONS { indexConfig: { "
        "  `vector.dimensions`: 768, "
        "  `vector.similarity_function`: 'cosine' "
        "}}"
    ),
]


async def apply_constraints(neo4j_client: Neo4jClient) -> None:
    """Run all uniqueness constraints + vector indexes against Neo4j.
    Call once on startup. Idempotent: each statement uses IF NOT EXISTS."""
    for stmt in CONSTRAINTS:
        await neo4j_client.run(stmt)
    for stmt in VECTOR_INDEXES:
        await neo4j_client.run(stmt)
    logger.info(
        "graph schema: {} constraints + {} vector indexes applied",
        len(CONSTRAINTS), len(VECTOR_INDEXES),
    )
