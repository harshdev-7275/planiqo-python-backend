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
    BLOCKS       = "BLOCKS"


CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User)    REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue)   REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sprint)  REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (st:Status) REQUIRE st.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label)   REQUIRE l.id IS UNIQUE",
]


async def apply_constraints(neo4j_client: Neo4jClient) -> None:
    """Run all uniqueness constraints against Neo4j. Call once on startup."""
    for stmt in CONSTRAINTS:
        await neo4j_client.run(stmt)
    logger.info("graph schema: {} constraints applied", len(CONSTRAINTS))
