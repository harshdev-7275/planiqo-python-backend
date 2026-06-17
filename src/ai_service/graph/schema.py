"""Knowledge graph schema — node labels, relationship types, and constraints.

Node labels:
    Org, Project, Category, Sprint, Issue, User

Structural relationships:
    (Project)-[:BELONGS_TO]->(Org)
    (Category)-[:IN_PROJECT]->(Project)
    (Sprint)-[:IN_PROJECT]->(Project)
    (Issue)-[:IN_CATEGORY]->(Category)
    (Issue)-[:IN_SPRINT]->(Sprint)
    (User)-[:ASSIGNED_TO]->(Issue)
    (Issue)-[:SUBTASK_OF]->(Issue)
    (User)-[:MEMBER_OF]->(Org)

Behavioral relationships (signals for smart assignee):
    (User)-[:RESOLVED {at}]->(Issue)         — assignee of a completed issue
    (User)-[:COMMENTED_ON {count, lastAt}]->(Issue)
    (User)-[:CHANGED {count, lastAt}]->(Issue)
"""

from __future__ import annotations

import logging

from ai_service.neo4j import Neo4jClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label + relationship type constants
# ---------------------------------------------------------------------------

ORG = "Org"
PROJECT = "Project"
CATEGORY = "Category"
SPRINT = "Sprint"
ISSUE = "Issue"
USER = "User"

BELONGS_TO = "BELONGS_TO"
IN_PROJECT = "IN_PROJECT"
IN_CATEGORY = "IN_CATEGORY"
IN_SPRINT = "IN_SPRINT"
ASSIGNED_TO = "ASSIGNED_TO"
SUBTASK_OF = "SUBTASK_OF"
MEMBER_OF = "MEMBER_OF"
RESOLVED = "RESOLVED"
COMMENTED_ON = "COMMENTED_ON"
CHANGED = "CHANGED"

# ---------------------------------------------------------------------------
# Uniqueness constraints — applied once at startup via bootstrap_schema()
# ---------------------------------------------------------------------------

_CONSTRAINTS: list[tuple[str, str]] = [
    ("unique_org_id",      "FOR (n:Org)      REQUIRE n.id IS UNIQUE"),
    ("unique_project_id",  "FOR (n:Project)  REQUIRE n.id IS UNIQUE"),
    ("unique_category_id", "FOR (n:Category) REQUIRE n.id IS UNIQUE"),
    ("unique_sprint_id",   "FOR (n:Sprint)   REQUIRE n.id IS UNIQUE"),
    ("unique_issue_id",    "FOR (n:Issue)    REQUIRE n.id IS UNIQUE"),
    ("unique_user_id",     "FOR (n:User)     REQUIRE n.id IS UNIQUE"),
]


async def bootstrap_schema(client: Neo4jClient) -> None:
    """Create uniqueness constraints if they do not already exist.

    Safe to call repeatedly — `IF NOT EXISTS` makes each statement a no-op
    when the constraint is already in place.
    """
    async with client.session() as sess:
        for name, body in _CONSTRAINTS:
            cypher = f"CREATE CONSTRAINT {name} IF NOT EXISTS {body}"
            await sess.run(cypher)
            logger.debug("graph.schema.constraint_applied", extra={"name": name})
    logger.info("graph.schema.bootstrap_complete", extra={"constraint_count": len(_CONSTRAINTS)})


__all__ = [
    "ASSIGNED_TO",
    "BELONGS_TO",
    "CATEGORY",
    "CHANGED",
    "COMMENTED_ON",
    "IN_CATEGORY",
    "IN_PROJECT",
    "IN_SPRINT",
    "ISSUE",
    "MEMBER_OF",
    "ORG",
    "PROJECT",
    "RESOLVED",
    "SPRINT",
    "SUBTASK_OF",
    "USER",
    "bootstrap_schema",
]
