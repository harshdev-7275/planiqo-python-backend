from enum import Enum
from pydantic import BaseModel


class Intent(str, Enum):
    CREATE_ISSUE = "CREATE_ISSUE"
    UPDATE_ISSUE = "UPDATE_ISSUE"
    QUERY_ISSUES = "QUERY_ISSUES"
    QUERY_SPRINT = "QUERY_SPRINT"
    CREATE_SPRINT = "CREATE_SPRINT"
    QUERY_MEMBER = "QUERY_MEMBER"
    SUMMARIZE = "SUMMARIZE"
    TEAMS_CONTEXT = "TEAMS_CONTEXT"
    UNKNOWN = "UNKNOWN"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float
    entities: dict
