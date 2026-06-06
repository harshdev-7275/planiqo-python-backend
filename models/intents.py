from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    # Defaults to {} when the model omits it. This matters for the pre-router,
    # whose prompt asks for ONLY intent + confidence (no entity extraction) —
    # without a default, every pre-router response failed validation
    # ("entities Field required") and the pre-router silently deferred to the
    # heavy classify chain on every message. classify still extracts entities;
    # this just stops a missing field from collapsing a valid result to UNKNOWN.
    entities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entities(cls, v: Any) -> dict[str, Any]:
        # Some models return [{name: k, value: v}, ...] instead of {k: v}
        if isinstance(v, list):
            return {item["name"]: item["value"] for item in v if "name" in item}
        return v if isinstance(v, dict) else {}
