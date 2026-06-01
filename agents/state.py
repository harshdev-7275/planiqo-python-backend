from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from models.intents import IntentResult


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    org_slug: str
    project_id: str | None
    intent: IntentResult | None
    result: dict | None
    error: str | None
    next: str
