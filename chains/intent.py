import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from clients.llm_client import get_fast
from models.intents import Intent, IntentResult

_SYSTEM = (
    "Classify the user message into one intent and extract entities. "
    "Intents: CREATE_ISSUE, UPDATE_ISSUE, QUERY_ISSUES, QUERY_SPRINT, "
    "CREATE_SPRINT, QUERY_MEMBER, SUMMARIZE, TEAMS_CONTEXT, UNKNOWN. "
    "entities: any relevant values (title, priority, member, sprint, etc). "
    "confidence: 0.0-1.0. If unclear use UNKNOWN with confidence 0.0. "
    "Respond only in JSON. No explanation."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "{message}"),
])


def _parse_response(message: AIMessage) -> IntentResult:
    content = message.content if isinstance(message.content, str) else str(message.content)
    # Reasoning models (e.g. MiniMax-M1) wrap output in <think>…</think> before JSON
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Strip markdown code fences if present
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code_match:
        content = code_match.group(1).strip()
    return IntentResult.model_validate_json(content)


def _build_chain(llm: BaseChatModel):
    return _PROMPT | llm | RunnableLambda(_parse_response)


async def classify(message: str, llm: BaseChatModel | None = None) -> IntentResult:
    if not message.strip():
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0, entities={})

    chain = _build_chain(llm or get_fast())
    result = await chain.ainvoke({"message": message})
    return result  # type: ignore[return-value]
