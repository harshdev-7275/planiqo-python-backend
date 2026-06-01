from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

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


def _build_chain(llm: ChatGroq):
    return _PROMPT | llm.with_structured_output(IntentResult)


async def classify(message: str, llm: ChatGroq | None = None) -> IntentResult:
    if not message.strip():
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0, entities={})

    chain = _build_chain(llm or get_fast())
    result = await chain.ainvoke({"message": message})
    return result  # type: ignore[return-value]
