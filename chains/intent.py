import re

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import ValidationError

from clients.llm_client import ChatRunnable, get_fast_resilient
from models.intents import Intent, IntentResult

_SYSTEM = (
    "You are a project management assistant's intent classifier. "
    "Output a single JSON object with three fields: "
    "  - intent: one of CREATE_ISSUE, UPDATE_ISSUE, QUERY_ISSUES, QUERY_SPRINT, "
    "           CREATE_SPRINT, QUERY_MEMBER, SUMMARIZE, TEAMS_CONTEXT, UNKNOWN. "
    "  - confidence: 0.0-1.0 (use 0.0 for UNKNOWN). "
    "  - entities: a flat object of relevant values (title, priority, type, "
    "             assignee, assignee_id, sprint, sprint_id, issue, number, name, goal). "
    "TRANSCRIBE the user's words into entities. NEVER paraphrase, never add a "
    "word the user did not say. If the user said 'for the checkout flow', the "
    "title is 'checkout flow' — do NOT add 'Bug' or 'Feature'. If the user "
    "named a sprint 'Q3 Hardening', the name is 'Q3 Hardening' — do NOT "
    "replace it with a generic placeholder like 'a new sprint'.\n"
    "Intent rules: "
    "  CREATE_ISSUE if the user wants to create an issue / bug / task / story / "
    "  ticket / 'open a ticket' / 'log a bug' / 'file a bug' / 'add a task' / "
    "  'make a story' / 'I need to track this'. The verbs include create, open, "
    "  file, log, raise, add, make, track, submit. The subject can be any "
    "  noun for a unit of work. "
    "  UPDATE_ISSUE if the user names an existing issue by NUMBER and wants to "
    "  change a field: set priority, change title, rename, reassign, move to "
    "  done, close, etc. REQUIRES the issue number in entities.issue or "
    "  entities.number — without it, return UNKNOWN. "
    "  QUERY_ISSUES if the user wants a list of issues filtered some way. "
    "  QUERY_MEMBER if the user asks about a PERSON's work — 'what is X "
    "  working on', 'X's issues', 'who is working on Y', 'show me what X "
    "  is doing', or any phrasing of 'issues assigned to X' / 'tasks for X'. "
    "  SUMMARIZE if the user wants a sprint summary: 'summarize sprint X', "
    "  'how did the sprint go', 'what did we accomplish in sprint X', "
    "  'sprint X recap', 'sprint summary'. "
    "  QUERY_SPRINT for sprint status / progress (not summaries). "
    "  CREATE_SPRINT for 'create / start a sprint'. "
    "  TEAMS_CONTEXT, UNKNOWN for anything else. "
    "Respond ONLY with the JSON object — no prose, no markdown, no code fences."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "{message}"),
])


_UNKNOWN = IntentResult(intent=Intent.UNKNOWN, confidence=0.0, entities={})


def _parse_response(message: AIMessage) -> IntentResult:
    """Parse the LLM's raw output into an IntentResult — NEVER raises.

    Per AIService.md, every LLM call returns AIResult and never raises to the
    caller. A model that returns prose, empty content, malformed JSON, or a
    schema-invalid object all map to ``Intent.UNKNOWN`` (the safe default).
    The caller (``classify``) gets a well-typed ``IntentResult`` it can
    branch on; the supervisor never has to wrap this in a try/except.
    """
    content = message.content if isinstance(message.content, str) else str(message.content)
    # Reasoning models (e.g. MiniMax-M1) wrap output in <think>…</think> before JSON
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Strip markdown code fences if present
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if code_match:
        content = code_match.group(1).strip()
    if not content:
        return _UNKNOWN
    try:
        return IntentResult.model_validate_json(content)
    except ValidationError as e:
        # Expected: model returned something that didn't fit the schema. Log
        # the raw content at debug (not info — could be verbose) so we can
        # debug model drift without crashing the chat.
        logger.debug("classify: model output did not validate as IntentResult: {}", e)
        return _UNKNOWN
    except (ValueError, TypeError) as e:
        # ValueError: malformed JSON; TypeError: unexpected content type.
        logger.debug("classify: model output was not valid JSON: {}", e)
        return _UNKNOWN


def _build_chain(llm: ChatRunnable) -> Runnable[dict[str, str], IntentResult]:
    return _PROMPT | llm | RunnableLambda(_parse_response)


async def classify(
    message: str, llm: ChatRunnable | None = None, callbacks: Callbacks = None
) -> IntentResult:
    if not message.strip():
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0, entities={})

    chain = _build_chain(llm or get_fast_resilient())
    config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
    return await chain.ainvoke({"message": message}, config=config)
