from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from groq import RateLimitError
from loguru import logger

from config.settings import settings
from models.results import LLMResult

_fast = ChatGroq(
    model=settings.GROQ_FAST_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_large = ChatGroq(
    model=settings.GROQ_LARGE_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_tool = ChatGroq(
    model=settings.GROQ_TOOL_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_TIERS: list[ChatGroq] = [_fast, _large, _tool]
_TIER_NAMES = [settings.GROQ_FAST_MODEL, settings.GROQ_LARGE_MODEL, settings.GROQ_TOOL_MODEL]


def get_fast() -> ChatGroq:
    return _fast


def get_large() -> ChatGroq:
    return _large


def get_tool() -> ChatGroq:
    return _tool


async def run_with_fallback(prompt: str, preferred: ChatGroq | None = None) -> LLMResult:
    models = [preferred, *_TIERS] if preferred else _TIERS
    # deduplicate while preserving order
    seen: set[str] = set()
    candidates: list[ChatGroq] = []
    for m in models:
        if m is not None and m.model_name not in seen:
            seen.add(m.model_name)
            candidates.append(m)

    last_error = "No models available"
    for model in candidates:
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
            tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
            logger.info(
                "llm_call model={} tokens={}",
                model.model_name,
                tokens,
            )
            return LLMResult(
                success=True,
                content=str(response.content),
                model_used=model.model_name,
                tokens_used=tokens,
            )
        except RateLimitError:
            logger.warning("Rate limited on {}, trying next tier", model.model_name)
            last_error = f"Rate limited on {model.model_name}"
            continue
        except Exception as e:
            logger.error("LLM call failed on {}: {}", model.model_name, e)
            return LLMResult(success=False, error=str(e), model_used=model.model_name)

    return LLMResult(success=False, error=last_error, model_used="none")
