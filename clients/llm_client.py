from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import settings
from models.results import LLMResult


def _make_groq(model: str) -> ChatGroq:
    return ChatGroq(model=model, api_key=settings.GROQ_API_KEY, temperature=0)


def _make_kimi(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.KIMI_API_KEY,
        base_url=settings.KIMI_BASE_URL,
        temperature=0,
    )


def _make_minimax(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.MINIMAX_API_KEY,
        base_url=settings.MINIMAX_BASE_URL,
        temperature=0,
    )


def _build_tiers() -> tuple[list[BaseChatModel], list[str]]:
    if settings.AI_PROVIDER == "kimi":
        models = [
            settings.KIMI_FAST_MODEL,
            settings.KIMI_LARGE_MODEL,
            settings.KIMI_TOOL_MODEL,
        ]
        return [_make_kimi(m) for m in models], models
    if settings.AI_PROVIDER == "minimax":
        models = [
            settings.MINIMAX_FAST_MODEL,
            settings.MINIMAX_LARGE_MODEL,
            settings.MINIMAX_TOOL_MODEL,
        ]
        return [_make_minimax(m) for m in models], models
    models = [
        settings.GROQ_FAST_MODEL,
        settings.GROQ_LARGE_MODEL,
        settings.GROQ_TOOL_MODEL,
    ]
    return [_make_groq(m) for m in models], models


_TIERS, _TIER_NAMES = _build_tiers()
_fast, _large, _tool = _TIERS


def get_fast() -> BaseChatModel:
    return _fast


def get_large() -> BaseChatModel:
    return _large


def get_tool() -> BaseChatModel:
    return _tool


def _is_rate_limit(exc: Exception) -> bool:
    cls = type(exc).__name__
    return cls == "RateLimitError"


async def run_with_fallback(prompt: str, preferred: BaseChatModel | None = None) -> LLMResult:
    models = [preferred, *_TIERS] if preferred else _TIERS
    seen: set[str] = set()
    candidates: list[BaseChatModel] = []
    for m in models:
        name = getattr(m, "model_name", getattr(m, "model", "unknown"))
        if m is not None and name not in seen:
            seen.add(name)
            candidates.append(m)

    last_error = "No models available"
    for model in candidates:
        model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
            tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
            logger.info("llm_call provider={} model={} tokens={}", settings.AI_PROVIDER, model_name, tokens)
            return LLMResult(
                success=True,
                content=str(response.content),
                model_used=model_name,
                tokens_used=tokens,
            )
        except Exception as e:
            if _is_rate_limit(e):
                logger.warning("Rate limited on {}, trying next tier", model_name)
                last_error = f"Rate limited on {model_name}"
                continue
            logger.error("LLM call failed on {}: {}", model_name, e)
            return LLMResult(success=False, error=str(e), model_used=model_name)

    return LLMResult(success=False, error=last_error, model_used="none")
