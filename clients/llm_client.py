from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config.settings import settings

# A chat model — or a fallback-wrapped one — viewed as a Runnable.
ChatRunnable = Runnable[LanguageModelInput, BaseMessage]


def _make_groq(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=SecretStr(settings.GROQ_API_KEY),
        temperature=0,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _make_kimi(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.KIMI_API_KEY),
        base_url=settings.KIMI_BASE_URL,
        temperature=0,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _make_minimax(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.MINIMAX_API_KEY),
        base_url=settings.MINIMAX_BASE_URL,
        temperature=0,
        max_retries=settings.LLM_MAX_RETRIES,
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


def _model_name(model: BaseChatModel) -> str:
    return getattr(model, "model_name", getattr(model, "model", "unknown"))


def _with_fallbacks(primary: BaseChatModel, alternates: list[BaseChatModel]) -> ChatRunnable:
    """Wrap a single-shot model so it falls back to the other tiers on error
    (rate limit / 5xx). SAFE ONLY for stateless single calls — never for
    tool-calling agents, where a re-run would repeat side effects."""
    seen = {_model_name(primary)}
    uniq: list[BaseChatModel] = []
    for model in alternates:
        name = _model_name(model)
        if name not in seen:
            seen.add(name)
            uniq.append(model)
    return primary.with_fallbacks(uniq) if uniq else primary


# Built once: fast tier with the remaining tiers as ordered fallbacks.
_FAST_RESILIENT = _with_fallbacks(_fast, [_large, _tool])


def get_fast() -> BaseChatModel:
    return _fast


def get_large() -> BaseChatModel:
    return _large


def get_tool() -> BaseChatModel:
    """Tool-calling tier for ReAct agents. Resilience here is per-call retry
    (max_retries on the model), NOT cross-tier fallback — re-running a partially
    executed agent would duplicate tool side effects."""
    return _tool


def get_fast_resilient() -> ChatRunnable:
    """Fast tier with cross-tier fallback — for stateless single-shot calls
    such as intent classification."""
    return _FAST_RESILIENT
