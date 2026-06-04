from langchain_core.runnables import Runnable

from clients.llm_client import get_fast, get_fast_resilient, get_tool
from config.settings import settings


def test_models_have_retries_configured() -> None:
    # Every live model retries transient errors (rate limit / 5xx) before failing.
    assert getattr(get_fast(), "max_retries", None) == settings.LLM_MAX_RETRIES
    assert getattr(get_tool(), "max_retries", None) == settings.LLM_MAX_RETRIES


def test_get_fast_resilient_is_a_runnable() -> None:
    assert isinstance(get_fast_resilient(), Runnable)
