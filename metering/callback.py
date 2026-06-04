"""LangChain callback that meters token usage per tenant.

Attached to the intent chain and the ReAct agents through the run config, so
every live LLM call — including the agents' internal tool-calling steps, which
otherwise leave no trace — is counted against the org's usage.
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger

from metering.usage import RequestTokens, UsageStore


def extract_total_tokens(response: LLMResult) -> int:
    """Best-effort total-token count across provider response shapes."""
    # OpenAI-compatible providers (Groq / Kimi / MiniMax) report on llm_output.
    out = response.llm_output or {}
    usage = out.get("token_usage") or out.get("usage") or {}
    total = usage.get("total_tokens")
    if total:
        return int(total)
    # Chat models attach usage_metadata to each generated message.
    summed = 0
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None)
            if metadata:
                summed += int(metadata.get("total_tokens", 0))
    return summed


class UsageCallback(BaseCallbackHandler):
    def __init__(
        self,
        org_slug: str,
        user_id: str | None,
        store: UsageStore,
        request_tokens: RequestTokens | None = None,
        feature: str = "chat",
    ) -> None:
        self.org_slug = org_slug
        self.user_id = user_id
        self.store = store
        self.request_tokens = request_tokens
        self.feature = feature

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        tokens = extract_total_tokens(response)
        if tokens <= 0:
            return
        self.store.add(self.org_slug, tokens)
        if self.request_tokens is not None:
            self.request_tokens.add(tokens)
        logger.info(
            "llm_usage org={} user={} feature={} tokens={} org_total={}",
            self.org_slug, self.user_id, self.feature, tokens, self.store.get(self.org_slug),
        )
