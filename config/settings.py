from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Provider toggle — "groq" | "kimi" | "minimax"
    AI_PROVIDER: Literal["groq", "kimi", "minimax"] = "groq"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_LARGE_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TOOL_MODEL: str = "llama-3.3-70b-versatile"

    # Kimi (Moonshot AI — OpenAI-compatible)
    KIMI_API_KEY: str = ""
    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
    KIMI_FAST_MODEL: str = "moonshot-v1-8k"
    KIMI_LARGE_MODEL: str = "moonshot-v1-32k"
    KIMI_TOOL_MODEL: str = "moonshot-v1-128k"

    # MiniMax (OpenAI-compatible)
    MINIMAX_API_KEY: str = ""
    MINIMAX_BASE_URL: str = "https://api.minimax.io/v1"
    MINIMAX_FAST_MODEL: str = "MiniMax-M2.5"
    MINIMAX_LARGE_MODEL: str = "MiniMax-M3"
    MINIMAX_TOOL_MODEL: str = "MiniMax-M2.7"

    # Claude (kept for future option 2 routing)
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    @model_validator(mode="after")
    def _check_provider_key(self) -> "Settings":
        if self.AI_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when AI_PROVIDER=groq")
        if self.AI_PROVIDER == "kimi" and not self.KIMI_API_KEY:
            raise ValueError("KIMI_API_KEY is required when AI_PROVIDER=kimi")
        if self.AI_PROVIDER == "minimax" and not self.MINIMAX_API_KEY:
            raise ValueError("MINIMAX_API_KEY is required when AI_PROVIDER=minimax")
        return self

    # CORS — comma-separated origins allowed to call this service
    CORS_ORIGINS: list[str] = ["http://localhost:5173","https://ai-pm-frontend-gamma.vercel.app"]

    # Service auth
    INTERNAL_SECRET: str = "dev-secret"

    # Node.js API
    NODE_API_URL: str = "http://localhost:4000"
    BOT_SECRET: str = "dev-bot-secret"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # Limits
    MAX_HISTORY_TURNS: int = 10
    MAX_TOKENS_PER_REQUEST: int = 5000
    GROQ_RATE_LIMIT_RPM: int = 28

    # How long a pending write proposal waits for the user's yes/no before it
    # is dropped (seconds). Tune per deployment — a chattier channel may want
    # a shorter window so a walked-away user can't confirm a stale proposal.
    PENDING_TTL_SECONDS: float = 600.0

    # Resilience — per-call retry on transient errors (rate limit / 5xx)
    LLM_MAX_RETRIES: int = 2

    # Cost control — cumulative token quota per org (0 = unlimited / disabled)
    ORG_TOKEN_QUOTA: int = 0

    # Metering backend — "inprocess" (default) or "postgres".
    # The in-process store is single-instance; "postgres" talks to the
    # node-api's /admin/metering/* routes for multi-instance-safe counters.
    # See metering/usage.py for the protocol + factory.
    METERING_BACKEND: Literal["inprocess", "postgres"] = "inprocess"

    # Embedding provider — "noop" (default, returns zero vector, no API
    # key needed) or "google" (real semantic embeddings via Google AI
    # Studio free tier, requires GOOGLE_AI_API_KEY).
    EMBEDDING_PROVIDER: Literal["noop", "google"] = "noop"
    GOOGLE_AI_API_KEY: str = ""

    # ── Persona / Presentation / Insight (NEW) ────────────────────────
    # Global default persona. Org/user/thread override this. See
    # chains/persona_resolver.py for the resolution chain.
    PERSONA_DEFAULT: Literal["senior_pm", "auditor", "assistant"] = "senior_pm"
    # Cache persona resolution per (org_slug, user_id) for N seconds.
    # Set to 0 to disable caching (resolve every turn — slow, but never stale).
    PERSONA_CACHE_TTL_SECONDS: int = 300
    # Master switch for the insight layer (chains/insight.py).
    INSIGHT_ENABLED: bool = True
    # Master switch for the presentation layer (chains/presentation.py).
    PRESENTATION_ENABLED: bool = True
    # Confidence gate for insights (below this the insight is dropped).
    INSIGHT_MIN_CONFIDENCE: float = 0.6

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
