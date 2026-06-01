from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_LARGE_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TOOL_MODEL: str = "llama-3.3-70b-versatile"
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # CORS — comma-separated origins allowed to call this service
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

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

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
