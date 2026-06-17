"""Application settings — single source of truth for environment variables.

Validates at startup with pydantic-settings so misconfiguration fails fast.
Override any field via env var or `.env` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Validated application settings.

    All fields read from env vars (matching the field name, case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity ---
    app_name: str = Field(default="ai-service", description="Service name (used in logs, OpenAPI).")
    app_version: str = Field(default="0.1.0", description="Service version (semver).")
    app_env: Environment = Field(
        default="development",
        description="Deployment environment.",
    )

    # --- Server ---
    # Binding 0.0.0.0 is intentional for container deployments where the port
    # is mapped from the host. Override via env (HOST) when running locally.
    host: str = Field(default="0.0.0.0", description="Bind address.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port.")
    # NoDecode keeps pydantic-settings from trying to JSON-parse the comma-separated
    # string before our field_validator runs. Without it, "" or "a,b" raises.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins. Comma-separated in env.",
    )

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level.",
    )

    # --- Neo4j ---
    neo4j_url: str = Field(
        default="",
        description="Neo4j connection URI (e.g. neo4j+s://xxxxx.databases.neo4j.io).",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username.")
    neo4j_password: str = Field(default="", description="Neo4j password (never log this).")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name.")
    neo4j_max_connection_pool_size: int = Field(
        default=50, ge=1, le=1000, description="Max driver connection pool size."
    )
    neo4j_connection_timeout_sec: float = Field(
        default=10.0, gt=0, description="Connection timeout in seconds."
    )
    # Safety: must be true to run destructive Neo4j operations (reset_graph, etc.).
    # In production this MUST stay false. The reset function will refuse without it.
    neo4j_allow_destructive: bool = Field(
        default=False,
        description="Allow destructive Neo4j operations. NEVER true in production.",
    )

    # --- Node-backend (the only data source the ai-service talks to) ---
    node_backend_url: str = Field(
        default="http://localhost:3000",
        description="Base URL of the node-backend service.",
    )
    node_backend_service_token: str = Field(
        default="",
        description="Shared service-to-service auth token. Must match AI_SERVICE_TOKEN in node-backend.",
    )
    node_backend_timeout_sec: float = Field(
        default=90.0, gt=0, description="HTTP timeout for node-backend calls."
    )

    # --- LLM providers (Groq + MiniMax) ---
    llm_provider: Literal["groq", "minimax"] = Field(
        default="groq", description="LLM provider to use."
    )
    # Groq
    groq_api_key: str = Field(default="", description="Groq API key.")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name.")
    # MiniMax (OpenAI-compatible)
    minimax_api_key: str = Field(default="", description="MiniMax API key.")
    minimax_base_url: str = Field(
        default="https://api.minimaxi.com/v1",
        description="MiniMax OpenAI-compatible base URL.",
    )
    minimax_model: str = Field(default="MiniMax-Text-01", description="MiniMax model name.")
    # Common
    agent_max_steps: int = Field(default=8, ge=1, le=50, description="Max agent loop steps.")
    llm_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature."
    )

    @property
    def node_backend_configured(self) -> bool:
        """True if node-backend URL and service token are set."""
        return bool(self.node_backend_url and self.node_backend_service_token)

    @property
    def llm_configured(self) -> bool:
        """True if the active LLM provider has an API key configured."""
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "minimax":
            return bool(self.minimax_api_key)
        return False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        """Accept comma-separated string from env, or pass through a list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return value
        msg = f"cors_origins must be a string or list, got {type(value).__name__}"
        raise ValueError(msg)

    @property
    def neo4j_configured(self) -> bool:
        """True if Neo4j connection settings are populated (url + password)."""
        return bool(self.neo4j_url and self.neo4j_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so re-imports don't re-read .env. Tests can call `get_settings.cache_clear()`
    to force a fresh read.
    """
    return Settings()
