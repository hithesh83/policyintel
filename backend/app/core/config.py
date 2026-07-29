"""
Application Configuration (Pydantic Settings)
==============================================

Centralised settings loaded from .env via python-dotenv + pydantic-settings.

All environment variables are validated and typed at startup.
This prevents the "it works on my machine" problem from misconfigured env vars.

Usage:
    from app.core.config import settings
    print(settings.ollama_url)
    print(settings.ollama_model)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    Organised by subsystem for clarity.
    Add new settings here as new modules are built (Week 2+).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars gracefully
    )

    # ------------------------------------------------------------------
    # Application Core
    # ------------------------------------------------------------------
    project_name: str = Field(default="PolicyIntel AI")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    api_v1_str: str = Field(default="/api/v1")

    # ------------------------------------------------------------------
    # Ollama LLM Settings (Week 1)
    # ------------------------------------------------------------------
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=120.0, alias="OLLAMA_TIMEOUT")
    ollama_temperature: float = Field(default=0.7, alias="OLLAMA_TEMPERATURE")
    ollama_top_p: float = Field(default=0.9, alias="OLLAMA_TOP_P")
    ollama_num_predict: int = Field(default=-1, alias="OLLAMA_NUM_PREDICT")
    ollama_max_retries: int = Field(default=3, alias="OLLAMA_MAX_RETRIES")
    ollama_retry_min_wait: float = Field(default=1.0, alias="OLLAMA_RETRY_MIN_WAIT")
    ollama_retry_max_wait: float = Field(default=30.0, alias="OLLAMA_RETRY_MAX_WAIT")
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    # ------------------------------------------------------------------
    # PostgreSQL (Week 2+)
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://policy_user:policy_password@localhost:5432/policyintel_db",
        alias="DATABASE_URL",
    )

    # ------------------------------------------------------------------
    # Qdrant (Week 2+)
    # ------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="policy_chunks", alias="QDRANT_COLLECTION_NAME")

    # ------------------------------------------------------------------
    # Neo4j (Week 2+)
    # ------------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="policy_password", alias="NEO4J_PASSWORD")

    # ------------------------------------------------------------------
    # Pipeline Tuning (Week 2+)
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    max_retrieved_docs: int = Field(default=10, alias="MAX_RETRIEVED_DOCS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    Uses ``@lru_cache`` to ensure .env is read only once.
    Call ``get_settings.cache_clear()`` in tests that need fresh settings.
    """
    return Settings()


# Convenience module-level singleton
settings = get_settings()
