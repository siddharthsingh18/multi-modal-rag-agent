"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: str = Field(default="anthropic", description="anthropic or gemini")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Claude model to use",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=4096, ge=1, le=200000)

    # Vector Database
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    chroma_persist_dir: str = Field(default="./data/chroma")
    vector_db_type: str = Field(default="qdrant", description="qdrant or chroma")

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Database
    database_url: str = Field(default="sqlite:///./data/rag_system.db")

    # Observability
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="multimodal-rag-agent")
    langsmith_tracing: bool = Field(default=False)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    rate_limit_per_minute: int = Field(default=60, ge=1)
    api_auth_key: str = Field(default="")

    # Embedding Model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=10, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=512, ge=100)
    chunk_overlap: int = Field(default=50, ge=0)

    # Ingestion limits
    ingest_data_dir: str = Field(default="./data/ingest")
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)

    # Environment
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    enable_code_executor: bool = Field(default=False)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON string format
            import json

            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle comma-separated format
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
