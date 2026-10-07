"""
Application configuration management using Pydantic Settings.
Loads environment variables from .env file with validation.
"""


from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    # --- LLM Configuration ---
    OPENAI_API_KEY: Optional[str] = Field(
        default=None, description="OpenAI API key for LLM operations"
    )
    OPENAI_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    LLM_MODEL: str = Field(
        default="deepseek/deepseek-v4-flash:exacto", description="Primary LLM model"
    )
    LLM_REASONING_ENABLED: bool = Field(
        default=True,
        description="Send provider reasoning controls with LLM requests when supported",
    )
    LLM_REASONING_EFFORT: Literal["xhigh", "high", "medium", "low", "minimal", "none"] = Field(
        default="medium", description="Reasoning effort for supported LLM providers"
    )
    LLM_REASONING_MAX_TOKENS: Optional[int] = Field(
        default=2000,
        description="Maximum reasoning tokens for supported LLM providers",
    )
    LLM_REASONING_EXCLUDE: bool = Field(
        default=True,
        description="Use reasoning internally without returning reasoning text when supported",
    )
    EMBEDDING_MODEL: str = Field(
        default="qwen/qwen3-embedding-8b:nitro",
        description="Embedding model for vector search",
    )

    # --- RAGAS Evaluation ---
    RAGAS_LLM_MODEL: str = Field(
        default="",
        description="LLM model for RAGAS evaluator judge (falls back to LLM_MODEL if empty)",
    )
    RAGAS_LLM_REASONING_ENABLED: Optional[bool] = Field(
        default=None,
        description="Override provider reasoning controls for RAGAS evaluator LLM only",
    )
    RAGAS_LLM_REASONING_EFFORT: Optional[
        Literal["xhigh", "high", "medium", "low", "minimal", "none"]
    ] = Field(
        default=None,
        description="Override reasoning effort for RAGAS evaluator LLM only",
    )
    RAGAS_LLM_REASONING_MAX_TOKENS: Optional[int] = Field(
        default=None,
        description="Override maximum reasoning tokens for RAGAS evaluator LLM only",
    )
    RAGAS_LLM_REASONING_EXCLUDE: Optional[bool] = Field(
        default=None,
        description="Override reasoning exclusion for RAGAS evaluator LLM only",
    )
    RAGAS_EMBEDDING_MODEL: str = Field(
        default="",
        description="Embedding model for RAGAS semantic similarity metrics (falls back to EMBEDDING_MODEL if empty)",
    )

    # --- GLM-4 OCR Configuration (Zhipu AI) ---
    ZHIPU_API_KEY: Optional[str] = Field(default=None, description="Zhipu AI API key for GLM-4 OCR")
    ZHIPU_BASE_URL: str = Field(default="https://open.bigmodel.cn/api/paas/v4")
    GLM_OCR_MODEL: str = Field(default="glm-ocr", description="GLM vision model for OCR")
    GLM_LAYOUT_MODEL: str = Field(default="glm-ocr", description="GLM model for layout parsing")

    # --- Qdrant Vector Database ---
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION_NAME: str = Field(default="student_documents_1.2")
    VECTOR_SIZE: int = Field(default=4096, description="Embedding vector dimension")
    RETRIEVAL_STRATEGY: Literal["similarity", "rrf", "reranker"] = Field(
        default="similarity",
        description="Retrieval ranking strategy: similarity, rrf, or reranker",
    )
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        ge=1,
        description="Number of hydrated context records to retrieve and expose to response generation",
    )
    RERANKER_MODEL: Optional[str] = Field(
        default=None,
        description="Reranker model used when RETRIEVAL_STRATEGY=reranker",
    )
    RERANKER_BASE_URL: Optional[str] = Field(
        default=None,
        description="Remote reranker base URL. If unset, FastEmbed local reranking is used.",
    )
    RERANKER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for the remote reranker endpoint",
    )
    RERANKER_CANDIDATE_MULTIPLIER: int = Field(
        default=6,
        description="How many first-stage candidates to fetch per requested result before reranking",
    )

    # --- Chunking Configuration ---
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Target character length per chunk for document indexing",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        description="Character overlap between consecutive chunks",
    )
    CHUNK_SEPARATORS: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""],
        description="Ordered separators for RecursiveCharacterTextSplitter",
    )

    # --- Redis Cache ---
    REDIS_URL: Optional[str] = Field(default=None)
    REDIS_KEY_PREFIX: str = Field(default="student-records-chatbot")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=300)

    # --- LangSmith Configuration ---
    LANGSMITH_TRACING: bool = Field(default=True)
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGSMITH_API_KEY: Optional[str] = Field(
        default=None, description="LangSmith API key for tracing"
    )
    LANGSMITH_PROJECT: str = Field(default="student-records-chatbot")

    # --- Application Configuration ---
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_APP_NAME: str = Field(default="PASsistant")
    LOG_SYSLOG_FACILITY: int = Field(
        default=16,
        description="RFC 5424 facility code. Defaults to local0.",
    )
    DATA_DIR: Path = Field(default=Path("data"))
    CORS_ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    RATE_LIMIT_PER_MINUTE: int = Field(default=20)
    SESSION_MAX_ACTIVE: int = Field(default=100)

    # --- Telegram Bot ---
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None)
    TELEGRAM_WEBHOOK_URL: Optional[str] = Field(default=None)
    TELEGRAM_WEBHOOK_SECRET_TOKEN: Optional[str] = Field(default=None)
    TELEGRAM_ENABLED: bool = Field(default=False)
    TELEGRAM_MAX_FILE_BYTES: int = Field(default=20_000_000)
    TELEGRAM_ALLOWED_FILE_MIME_TYPES: list[str] | None = Field(default=None)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.APP_ENV.lower() == "production"

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw data directory."""
        return self.DATA_DIR / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed data directory."""
        return self.DATA_DIR / "processed"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.

    Returns:
        Settings: Application configuration instance
    """
    return Settings()
