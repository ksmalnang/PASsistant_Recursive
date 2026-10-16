"""Shared embeddings client."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from src.config import get_settings

_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Return the process-wide embeddings client."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            base_url=settings.OPENAI_BASE_URL,
            check_embedding_ctx_length=False,
            model_kwargs=_embedding_model_kwargs(settings.EMBEDDING_PROVIDER),
        )
    return _embeddings


def _embedding_model_kwargs(configured_provider: str) -> dict[str, Any]:
    """Pin embeddings to the configured OpenRouter providers, matching other clients."""
    providers = [name.strip() for name in configured_provider.split(",") if name.strip()]
    if not providers:
        return {}
    return {"extra_body": {"provider": {"order": providers, "allow_fallbacks": False}}}
