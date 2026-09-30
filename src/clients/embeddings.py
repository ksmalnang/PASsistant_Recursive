"""Shared embeddings client."""

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
            model_kwargs={
                "extra_body": {"provider": {"order": ["deepinfra"], "allow_fallbacks": False}}
            },
        )
    return _embeddings
