"""Shared LLM client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings

if TYPE_CHECKING:  # runtime import would cycle through src.utils.tools -> src.clients
    from src.services.contracts import InvokableLLM

_llm_instance: InvokableLLM | None = None


def _build_reasoning_extra_body(settings: Any) -> dict[str, Any] | None:
    """Build provider reasoning controls for OpenAI-compatible requests."""
    if not settings.LLM_REASONING_ENABLED:
        return None

    reasoning = {"exclude": settings.LLM_REASONING_EXCLUDE}
    if settings.LLM_REASONING_MAX_TOKENS is not None:
        reasoning["max_tokens"] = settings.LLM_REASONING_MAX_TOKENS
    else:
        reasoning["effort"] = settings.LLM_REASONING_EFFORT

    return {"reasoning": reasoning}


def get_llm() -> InvokableLLM | None:
    """Return the process-wide LLM client, or None without an API key."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None
        extra_body = _build_reasoning_extra_body(settings)
        _llm_instance = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            base_url=settings.OPENAI_BASE_URL,
            temperature=0.3,
            extra_body=extra_body,
        )
    return _llm_instance
