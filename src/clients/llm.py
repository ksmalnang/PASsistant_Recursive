"""Shared LLM clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings

if TYPE_CHECKING:  # runtime import would cycle through src.utils.tools -> src.clients
    from src.services.contracts import InvokableLLM

_llm_instance: InvokableLLM | None = None
_rewrite_llm_instance: InvokableLLM | None = None
_response_llm_instance: InvokableLLM | None = None

# A rewritten query is one short line, so a low cap bounds rewrite latency.
REWRITE_MAX_TOKENS = 64

# Without this the provider spends the rewrite budget on reasoning and returns
# an empty message, which silently turns the rewrite into a no-op.
REWRITE_EXTRA_BODY = {"reasoning": {"enabled": False}}


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


def _apply_provider_restriction(body: dict[str, Any], configured: str) -> None:
    """Restrict one request to the OpenRouter provider endpoints named in a setting."""
    providers = [name.strip() for name in configured.split(",") if name.strip()]
    if providers:
        body["provider"] = {"only": providers, "allow_fallbacks": False}


def _build_extra_body(settings: Any) -> dict[str, Any] | None:
    """Merge reasoning controls with the OpenRouter provider restriction for LLM_MODEL."""
    body = _build_reasoning_extra_body(settings) or {}
    _apply_provider_restriction(body, settings.LLM_PROVIDER)
    return body or None


def _build_rewrite_extra_body(settings: Any) -> dict[str, Any]:
    """Rewrite requests disable reasoning and honour QUERY_REWRITE_PROVIDER."""
    body = dict(REWRITE_EXTRA_BODY)
    _apply_provider_restriction(body, settings.QUERY_REWRITE_PROVIDER)
    return body


def _build_chat_model(
    settings: Any,
    *,
    model: str,
    extra_body: dict[str, Any] | None,
    temperature: float = 0.0,
    streaming: bool = False,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Build one chat model from settings."""
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        # langchain-openai only enables stream_usage by default for its own base
        # URL; with a custom base URL a streaming call reports no token usage.
        stream_usage=streaming,
        extra_body=extra_body,
        **kwargs,
    )


def get_llm() -> InvokableLLM | None:
    """Return the process-wide LLM client, or None without an API key."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None
        _llm_instance = _build_chat_model(
            settings,
            model=settings.LLM_MODEL,
            extra_body=_build_extra_body(settings),
        )
    return _llm_instance


def get_rewrite_llm() -> InvokableLLM | None:
    """Return the query-rewrite client: small model, no reasoning, capped output."""
    global _rewrite_llm_instance
    if _rewrite_llm_instance is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None
        _rewrite_llm_instance = _build_chat_model(
            settings,
            model=getattr(settings, "QUERY_REWRITE_MODEL", "") or settings.LLM_MODEL,
            extra_body=_build_rewrite_extra_body(settings),
            temperature=0.0,
            max_tokens=REWRITE_MAX_TOKENS,
        )
    return _rewrite_llm_instance


def get_response_llm() -> InvokableLLM | None:
    """Return the answer-generation client, streaming tokens for live delivery."""
    global _response_llm_instance
    if _response_llm_instance is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None
        _response_llm_instance = _build_chat_model(
            settings,
            model=settings.LLM_MODEL,
            extra_body=_build_extra_body(settings),
            streaming=True,
        )
    return _response_llm_instance
