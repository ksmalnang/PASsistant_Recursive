"""Unit tests for the process-wide client providers."""

from types import SimpleNamespace

import pytest

from src.clients import (
    aclose_all_clients,
    aclose_decision_client,
    close_cache,
    close_qdrant_client,
    get_cache,
    get_decision_client,
    get_llm,
    get_qdrant_client,
    get_reranker,
)
from src.clients.llm import REWRITE_MAX_TOKENS, get_response_llm, get_rewrite_llm
from src.clients.reranker import RemoteReranker
from src.utils.vector_store import VectorStoreTools


def test_unit_01_providers_return_process_wide_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each provider resolves the same instance across calls."""
    monkeypatch.setattr(
        "src.clients.llm.get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://openrouter.invalid/api/v1",
            LLM_MODEL="test-model",
            LLM_REASONING_ENABLED=False,
        ),
    )
    monkeypatch.setattr("src.clients.llm._llm_instance", None)

    assert get_cache() is get_cache()
    assert get_qdrant_client() is get_qdrant_client()
    assert get_llm() is get_llm()


def test_unit_07_rewrite_provider_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rewrite client opts out of reasoning so its token cap yields an answer."""
    monkeypatch.setattr(
        "src.clients.llm.get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://openrouter.invalid/api/v1",
            LLM_MODEL="test-model",
            QUERY_REWRITE_MODEL="",
        ),
    )
    monkeypatch.setattr("src.clients.llm._rewrite_llm_instance", None)

    llm = get_rewrite_llm()

    assert llm is not None
    assert llm.extra_body == {"reasoning": {"enabled": False}}
    assert llm.max_tokens == REWRITE_MAX_TOKENS


def test_unit_09_response_provider_requests_stream_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming answer model must ask for usage, or token totals stay empty.

    langchain-openai only enables ``stream_usage`` by default for its own base URL,
    so a custom (OpenRouter) base URL has to request it explicitly.
    """
    monkeypatch.setattr(
        "src.clients.llm.get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://openrouter.invalid/api/v1",
            LLM_MODEL="test-model",
            LLM_REASONING_ENABLED=False,
        ),
    )
    monkeypatch.setattr("src.clients.llm._response_llm_instance", None)

    llm = get_response_llm()

    assert llm is not None
    assert llm.streaming is True
    assert llm.stream_usage is True


def test_unit_02_closes_are_noops_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing providers that were never used does not raise."""
    monkeypatch.setattr("src.clients.qdrant._qdrant_client", None)
    monkeypatch.setattr("src.clients.redis._cache_instance", None)

    close_cache()
    close_qdrant_client()


async def test_unit_03_closes_are_idempotent() -> None:
    """Closing every process-wide client twice does not raise."""
    get_qdrant_client()
    get_cache()

    await aclose_all_clients()
    await aclose_all_clients()


async def test_unit_08_shutdown_closes_decision_client_from_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown on the event loop thread closes the async decision client."""
    monkeypatch.setattr(
        "src.clients.decision.get_settings",
        lambda: SimpleNamespace(DECISION_API_KEY="test-key"),
    )
    monkeypatch.setattr("src.clients.decision._decision_client", None)
    client = get_decision_client()

    await aclose_all_clients()

    assert client._client.is_closed

    replacement = get_decision_client()
    assert replacement is not client

    await aclose_decision_client()
    assert replacement._client.is_closed


def test_unit_04_vector_store_uses_shared_client() -> None:
    """The vector-store facade reuses the process-wide Qdrant client."""
    tools = VectorStoreTools()

    assert tools.client is get_qdrant_client()


def test_unit_05_reranker_provider_returns_process_wide_remote_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reranker provider resolves one shared remote client across calls."""
    monkeypatch.setattr(
        "src.clients.reranker.get_settings",
        lambda: SimpleNamespace(
            RERANKER_MODEL="test-reranker",
            RERANKER_BASE_URL="https://reranker.invalid/api/v1",
            RERANKER_API_KEY="test-key",
        ),
    )
    monkeypatch.setattr("src.clients.reranker._reranker", None)

    reranker = get_reranker()

    assert isinstance(reranker, RemoteReranker)
    assert get_reranker() is reranker
    assert VectorStoreTools()._get_reranker() is reranker


def test_unit_06_reranker_provider_requires_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing reranker model is reported instead of building a broken client."""
    monkeypatch.setattr(
        "src.clients.reranker.get_settings",
        lambda: SimpleNamespace(
            RERANKER_MODEL=None,
            RERANKER_BASE_URL=None,
            RERANKER_API_KEY=None,
        ),
    )
    monkeypatch.setattr("src.clients.reranker._reranker", None)

    with pytest.raises(ValueError, match="RERANKER_MODEL"):
        get_reranker()
