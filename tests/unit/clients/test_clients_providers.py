"""Unit tests for the process-wide client providers."""

from types import SimpleNamespace

import pytest

from src.clients import (
    close_all_clients,
    close_cache,
    close_qdrant_client,
    get_cache,
    get_llm,
    get_qdrant_client,
)
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


def test_unit_02_closes_are_noops_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing providers that were never used does not raise."""
    monkeypatch.setattr("src.clients.qdrant._qdrant_client", None)
    monkeypatch.setattr("src.clients.redis._cache_instance", None)

    close_cache()
    close_qdrant_client()


def test_unit_03_closes_are_idempotent() -> None:
    """Closing every process-wide client twice does not raise."""
    get_qdrant_client()
    get_cache()

    close_all_clients()
    close_all_clients()


def test_unit_04_vector_store_uses_shared_client() -> None:
    """The vector-store facade reuses the process-wide Qdrant client."""
    tools = VectorStoreTools()

    assert tools.client is get_qdrant_client()
