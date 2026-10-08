"""Unit tests for RemoteReranker with OpenRouter provider and Jina fallback."""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.clients.reranker import RemoteReranker, get_reranker


def test_unit_reranker_openrouter_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter returns scores aligned to original document order."""
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        assert "openrouter.ai" in url
        request = httpx.Request("POST", url)
        data = {
            "id": "gen-rerank-123",
            "model": "cohere/rerank-v3.5",
            "results": [
                {"index": 1, "relevance_score": 0.92, "document": {"text": "Doc B"}},
                {"index": 0, "relevance_score": 0.35, "document": {"text": "Doc A"}},
            ],
        }
        return httpx.Response(200, json=data, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)

    reranker = RemoteReranker(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="cohere/rerank-v3.5",
        provider_name="openrouter",
    )

    scores = reranker.rerank(query="test query", documents=["Doc A", "Doc B"])
    # Scores must be ordered matching input docs: Doc A (0.35), Doc B (0.92)
    assert scores == [0.35, 0.92]


def test_unit_reranker_fallback_to_jina_on_openrouter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OpenRouter fails, it falls back to Jina and returns fallback scores."""
    called_endpoints: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        called_endpoints.append(url)
        request = httpx.Request("POST", url)
        if "openrouter.ai" in url:
            return httpx.Response(502, text="Bad Gateway", request=request)
        if "api.jina.ai" in url:
            data = {
                "model": "jina-reranker-v3",
                "results": [
                    {"index": 0, "relevance_score": 0.88},
                    {"index": 1, "relevance_score": 0.15},
                ],
            }
            return httpx.Response(200, json=data, request=request)
        raise ValueError(f"Unexpected endpoint: {url}")

    monkeypatch.setattr(httpx, "post", fake_post)

    jina_fallback = RemoteReranker(
        base_url="https://api.jina.ai/v1",
        api_key="jina_test",
        model="jina-reranker-v3",
        provider_name="jina",
    )
    openrouter_reranker = RemoteReranker(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="cohere/rerank-v3.5",
        fallback=jina_fallback,
        provider_name="openrouter",
    )

    scores = openrouter_reranker.rerank(
        query="test query", documents=["Doc A", "Doc B"]
    )
    assert scores == [0.88, 0.15]
    assert len(called_endpoints) == 2
    assert "openrouter.ai" in called_endpoints[0]
    assert "api.jina.ai" in called_endpoints[1]


def test_unit_reranker_both_primary_and_fallback_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both OpenRouter and Jina fail, a RuntimeError is raised."""
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(500, text="Internal Server Error", request=request)

    monkeypatch.setattr(httpx, "post", fake_post)

    jina_fallback = RemoteReranker(
        base_url="https://api.jina.ai/v1",
        api_key="jina_test",
        model="jina-reranker-v3",
        provider_name="jina",
    )
    openrouter_reranker = RemoteReranker(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="cohere/rerank-v3.5",
        fallback=jina_fallback,
        provider_name="openrouter",
    )

    with pytest.raises(RuntimeError, match="Both primary reranker.*and fallback reranker"):
        openrouter_reranker.rerank(query="test", documents=["Doc A", "Doc B"])


def test_unit_reranker_empty_documents_returns_empty_list() -> None:
    """Empty document lists return empty results without network calls."""
    reranker = RemoteReranker(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        model="cohere/rerank-v3.5",
    )
    assert reranker.rerank(query="test", documents=[]) == []


def test_unit_get_reranker_creates_openrouter_with_jina_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_reranker configures OpenRouter with Jina fallback when credentials exist."""
    monkeypatch.setattr(
        "src.clients.reranker.get_settings",
        lambda: SimpleNamespace(
            RERANKER_PROVIDER="openrouter",
            RERANKER_MODEL="cohere/rerank-v3.5",
            RERANKER_BASE_URL="https://openrouter.ai/api/v1",
            RERANKER_API_KEY=None,
            OPENAI_API_KEY="sk-or-test-key",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            RERANKER_FALLBACK_PROVIDER="jina",
            RERANKER_FALLBACK_MODEL="jina-reranker-v3",
            RERANKER_FALLBACK_BASE_URL="https://api.jina.ai/v1",
            RERANKER_FALLBACK_API_KEY=None,
            JINA_API_KEY="jina_test_key",
        ),
    )
    monkeypatch.setattr("src.clients.reranker._reranker", None)

    reranker = get_reranker()
    assert isinstance(reranker, RemoteReranker)
    assert reranker.provider_name == "openrouter"
    assert reranker.model == "cohere/rerank-v3.5"
    assert reranker.fallback is not None
    assert reranker.fallback.provider_name == "jina"
    assert reranker.fallback.model == "jina-reranker-v3"


def test_unit_get_reranker_uses_jina_directly_when_no_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_reranker uses Jina directly if OpenRouter API key is not configured."""
    monkeypatch.setattr(
        "src.clients.reranker.get_settings",
        lambda: SimpleNamespace(
            RERANKER_PROVIDER="openrouter",
            RERANKER_MODEL="cohere/rerank-v3.5",
            RERANKER_BASE_URL="https://openrouter.ai/api/v1",
            RERANKER_API_KEY=None,
            OPENAI_API_KEY=None,
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            RERANKER_FALLBACK_PROVIDER="jina",
            RERANKER_FALLBACK_MODEL="jina-reranker-v3",
            RERANKER_FALLBACK_BASE_URL="https://api.jina.ai/v1",
            RERANKER_FALLBACK_API_KEY=None,
            JINA_API_KEY="jina_test_key",
        ),
    )
    monkeypatch.setattr("src.clients.reranker._reranker", None)

    reranker = get_reranker()
    assert isinstance(reranker, RemoteReranker)
    assert reranker.provider_name == "jina"
    assert reranker.model == "jina-reranker-v3"
