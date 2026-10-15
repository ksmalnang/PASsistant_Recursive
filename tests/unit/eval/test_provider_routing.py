"""Unit tests for RAGAS OpenRouter provider routing."""

from __future__ import annotations

import pytest

from src.config import get_settings
from src.eval.openrouter import ragas_extra_body


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_llm_inherits_the_app_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "baidu/fp8")
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "")
    get_settings.cache_clear()

    assert ragas_extra_body("llm")["provider"] == {
        "only": ["baidu/fp8"],
        "allow_fallbacks": False,
    }


def test_ragas_llm_provider_overrides_the_app_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "baidu/fp8")
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "deepinfra/fp8, fireworks")
    get_settings.cache_clear()

    assert ragas_extra_body("llm")["provider"]["only"] == ["deepinfra/fp8", "fireworks"]


def test_embeddings_inherit_the_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deepinfra")
    monkeypatch.setenv("RAGAS_EMBEDDING_PROVIDER", "")
    get_settings.cache_clear()

    assert ragas_extra_body("embedding")["provider"] == {
        "order": ["deepinfra"],
        "allow_fallbacks": False,
    }


def test_embeddings_override_the_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deepinfra")
    monkeypatch.setenv("RAGAS_EMBEDDING_PROVIDER", "fireworks")
    get_settings.cache_clear()

    assert ragas_extra_body("embedding")["provider"]["order"] == ["fireworks"]


def test_no_provider_setting_sends_no_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "")
    get_settings.cache_clear()

    assert "provider" not in ragas_extra_body("llm")
