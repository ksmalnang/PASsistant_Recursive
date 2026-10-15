"""Unit tests for OpenRouter transport, accounting, and the RAGAS judge adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.eval.models import OpenRouterSettings
from src.eval.openrouter import (
    OpenRouterClient,
    OpenRouterEmbeddings,
    OpenRouterRagasLLM,
    Usage,
    UsageTrackingChatModel,
    _is_retryable,
)


def _chat_response(prompt: int, completion: int, cost: float) -> Any:
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )
    return SimpleNamespace(usage=usage, model_dump=lambda: {"usage": {"cost": cost}})


def test_usage_accumulates_tokens_and_cost() -> None:
    usage = Usage()

    usage.record_chat_response(_chat_response(10, 5, 0.01))
    usage.record_embedding_response(_chat_response(4, 0, 0.0))
    usage.record_message({"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})

    assert usage.chat_requests == 2
    assert usage.embedding_requests == 1
    assert usage.request_count == 3
    assert usage.prompt_tokens == 17
    assert usage.total_tokens == 24
    assert usage.as_dict()["cost_usd"] == 0.01


def test_retryable_error_detection() -> None:
    assert _is_retryable(_HttpError(429)) is True
    assert _is_retryable(_HttpError(503)) is True
    assert _is_retryable(RuntimeError("rate limit exceeded")) is True
    assert _is_retryable(ValueError("invalid request")) is False


def test_client_retries_transient_failure_then_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenRouterClient(OpenRouterSettings(api_key="test"))
    monkeypatch.setattr(client, "_wait_for_slot", lambda: None)
    monkeypatch.setattr("src.eval.openrouter.time.sleep", lambda _seconds: None)

    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporarily unavailable")
        return "ok"

    assert client._request(flaky, kind="chat", model="m") == "ok"
    assert attempts["count"] == 2


def test_client_does_not_retry_permanent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenRouterClient(OpenRouterSettings(api_key="test"))
    monkeypatch.setattr(client, "_wait_for_slot", lambda: None)

    def boom() -> str:
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        client._request(boom, kind="chat", model="m")


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> str:
        self.calls += 1
        self.last_kwargs = kwargs
        return f"completion-{self.calls}"

    def embed(self, *, model: str, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_judge_adapter_issues_one_call_per_completion() -> None:
    fake = _FakeClient()
    llm = OpenRouterRagasLLM(fake, model="judge")

    result = llm.generate_text("prompt", n=3)

    assert [generation.text for generation in result.generations[0]] == [
        "completion-1",
        "completion-2",
        "completion-3",
    ]
    assert llm.is_finished(result) is True
    assert fake.last_kwargs["json_mode"] is True


async def test_embeddings_adapter_can_embed_without_set_run_config() -> None:
    embeddings = OpenRouterEmbeddings(_FakeClient(), model="embed")

    assert await embeddings.embed_text("hello") == [0.0]


def test_judge_adapter_forwards_reasoning_controls() -> None:
    fake = _FakeClient()
    extra_body = {"reasoning": {"exclude": True}}
    llm = OpenRouterRagasLLM(fake, model="judge", extra_body=extra_body)

    llm.generate_text("prompt")

    assert fake.last_kwargs["extra_body"] == extra_body


class _FakeMessage:
    usage_metadata = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


class _FakeChatModel:
    def invoke(self, _messages: Any, **_kwargs: Any) -> _FakeMessage:
        return _FakeMessage()

    async def ainvoke(self, _messages: Any, **_kwargs: Any) -> _FakeMessage:
        return _FakeMessage()


def test_usage_tracking_chat_model_counts_invoke() -> None:
    usage = Usage()

    UsageTrackingChatModel(_FakeChatModel(), usage).invoke([])

    assert usage.chat_requests == 1
    assert usage.total_tokens == 5


async def test_usage_tracking_chat_model_counts_ainvoke() -> None:
    usage = Usage()

    await UsageTrackingChatModel(_FakeChatModel(), usage).ainvoke([])

    assert usage.chat_requests == 1
    assert usage.prompt_tokens == 3


class _HttpError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
