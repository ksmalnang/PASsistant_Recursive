"""OpenRouter transport plus the RAGAS LLM and embedding adapters built on it."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.outputs import Generation, LLMResult
from openai import OpenAI
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from src.eval.models import OpenRouterSettings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
_RETRYABLE_MESSAGES = (
    "cloudflare",
    "ddos",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "timeout",
    "connection",
)


@dataclass
class Usage:
    """Token and cost accounting for every OpenRouter request in a run."""

    chat_requests: int = 0
    embedding_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def request_count(self) -> int:
        return self.chat_requests + self.embedding_requests

    def record_chat_response(self, response: Any) -> None:
        self.chat_requests += 1
        self._record_tokens(getattr(response, "usage", None))
        self.cost_usd += _reported_cost(response)

    def record_embedding_response(self, response: Any) -> None:
        self.embedding_requests += 1
        self._record_tokens(getattr(response, "usage", None))
        self.cost_usd += _reported_cost(response)

    def record_message(self, usage_metadata: dict[str, Any]) -> None:
        """Account for a token usage block reported by a LangChain chat model."""
        self.chat_requests += 1
        prompt = int(usage_metadata.get("input_tokens") or 0)
        completion = int(usage_metadata.get("output_tokens") or 0)
        total = int(usage_metadata.get("total_tokens") or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total or prompt + completion

    def _record_tokens(self, usage: Any) -> None:
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_count"] = self.request_count
        payload["cost_usd"] = round(self.cost_usd, 6)
        return payload


class OpenRouterClient:
    """OpenAI-compatible client with request pacing, retries, and usage accounting."""

    def __init__(self, settings: OpenRouterSettings, usage: Usage | None = None):
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self.usage = usage if usage is not None else Usage()

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        json_mode: bool = False,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Return the first completion's text content."""

        def call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if stop:
                kwargs["stop"] = stop
            if extra_body:
                kwargs["extra_body"] = extra_body
            return self._client.chat.completions.create(**kwargs)

        response = self._request(call, kind="chat", model=model)
        self.usage.record_chat_response(response)
        return response.choices[0].message.content or ""

    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        extra_body: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Return one embedding vector per input text."""

        def call() -> Any:
            kwargs: dict[str, Any] = {"model": model, "input": list(texts)}
            if extra_body:
                kwargs["extra_body"] = extra_body
            return self._client.embeddings.create(**kwargs)

        response = self._request(call, kind="embedding", model=model)
        self.usage.record_embedding_response(response)
        return [list(item.embedding) for item in response.data]

    def _request(self, call: Any, *, kind: str, model: str) -> Any:
        for attempt in range(1, self._settings.max_retries + 1):
            self._wait_for_slot()
            try:
                return call()
            except Exception as exc:
                if attempt == self._settings.max_retries or not _is_retryable(exc):
                    raise
                delay = self._settings.backoff_base_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0.0, self._settings.jitter_seconds)
                logger.warning(
                    "OpenRouter %s request failed (attempt %d/%d, model=%s): %s; retrying in %.2fs",
                    kind,
                    attempt,
                    self._settings.max_retries,
                    model,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable: retry loop always returns or raises")

    def _wait_for_slot(self) -> None:
        with self._lock:
            delay = self._settings.min_interval_seconds
            delay += random.uniform(0.0, self._settings.jitter_seconds)
            wait = self._last_request_at + delay - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.perf_counter()


class OpenRouterRagasLLM(BaseRagasLLM):
    """RAGAS judge LLM that calls OpenRouter once per requested completion.

    RAGAS asks some metrics for several completions (AnswerRelevancy uses n=3);
    issuing them sequentially keeps every completion distinct instead of relying
    on the provider to honour `n`.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        json_mode: bool = True,
        extra_body: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.client = client
        self.model = model
        self.json_mode = json_mode
        self.extra_body = extra_body

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self._generate(prompt, n, temperature, stop)

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._generate,
            prompt,
            n,
            temperature if temperature is not None else 0.01,
            stop,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return bool(getattr(response, "generations", None))

    def _generate(
        self,
        prompt: Any,
        n: int,
        temperature: float,
        stop: list[str] | None,
    ) -> LLMResult:
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        generations = [
            Generation(
                text=self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    temperature=temperature,
                    json_mode=self.json_mode,
                    stop=stop,
                    extra_body=self.extra_body,
                )
            )
            for _ in range(max(n, 1))
        ]
        return LLMResult(generations=[generations])


class OpenRouterEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings backed by the OpenRouter embeddings endpoint."""

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        extra_body: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.client = client
        self.model = model
        self.extra_body = extra_body
        # RAGAS calls embed_text() during testset generation without calling
        # set_run_config(), so the adapter needs a usable default.
        self.run_config = RunConfig()

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(model=self.model, texts=[text], extra_body=self.extra_body)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(model=self.model, texts=list(texts), extra_body=self.extra_body)

    async def aembed_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, list(texts))


class UsageTrackingChatModel:
    """Record token usage from a LangChain chat model without altering it."""

    def __init__(self, model: Any, usage: Usage):
        self._model = model
        self._usage = usage

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._record(self._model.invoke(messages, **kwargs))

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._record(await self._model.ainvoke(messages, **kwargs))

    def _record(self, message: Any) -> Any:
        metadata = getattr(message, "usage_metadata", None)
        if isinstance(metadata, dict):
            self._usage.record_message(metadata)
        return message


def ragas_extra_body(role: str) -> dict[str, Any]:
    """OpenRouter extra_body for a RAGAS call: provider routing plus reasoning controls.

    `role` is "llm" (judge and testset generator) or "embedding". Provider settings fall
    back to the app-wide `LLM_PROVIDER` / `EMBEDDING_PROVIDER`, so RAGAS reaches the same
    provider endpoints as the pipeline it evaluates.
    """
    from src.config import get_settings

    settings = get_settings()
    body: dict[str, Any] = {}

    if role == "embedding":
        configured = settings.RAGAS_EMBEDDING_PROVIDER or settings.EMBEDDING_PROVIDER
        _apply_provider_restriction(body, configured, key="order")
    else:
        configured = settings.RAGAS_LLM_PROVIDER or settings.LLM_PROVIDER
        _apply_provider_restriction(body, configured, key="only")
        reasoning = _reasoning_controls(settings)
        if reasoning is not None:
            body["reasoning"] = reasoning

    return body


def _reasoning_controls(settings: Any) -> dict[str, Any] | None:
    """Reasoning controls for the RAGAS LLM, honouring the RAGAS_* overrides."""
    enabled = settings.RAGAS_LLM_REASONING_ENABLED
    if enabled is None:
        enabled = settings.LLM_REASONING_ENABLED
    if not enabled:
        return None

    exclude = settings.RAGAS_LLM_REASONING_EXCLUDE
    if exclude is None:
        exclude = settings.LLM_REASONING_EXCLUDE

    reasoning: dict[str, Any] = {"exclude": exclude}
    effort = settings.RAGAS_LLM_REASONING_EFFORT
    if effort is not None:
        reasoning["effort"] = effort
    else:
        max_tokens = settings.RAGAS_LLM_REASONING_MAX_TOKENS
        if max_tokens is None:
            max_tokens = settings.LLM_REASONING_MAX_TOKENS
        if max_tokens is not None:
            reasoning["max_tokens"] = max_tokens
        else:
            reasoning["effort"] = settings.LLM_REASONING_EFFORT
    return reasoning


def _apply_provider_restriction(body: dict[str, Any], configured: str, *, key: str) -> None:
    """Restrict a request to the OpenRouter provider endpoints named in a setting."""
    providers = [name.strip() for name in configured.split(",") if name.strip()]
    if providers:
        body["provider"] = {key: providers, "allow_fallbacks": False}


def _is_retryable(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in _RETRYABLE_STATUS:
        return True
    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_MESSAGES)


def _reported_cost(response: Any) -> float:
    payload = response.model_dump() if hasattr(response, "model_dump") else response
    costs = _collect_costs(payload)
    return max(costs) if costs else 0.0


def _collect_costs(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        return []
    costs: list[float] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            costs.extend(_collect_costs(value))
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "cost" in str(key).lower()
        ):
            costs.append(float(value))
    return costs
