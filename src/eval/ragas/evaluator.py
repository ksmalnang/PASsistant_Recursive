from __future__ import annotations

import functools
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from src.eval.ragas.data import load_dataset, load_fixtures
from src.eval.ragas.models import ProgressTracker, RagasEvalConfig, RagasEvalSample
from src.eval.ragas.reporting import build_report, coerce_score, save_report

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[\d+\]")
_SOURCES_SECTION_RE = re.compile(
    r"\n\s*(?:Sources|Sumber|References)\s*:\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_citation_markers(text: str) -> str:
    """Remove inline numeric citation markers and trailing Sources section."""
    cleaned = _SOURCES_SECTION_RE.sub("", text)
    return _CITATION_RE.sub("", cleaned).strip()


class _EmbeddingAdapter:
    """Minimal embedding interface compatible with the RAGAS evaluate() path used here."""

    def __init__(self, adapter_client: Any, adapter_model: str):
        self.client = adapter_client
        self.model = adapter_model
        self.run_config: Any = None

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return list(response.data[0].embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [list(item.embedding) for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def set_run_config(self, run_config: Any) -> None:
        self.run_config = run_config


class _DirectOpenRouterChatLLM:
    """Invoke-compatible chat adapter used by live eval generation."""

    def __init__(self, client: Any, model: str, temperature: float = 0.3):
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list[BaseMessage], /, **kwargs: Any) -> AIMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[self._format_message(message) for message in messages],
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content or ""
        return AIMessage(content=content)

    def _format_message(self, message: BaseMessage) -> dict[str, Any]:
        role = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }.get(getattr(message, "type", ""), "user")
        content = getattr(message, "content", "")
        return {"role": role, "content": content if isinstance(content, str) else str(content)}


@functools.lru_cache(maxsize=1)
def _build_sequential_llm_class() -> Any:
    """Build the RAGAS LLM adapter lazily to avoid import-time RAGAS dependency."""
    import asyncio

    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    @dataclass
    class _SequentialOpenRouterLLM(BaseRagasLLM):
        openrouter_client: Any = None
        openrouter_model: str = ""

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            raise NotImplementedError("_SequentialOpenRouterLLM only supports async generation")

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            if self.openrouter_client is None:
                raise ValueError("openrouter_client is required")

            prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            generations: list[Any] = []
            loop = asyncio.get_running_loop()

            for _ in range(n):
                kwargs: dict[str, Any] = {
                    "model": self.openrouter_model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": temperature,
                    "n": 1,
                    "response_format": {"type": "json_object"},
                }
                if stop is not None:
                    kwargs["stop"] = stop

                response = await loop.run_in_executor(
                    None,
                    lambda kw=kwargs: self.openrouter_client.chat.completions.create(**kw),
                )
                text = response.choices[0].message.content or ""
                generations.append(Generation(text=text))

            return LLMResult(generations=[generations])

        def is_finished(self, response: Any) -> bool:
            return bool(getattr(response, "generations", None))

    return _SequentialOpenRouterLLM


class RagasEvaluator:
    """Orchestrates RAGAS evaluation of the RAG pipeline."""

    def __init__(self, config: RagasEvalConfig):
        self.config = config
        self._llm: Any = None
        self._embeddings: Any = None
        self._retriever: Any = None
        self._retrieval_node: Any = None
        self._response_service: Any = None
        self._request_lock = threading.Lock()
        self._last_openrouter_request_at = 0.0
        self._usage_summary: dict[str, Any] = {
            "request_count": 0,
            "llm_request_count": 0,
            "embedding_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd_reported": 0.0,
            "cost_fields": [],
        }

    def _build_ragas_reasoning_extra_body(self) -> dict[str, Any] | None:
        from src.config import get_settings

        settings = get_settings()
        enabled = settings.RAGAS_LLM_REASONING_ENABLED
        if enabled is None:
            enabled = settings.LLM_REASONING_ENABLED
        if not enabled:
            return None

        exclude = settings.RAGAS_LLM_REASONING_EXCLUDE
        if exclude is None:
            exclude = settings.LLM_REASONING_EXCLUDE

        reasoning: dict[str, Any] = {"exclude": exclude}

        # If a RAGAS-specific effort is set, it takes priority and skips max_tokens entirely.
        # This allows setting RAGAS_LLM_REASONING_EFFORT=none to fully suppress reasoning
        # even when LLM_REASONING_MAX_TOKENS has a non-None global default.
        ragas_effort = settings.RAGAS_LLM_REASONING_EFFORT
        if ragas_effort is not None:
            reasoning["effort"] = ragas_effort
        else:
            max_tokens = settings.RAGAS_LLM_REASONING_MAX_TOKENS
            if max_tokens is None:
                max_tokens = settings.LLM_REASONING_MAX_TOKENS

            if max_tokens is not None:
                reasoning["max_tokens"] = max_tokens
            else:
                effort = settings.LLM_REASONING_EFFORT
                reasoning["effort"] = effort

        return {"reasoning": reasoning}

    async def run(self) -> dict[str, Any]:
        run_start = time.perf_counter()
        logger.info(
            "═══ RAGAS Evaluation Started ═══ dataset=%s | mode=%s | metrics=%s | samples=loading...",
            self.config.dataset_path.name,
            self.config.mode,
            self.config.metrics_tier,
        )
        samples = load_dataset(self.config.dataset_path)
        logger.info(
            "Dataset loaded: %d samples from %s",
            len(samples),
            self.config.dataset_path.name,
        )

        pipeline_start = time.perf_counter()
        samples = await self._execute_pipeline(samples)
        pipeline_elapsed = time.perf_counter() - pipeline_start
        logger.info(
            "Pipeline phase completed in %.1fs — %d samples processed, %d with responses",
            pipeline_elapsed,
            len(samples),
            sum(1 for s in samples if s.response),
        )

        eval_start = time.perf_counter()
        results = self._evaluate_with_ragas(samples)
        eval_elapsed = time.perf_counter() - eval_start
        logger.info("RAGAS scoring phase completed in %.1fs", eval_elapsed)

        report = self._build_report(samples, results)
        self._save_report(report)

        total_elapsed = time.perf_counter() - run_start
        aggregate = report.get("aggregate_scores", {})
        usage = report.get("openrouter_usage", {})
        logger.info(
            "═══ RAGAS Evaluation Finished ═══ "
            "total_time=%.1fs | samples=%d | requests=%d | tokens=%d | cost=$%.4f",
            total_elapsed,
            len(samples),
            usage.get("request_count", 0),
            usage.get("total_tokens", 0),
            usage.get("cost_usd_reported", 0.0),
        )
        logger.info(
            "Aggregate scores: %s",
            " | ".join(f"{k}={v:.3f}" for k, v in aggregate.items()),
        )
        return report

    async def _execute_pipeline(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        if self.config.mode == "fixture":
            return self._apply_fixtures(samples)
        return await self._run_live_pipeline(samples)

    def _apply_fixtures(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        progress = ProgressTracker(len(samples), "Fixture")
        if self.config.fixture_path is None:
            logger.info(
                "Fixture mode: using dataset-provided contexts and responses (no fixture file)"
            )
            for sample in samples:
                if sample.response:
                    sample.retrieved_contexts = list(sample.provided_contexts)
                progress.advance(sample.id)
            return samples

        fixtures = load_fixtures(self.config.fixture_path)
        logger.info(
            "Fixture mode: loading pre-computed responses from %s", self.config.fixture_path.name
        )
        for sample in samples:
            fixture = fixtures.get(sample.id, {})
            sample.response = fixture.get("response", "")
            sample.retrieved_contexts = fixture.get("retrieved_contexts", [])
            progress.advance(sample.id)
        return samples

    async def _run_live_pipeline(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        self._init_pipeline_services()
        progress = ProgressTracker(len(samples), "Pipeline")
        logger.info(
            "Live pipeline: running retrieval + response generation for %d samples", len(samples)
        )

        for idx, sample in enumerate(samples, start=1):
            sample_start = time.perf_counter()
            try:
                from langchain_core.messages import HumanMessage

                from src.utils.state import AgentState

                state = AgentState(
                    messages=[HumanMessage(content=sample.question)],
                    requires_retrieval=True,
                    retrieval_query=sample.question,
                    current_intent="query_document",
                )
                retrieval_updates = await self._retrieval_node.run(state)
                state = state.model_copy(update=retrieval_updates)
                sample.retrieved_contexts = [
                    chunk.get("text", "") or chunk.get("child_text", "")
                    for chunk in state.retrieved_chunks
                ]

                result = self._response_service.generate(state)
                sample.response = result.get("draft_response", "")
                sample_elapsed = time.perf_counter() - sample_start
                logger.info(
                    "[%d/%d] %s — retrieved=%d chunks, response=%d chars (%.1fs)",
                    idx,
                    len(samples),
                    sample.id,
                    len(sample.retrieved_contexts),
                    len(sample.response),
                    sample_elapsed,
                )
            except Exception as exc:
                sample_elapsed = time.perf_counter() - sample_start
                logger.warning(
                    "[%d/%d] %s — FAILED after %.1fs: %s",
                    idx,
                    len(samples),
                    sample.id,
                    sample_elapsed,
                    exc,
                )
                sample.response = ""
                sample.retrieved_contexts = []
            finally:
                progress.advance(sample.id)

        return samples

    def _init_pipeline_services(self) -> None:
        if self._retriever is None:
            from src.utils.nodes.retrieval import RetrievalNode
            from src.utils.vector_store import VectorStoreTools

            self._retriever = VectorStoreTools()
            self._retrieval_node = RetrievalNode(retriever=self._retriever)
            logger.info(
                "Pipeline retriever initialized: strategy=%s, top_k=%d",
                self._retriever.retrieval_strategy,
                self._retrieval_node.top_k,
            )
        if self._response_service is None:
            from src.services.response_generation import ResponseGenerationService

            self._response_service = ResponseGenerationService(llm_provider=self._get_pipeline_llm)

    def _get_pipeline_llm(self) -> Any:
        from src.config import get_settings

        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None

        model = settings.LLM_MODEL
        client = self._build_tracked_openrouter_client(model)
        logger.info("Pipeline response LLM ready: model=%s", model)
        return _DirectOpenRouterChatLLM(client=client, model=model)

    def _evaluate_with_ragas(self, samples: list[RagasEvalSample]) -> Any:
        import pandas as pd
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        metrics = self._select_metrics()
        evaluator_llm = self._get_evaluator_llm()
        evaluator_embeddings = self._get_evaluator_embeddings()

        ragas_samples = [
            SingleTurnSample(
                user_input=sample.question,
                retrieved_contexts=sample.retrieved_contexts,
                response=_strip_citation_markers(sample.response),
                reference=sample.ground_truth,
                reference_contexts=sample.reference_contexts,
            )
            for sample in samples
        ]

        metric_names = [type(m).__name__ for m in metrics]
        logger.info(
            "RAGAS scoring: %d samples × %d metrics [%s] — batch mode, json_object enforced",
            len(ragas_samples),
            len(metrics),
            ", ".join(metric_names),
        )
        eval_dataset = EvaluationDataset(samples=ragas_samples)
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        frame = result.to_pandas()
        frame.insert(0, "sample_id", [sample.id for sample in samples])

        # Log per-sample scores for quick visibility
        for idx, sample in enumerate(samples):
            row = frame.iloc[idx]
            scores_str = " | ".join(
                f"{col}={coerce_score(row[col]):.2f}"
                for col in frame.columns
                if col != "sample_id" and coerce_score(row.get(col)) is not None
            )
            logger.info("  %s: %s", sample.id, scores_str or "no numeric scores")

        return frame

    def _before_openrouter_request(self, kind: str, model: str) -> None:
        with self._request_lock:
            now = time.perf_counter()
            min_delay = self.config.openrouter_min_interval_seconds + random.uniform(
                0.0, self.config.openrouter_jitter_seconds
            )
            wait_for = max((self._last_openrouter_request_at + min_delay) - now, 0.0)
            if wait_for > 0:
                logger.debug(
                    "Throttling OpenRouter %s request for %.2fs (model=%s)",
                    kind,
                    wait_for,
                    model,
                )
                time.sleep(wait_for)
            self._last_openrouter_request_at = time.perf_counter()

    def _should_retry_openrouter_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "cloudflare",
                "ddos",
                "rate limit",
                "too many requests",
                "temporarily unavailable",
                "timeout",
                "connection",
            )
        )

    def _extract_openrouter_usage(self, response: Any, headers: Any, kind: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        self._usage_summary["request_count"] += 1
        if kind == "llm":
            self._usage_summary["llm_request_count"] += 1
        else:
            self._usage_summary["embedding_request_count"] += 1
        self._usage_summary["prompt_tokens"] += prompt_tokens
        self._usage_summary["completion_tokens"] += completion_tokens
        self._usage_summary["total_tokens"] += total_tokens

        payload: dict[str, Any] = {}
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump()
            except Exception:
                payload = {}

        candidate_costs: list[float] = []
        for source in (payload, payload.get("usage", {})):
            if isinstance(source, dict):
                for key, value in source.items():
                    if "cost" in str(key).lower() and isinstance(value, (int, float)):
                        candidate_costs.append(float(value))

        if headers is not None:
            for key, value in headers.items():
                if "cost" in str(key).lower():
                    try:
                        candidate_costs.append(float(value))
                    except (TypeError, ValueError):
                        continue

        if candidate_costs:
            reported_cost = max(candidate_costs)
            self._usage_summary["cost_usd_reported"] += reported_cost
            self._usage_summary["cost_fields"].append({"kind": kind, "cost_usd": reported_cost})

    def _wrap_openrouter_create(
        self,
        create_fn: Any,
        raw_create_fn: Any,
        kind: str,
        model: str,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, self.config.openrouter_max_retries + 1):
                self._before_openrouter_request(kind, model)
                try:
                    if extra_body is not None:
                        existing_extra_body = kwargs.get("extra_body")
                        if isinstance(existing_extra_body, dict):
                            kwargs["extra_body"] = {**existing_extra_body, **extra_body}
                        else:
                            kwargs["extra_body"] = dict(extra_body)

                    if raw_create_fn is not None:
                        raw_response = raw_create_fn(*args, **kwargs)
                        parsed = raw_response.parse()
                        self._extract_openrouter_usage(
                            parsed, getattr(raw_response, "headers", {}), kind
                        )
                        return parsed

                    response = create_fn(*args, **kwargs)
                    self._extract_openrouter_usage(response, {}, kind)
                    return response
                except Exception as exc:
                    last_exc = exc
                    if not self._should_retry_openrouter_error(exc):
                        raise
                    sleep_seconds = self.config.openrouter_backoff_base_seconds * (
                        2 ** (attempt - 1)
                    )
                    sleep_seconds += random.uniform(0.0, self.config.openrouter_jitter_seconds)
                    logger.warning(
                        "OpenRouter %s request failed (attempt %d/%d, model=%s): %s; retrying in %.2fs",
                        kind,
                        attempt,
                        self.config.openrouter_max_retries,
                        model,
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)

            assert last_exc is not None
            raise last_exc

        return _wrapped_create

    def _build_tracked_openrouter_client(self, model: str) -> Any:
        from openai import OpenAI

        from src.config import get_settings

        settings = get_settings()
        ragas_reasoning_extra_body = self._build_ragas_reasoning_extra_body()
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

        client.chat.completions.create = self._wrap_openrouter_create(
            client.chat.completions.create,
            getattr(getattr(client.chat.completions, "with_raw_response", None), "create", None),
            "llm",
            model,
            ragas_reasoning_extra_body,
        )
        client.embeddings.create = self._wrap_openrouter_create(
            client.embeddings.create,
            getattr(getattr(client.embeddings, "with_raw_response", None), "create", None),
            "embedding",
            model,
        )
        return client

    def _select_metrics(self) -> list[Any]:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (  # type: ignore[import-untyped]
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )

        # Faithfulness: max_retries=3 records the intended retry policy. In RAGAS 0.4.3,
        # the parse retry loop is actually reached by using the BaseRagasLLM path, where
        # PydanticPrompt.generate_multiple defaults retries_left=3 and invokes
        # RagasOutputParser for malformed JSON repair.
        # AnswerRelevancy: strictness=3 keeps the 3-question cosine-similarity average.
        # _SequentialOpenRouterLLM converts generate(n=3) into 3 sequential n=1 calls,
        # so all 3 questions are genuinely generated instead of the Instructor path's
        # silent "1 question repeated" behaviour.
        core: list[Any] = [
            Faithfulness(max_retries=3),
            AnswerRelevancy(strictness=3),
            ContextPrecision(),
            ContextRecall(),
        ]

        if self.config.metrics_tier == "extended":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from ragas.metrics import (  # type: ignore[import-untyped]
                    ContextEntityRecall,
                    FactualCorrectness,
                    NoiseSensitivity,
                    SemanticSimilarity,
                )

            core.extend(
                [
                    FactualCorrectness(),
                    SemanticSimilarity(),
                    ContextEntityRecall(),
                    NoiseSensitivity(),
                ]
            )

        return core

    def _get_evaluator_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        from src.config import get_settings

        settings = get_settings()
        model = settings.RAGAS_LLM_MODEL or settings.LLM_MODEL

        client = self._build_tracked_openrouter_client(model)
        sequential_llm = _build_sequential_llm_class()
        self._llm = sequential_llm(openrouter_client=client, openrouter_model=model)
        logger.info(
            "Evaluator LLM ready: model=%s | json_mode=on | throttle=%.2fs+%.2fs jitter",
            model,
            self.config.openrouter_min_interval_seconds,
            self.config.openrouter_jitter_seconds,
        )

        return self._llm

    def _get_evaluator_embeddings(self) -> Any:
        if self._embeddings is not None:
            return self._embeddings

        from src.config import get_settings

        settings = get_settings()
        model = settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL

        try:
            client = self._build_tracked_openrouter_client(model)
            self._embeddings = _EmbeddingAdapter(client, model)
        except Exception as exc:
            logger.warning(
                "Tracked embedding adapter setup failed; falling back to adapter: %s",
                exc,
            )
            client = self._build_tracked_openrouter_client(model)
            self._embeddings = _EmbeddingAdapter(client, model)

        logger.info("Evaluator embeddings ready: model=%s", model)
        return self._embeddings

    def _build_report(self, samples: list[RagasEvalSample], results: Any) -> dict[str, Any]:
        return build_report(self.config, samples, results, self._usage_summary)

    def _get_metric_columns(self, scores_df: Any) -> list[str]:
        from src.eval.ragas.reporting import get_metric_columns

        return get_metric_columns(scores_df)

    def _build_aggregate_scores(
        self,
        scores_df: Any,
        metric_columns: list[str],
    ) -> dict[str, float]:
        from src.eval.ragas.reporting import build_aggregate_scores

        return build_aggregate_scores(scores_df, metric_columns)

    def _build_per_sample_breakdown(
        self,
        samples: list[RagasEvalSample],
        scores_df: Any,
        metric_columns: list[str],
    ) -> list[dict[str, Any]]:
        from src.eval.ragas.reporting import build_per_sample_breakdown

        return build_per_sample_breakdown(samples, scores_df, metric_columns)

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        return coerce_score(value)

    def _save_report(self, report: dict[str, Any]) -> None:
        save_report(self.config.output_path, report)
