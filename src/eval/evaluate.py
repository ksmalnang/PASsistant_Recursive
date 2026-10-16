"""Run a RAGAS evaluation over the pipeline and write an analysis report."""

from __future__ import annotations

import logging
import math
import time
import warnings
from datetime import UTC, datetime
from typing import Any

from src.eval.dataset import load_dataset
from src.eval.models import (
    DatasetSample,
    EvaluationConfig,
    OpenRouterSettings,
    RunContext,
    SampleResult,
)
from src.eval.openrouter import (
    OpenRouterClient,
    OpenRouterEmbeddings,
    OpenRouterRagasLLM,
    Usage,
    ragas_extra_body,
)
from src.eval.pipeline import LivePipeline, apply_fixtures, load_fixtures, strip_citations
from src.eval.reporting import build_report, write_report

logger = logging.getLogger(__name__)

_NON_METRIC_COLUMNS = {
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "reference_contexts",
    "rubrics",
}


async def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate one dataset and return the report that was written to disk."""
    started_at = datetime.now(UTC)
    started = time.perf_counter()

    samples = load_dataset(config.dataset_path)
    usage = Usage()
    client = OpenRouterClient(config.openrouter or default_openrouter_settings(), usage)
    results = await _collect_results(samples, config, usage)

    from src.config import get_settings

    settings = get_settings()
    judge_model = settings.RAGAS_LLM_MODEL or settings.LLM_MODEL
    embedding_model = settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL
    await _score(results, client, judge_model, embedding_model)

    context = RunContext(
        strategy=settings.RETRIEVAL_STRATEGY,
        top_k=settings.RETRIEVAL_TOP_K,
        judge_model=judge_model,
        embedding_model=embedding_model,
        pipeline_model=settings.LLM_MODEL,
        started_at=started_at,
        duration_seconds=time.perf_counter() - started,
    )
    report = build_report(config, results, usage.as_dict(), context)
    write_report(report, config)
    _log_summary(report)
    return report


def default_openrouter_settings() -> OpenRouterSettings:
    """Build transport settings from the application configuration."""
    from src.config import get_settings

    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY is required for evaluation")
    return OpenRouterSettings(api_key=api_key, base_url=settings.OPENAI_BASE_URL)


def build_metrics() -> list[Any]:
    """Return the three RAGAS metrics whose fields the testset provides.

    Faithfulness keeps a parse-retry budget for malformed judge output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (  # type: ignore[import-untyped]
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

    return [
        Faithfulness(max_retries=3),
        ContextRecall(),
        FactualCorrectness(),
    ]


async def _collect_results(
    samples: list[DatasetSample],
    config: EvaluationConfig,
    usage: Usage,
) -> list[SampleResult]:
    if config.mode == "live":
        return await LivePipeline(usage=usage).run(samples)
    fixtures = load_fixtures(config.fixture_path) if config.fixture_path else {}
    return apply_fixtures(samples, fixtures)


async def _score(
    results: list[SampleResult],
    client: OpenRouterClient,
    judge_model: str,
    embedding_model: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, SingleTurnSample, aevaluate

    metrics = build_metrics()
    judge = OpenRouterRagasLLM(client, judge_model, extra_body=ragas_extra_body("llm"))
    embeddings = OpenRouterEmbeddings(
        client, embedding_model, extra_body=ragas_extra_body("embedding")
    )
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=result.sample.user_input,
                retrieved_contexts=result.retrieved_contexts,
                response=strip_citations(result.response),
                reference=result.sample.reference,
                reference_contexts=list(result.sample.reference_contexts),
            )
            for result in results
        ]
    )
    logger.info(
        "Scoring %d samples with %d metrics [%s] using judge=%s",
        len(results),
        len(metrics),
        ", ".join(type(metric).__name__ for metric in metrics),
        judge_model,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        outcome = await aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )

    frame = outcome.to_pandas()
    if len(frame) != len(results):
        raise RuntimeError(f"RAGAS returned {len(frame)} score rows for {len(results)} samples")
    _verify_row_order(results, frame)
    _apply_scores(results, frame)


def _verify_row_order(results: list[SampleResult], frame: Any) -> None:
    """Refuse to attribute scores if RAGAS ever returns rows out of dataset order."""
    if "user_input" not in frame.columns:
        return
    for position, result in enumerate(results):
        if frame.iloc[position]["user_input"] != result.sample.user_input:
            raise RuntimeError(
                f"RAGAS score rows are out of order at position {position}; "
                "refusing to attribute scores to the wrong sample"
            )


def _apply_scores(results: list[SampleResult], frame: Any) -> None:
    metric_columns = [
        column
        for column in frame.columns
        if column not in _NON_METRIC_COLUMNS
        and any(_as_score(value) is not None for value in frame[column])
    ]
    logger.info("RAGAS score columns: %s", ", ".join(metric_columns))
    for position, result in enumerate(results):
        row = frame.iloc[position]
        for column in metric_columns:
            score = _as_score(row[column])
            if score is None:
                result.failures[column] = "RAGAS returned no score"
            else:
                result.scores[column] = round(score, 4)


def _as_score(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def _log_summary(report: dict[str, Any]) -> None:
    metrics = report["aggregate"]["metrics"]
    summary = " | ".join(
        f"{metric}={entry['mean']:.3f} ({entry['scored']} scored)"
        for metric, entry in metrics.items()
    )
    logger.info("Aggregate scores: %s", summary or "none")
    logger.info("Usage: %s", report["usage"])
