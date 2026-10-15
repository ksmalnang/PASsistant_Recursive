"""Unit tests for report comparison across retrieval strategies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.eval.compare import build_comparison, render_comparison_markdown
from src.eval.models import DatasetSample, EvaluationConfig, RunContext, SampleResult
from src.eval.reporting import build_report

STARTED = datetime(2026, 1, 1, tzinfo=UTC)


def _report(
    strategy: str,
    scores: dict[str, float],
    *,
    started_at: datetime = STARTED,
    top_k: int = 5,
    sample_count: int = 1,
) -> dict[str, Any]:
    results = []
    for index in range(sample_count):
        sample = DatasetSample(id=f"q{index}", user_input=f"Q{index}?", reference="A.")
        result = SampleResult(sample=sample, response="r", retrieved_contexts=["c"])
        result.scores = dict(scores)
        results.append(result)
    context = RunContext(
        strategy=strategy,
        top_k=top_k,
        judge_model="judge",
        embedding_model="embedding",
        pipeline_model="pipeline",
        started_at=started_at,
        duration_seconds=1.0,
    )
    return build_report(EvaluationConfig(dataset_path=Path("d.jsonl")), results, {}, context)


def test_best_strategy_and_delta_against_baseline() -> None:
    comparison = build_comparison(
        [
            _report("similarity", {"faithfulness": 0.5, "context_recall": 0.4}),
            _report("rrf", {"faithfulness": 0.8, "context_recall": 0.3}),
        ]
    )

    entry = comparison["datasets"][0]

    assert entry["best"]["faithfulness"] == "rrf"
    assert entry["best"]["context_recall"] == "similarity"
    assert entry["delta_vs_baseline"]["baseline"] == "similarity"
    assert entry["delta_vs_baseline"]["strategies"]["rrf"]["faithfulness"] == 0.3


def test_latest_run_wins_and_refreshes_sample_count() -> None:
    comparison = build_comparison(
        [
            _report(
                "rrf",
                {"faithfulness": 0.1},
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                sample_count=1,
            ),
            _report(
                "rrf",
                {"faithfulness": 0.9},
                started_at=datetime(2026, 2, 1, tzinfo=UTC),
                sample_count=3,
            ),
        ]
    )

    entry = comparison["datasets"][0]

    assert entry["strategies"]["rrf"]["metrics"]["faithfulness"]["mean"] == 0.9
    assert entry["sample_count"] == 3


def test_warns_when_strategies_differ_on_settings() -> None:
    comparison = build_comparison(
        [
            _report("similarity", {"faithfulness": 0.5}, top_k=5),
            _report("rrf", {"faithfulness": 0.5}, top_k=10),
        ]
    )

    assert comparison["datasets"][0]["warnings"] == [
        "strategies differ on top_k: rrf=10, similarity=5"
    ]


def test_markdown_lists_every_strategy_column() -> None:
    markdown = render_comparison_markdown(
        build_comparison(
            [
                _report("similarity", {"faithfulness": 0.5, "context_recall": 0.4}),
                _report("reranker", {"faithfulness": 0.7, "context_recall": 0.6}),
            ]
        )
    )

    assert "| metric | similarity | reranker | best |" in markdown
    assert "### Delta vs similarity" in markdown
