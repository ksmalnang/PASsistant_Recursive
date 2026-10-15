"""Unit tests for report assembly and artifact writing."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from src.eval.models import DatasetSample, EvaluationConfig, RunContext, SampleResult
from src.eval.reporting import build_report, render_markdown, write_report


def _result(
    sample_id: str,
    *,
    question_type: str,
    faithfulness: float | None,
) -> SampleResult:
    sample = DatasetSample(
        id=sample_id,
        user_input=f"Question {sample_id}?",
        reference="Reference answer.",
        metadata={"question_type": question_type, "topic": "skripsi"},
    )
    result = SampleResult(sample=sample, response="Model answer.", retrieved_contexts=["ctx"])
    result.scores["context_recall"] = 0.5
    if faithfulness is None:
        result.failures["faithfulness"] = "RAGAS returned no score"
    else:
        result.scores["faithfulness"] = faithfulness
    return result


def _report(results: list[SampleResult]) -> dict:
    config = EvaluationConfig(dataset_path=Path("tests/fixtures/example.jsonl"))
    context = RunContext(
        strategy="rrf",
        top_k=10,
        judge_model="judge-model",
        embedding_model="embedding-model",
        pipeline_model="pipeline-model",
        started_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        duration_seconds=12.5,
    )
    return build_report(config, results, {"total_tokens": 10}, context)


def test_aggregate_ignores_missing_scores() -> None:
    report = _report(
        [
            _result("q_001", question_type="simple", faithfulness=1.0),
            _result("q_002", question_type="multi_doc", faithfulness=None),
        ]
    )

    aggregate = report["aggregate"]["metrics"]

    assert aggregate["faithfulness"] == {"mean": 1.0, "scored": 1}
    assert aggregate["context_recall"] == {"mean": 0.5, "scored": 2}


def test_breakdown_groups_by_metadata_value() -> None:
    report = _report(
        [
            _result("q_001", question_type="simple", faithfulness=1.0),
            _result("q_002", question_type="simple", faithfulness=0.5),
            _result("q_003", question_type="multi_doc", faithfulness=0.0),
        ]
    )

    breakdown = report["breakdown"]["question_type"]

    assert breakdown["simple"]["metrics"]["faithfulness"]["mean"] == 0.75
    assert breakdown["multi_doc"]["metrics"]["faithfulness"]["mean"] == 0.0


def test_markdown_and_csv_contain_the_metric_table(tmp_path: Path) -> None:
    report = _report([_result("q_001", question_type="simple", faithfulness=1.0)])

    markdown = render_markdown(report)
    assert "| faithfulness | 1.0000 | 1/1 |" in markdown

    config = EvaluationConfig(
        dataset_path=Path("tests/fixtures/example.jsonl"),
        output_dir=tmp_path,
        run_name="run1",
    )
    paths = write_report(report, config)

    assert paths["json"].exists() and paths["markdown"].exists() and paths["csv"].exists()
    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert rows[0]["id"] == "q_001"
    assert rows[0]["faithfulness"] == "1.0"


def test_metric_failed_on_every_sample_is_still_reported(tmp_path: Path) -> None:
    result = _result("q_001", question_type="simple", faithfulness=None)
    result.scores = {"context_recall": 0.5}
    report = _report([result])

    assert report["aggregate"]["metrics"]["faithfulness"] == {"mean": None, "scored": 0}
    assert "faithfulness" in render_markdown(report)

    config = EvaluationConfig(
        dataset_path=Path("tests/fixtures/example.jsonl"),
        output_dir=tmp_path,
        run_name="run2",
    )
    header = write_report(report, config)["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "faithfulness" in header
