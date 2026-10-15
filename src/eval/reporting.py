"""Turn evaluation results into JSON, per-sample CSV, and a Markdown summary."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.eval.models import EvaluationConfig, RunContext, SampleResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BREAKDOWN_FIELDS = ("question_type", "topic", "answerable", "dataset_version", "source_doc_version")
CSV_METADATA_COLUMNS = ("question_type", "topic", "answerable", "dataset_version")


def build_report(
    config: EvaluationConfig,
    results: list[SampleResult],
    usage: dict[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    """Assemble the full report structure written to disk."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": run_name(config, context),
            "started_at": context.started_at.isoformat(),
            "duration_seconds": round(context.duration_seconds, 1),
            "mode": config.mode,
            "dataset": {
                "name": config.dataset_path.name,
                "path": str(config.dataset_path),
                "sample_count": len(results),
            },
            "retrieval": {"strategy": context.strategy, "top_k": context.top_k},
            "models": {
                "judge": context.judge_model,
                "embedding": context.embedding_model,
                "pipeline": context.pipeline_model,
            },
        },
        "aggregate": aggregate_scores(results),
        "breakdown": metadata_breakdown(results),
        "usage": usage,
        "samples": [_sample_record(result) for result in results],
    }


def write_report(report: dict[str, Any], config: EvaluationConfig) -> dict[str, Path]:
    """Write the report as JSON, CSV, and Markdown; return the three paths."""
    name = report["run"]["name"]
    paths = {
        "json": config.output_dir / f"{name}.json",
        "csv": config.output_dir / f"{name}.csv",
        "markdown": config.output_dir / f"{name}.md",
    }
    report["artifacts"] = {key: str(path) for key, path in paths.items()}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_samples_csv(report, paths["csv"])
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    logger.info("Wrote report artifacts: %s", ", ".join(str(path) for path in paths.values()))
    return paths


def run_name(config: EvaluationConfig, context: RunContext) -> str:
    """Return the report stem, defaulting to a timestamp plus dataset name."""
    if config.run_name:
        return config.run_name
    return f"{context.started_at:%Y%m%d_%H%M%S}_{config.dataset_path.stem}"


def aggregate_scores(results: list[SampleResult]) -> dict[str, Any]:
    """Mean score per metric plus how many samples produced a score.

    Metrics that failed on every sample still appear, with a null mean and
    `scored: 0`, so a metric can never silently vanish from the report.
    """
    metrics = sorted(
        {metric for result in results for metric in (*result.scores, *result.failures)}
    )
    aggregate: dict[str, Any] = {"sample_count": len(results), "metrics": {}}
    for metric in metrics:
        values = [result.scores[metric] for result in results if metric in result.scores]
        aggregate["metrics"][metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "scored": len(values),
        }
    return aggregate


def metadata_breakdown(results: list[SampleResult]) -> dict[str, Any]:
    """Aggregate scores per metadata value so weak slices are visible."""
    breakdown: dict[str, Any] = {}
    for field in BREAKDOWN_FIELDS:
        groups: dict[str, list[SampleResult]] = {}
        for result in results:
            value = result.sample.metadata.get(field)
            if value is not None:
                groups.setdefault(str(value), []).append(result)
        if groups:
            breakdown[field] = {
                value: aggregate_scores(group) for value, group in sorted(groups.items())
            }
    return breakdown


def render_markdown(report: dict[str, Any]) -> str:
    """Render a readable Markdown summary of the report."""
    run = report["run"]
    metrics = list(report["aggregate"]["metrics"])
    lines = [
        f"# RAGAS evaluation — {run['dataset']['name']}",
        "",
        f"- Run: `{run['name']}`",
        f"- Mode: {run['mode']} | Retrieval: {run['retrieval']['strategy']} "
        f"(top_k={run['retrieval']['top_k']})",
        f"- Judge: {run['models']['judge']} | Embeddings: {run['models']['embedding']}",
        f"- Samples: {run['dataset']['sample_count']} | Duration: {run['duration_seconds']}s",
        "",
        "## Aggregate scores",
        "",
        _table(
            ["metric", "mean", "scored"],
            [
                [
                    metric,
                    _number(entry["mean"]),
                    f"{entry['scored']}/{run['dataset']['sample_count']}",
                ]
                for metric, entry in report["aggregate"]["metrics"].items()
            ],
        ),
    ]

    for field, groups in report["breakdown"].items():
        lines += [
            "",
            f"## Breakdown by {field}",
            "",
            _table(
                ["value", "samples", *metrics],
                [
                    [value, str(entry["sample_count"]), *[_metric_cell(entry, m) for m in metrics]]
                    for value, entry in groups.items()
                ],
            ),
        ]

    lines += [
        "",
        "## Per-sample scores",
        "",
        _table(
            ["id", "topic", "type", "contexts", *metrics],
            [
                [
                    sample["id"],
                    str(sample["metadata"].get("topic", "-")),
                    str(sample["metadata"].get("question_type", "-")),
                    str(sample["retrieved_context_count"]),
                    *[_score_cell(sample, metric) for metric in metrics],
                ]
                for sample in report["samples"]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _write_samples_csv(report: dict[str, Any], path: Path) -> None:
    metrics = list(report["aggregate"]["metrics"])
    fieldnames = [
        "id",
        "user_input",
        *CSV_METADATA_COLUMNS,
        "retrieved_context_count",
        "response_chars",
        *metrics,
        "failed_metrics",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in report["samples"]:
            metadata = sample["metadata"]
            row: dict[str, Any] = {
                "id": sample["id"],
                "user_input": sample["user_input"],
                "retrieved_context_count": sample["retrieved_context_count"],
                "response_chars": len(sample["response"]),
                "failed_metrics": ";".join(sample["failures"]),
            }
            for column in CSV_METADATA_COLUMNS:
                row[column] = metadata.get(column, "")
            for metric in metrics:
                row[metric] = sample["scores"].get(metric, "")
            writer.writerow(row)


def _sample_record(result: SampleResult) -> dict[str, Any]:
    return {
        "id": result.sample.id,
        "user_input": result.sample.user_input,
        "reference": result.sample.reference,
        "response": result.response,
        "retrieved_context_count": len(result.retrieved_contexts),
        "scores": result.scores,
        "failures": result.failures,
        "metadata": result.sample.metadata,
    }


def _metric_cell(entry: dict[str, Any], metric: str) -> str:
    score = entry["metrics"].get(metric) or {}
    return _number(score.get("mean"))


def _score_cell(sample: dict[str, Any], metric: str) -> str:
    return _number(sample["scores"].get(metric))


def _number(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
