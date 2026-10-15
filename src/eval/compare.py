"""Compare evaluation reports across retrieval strategies."""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_STRATEGIES = ("similarity", "rrf", "reranker")


def load_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load report JSON files, rejecting documents that are not evaluation reports."""
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "run" not in payload or "aggregate" not in payload:
            raise ValueError(f"{path} is not a RAGAS evaluation report")
        reports.append(payload)
    return reports


def build_comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Group reports by dataset and strategy, then rank strategies per metric."""
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        run = report["run"]
        dataset = run["dataset"]
        entry = datasets.setdefault(
            (dataset["name"], dataset["path"]),
            {
                "dataset": dataset["name"],
                "path": dataset["path"],
                "sample_count": dataset["sample_count"],
                "strategies": {},
            },
        )
        strategy = run["retrieval"]["strategy"]
        previous = entry["strategies"].get(strategy)
        if previous is not None and run["started_at"] <= previous["started_at"]:
            continue
        entry["sample_count"] = dataset["sample_count"]
        entry["strategies"][strategy] = {
            "run": run["name"],
            "started_at": run["started_at"],
            "metrics": report["aggregate"]["metrics"],
            "settings": {"top_k": run["retrieval"]["top_k"], "judge": run["models"]["judge"]},
        }

    for entry in datasets.values():
        metrics = sorted(
            {metric for payload in entry["strategies"].values() for metric in payload["metrics"]}
        )
        entry["metrics"] = metrics
        entry["best"] = {metric: _best(entry["strategies"], metric) for metric in metrics}
        entry["delta_vs_baseline"] = _deltas(entry)
        entry["warnings"] = _setting_warnings(entry["strategies"])

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": list(datasets.values()),
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render the comparison as Markdown tables."""
    lines = ["# Retrieval strategy comparison", ""]
    for entry in comparison["datasets"]:
        strategies = list(entry["strategies"])
        lines += [
            f"## {entry['dataset']} ({entry['sample_count']} samples)",
            "",
            _table(
                ["metric", *strategies, "best"],
                [
                    [
                        metric,
                        *[
                            _number(
                                entry["strategies"][name]["metrics"].get(metric, {}).get("mean")
                            )
                            for name in strategies
                        ],
                        str(entry["best"].get(metric) or "-"),
                    ]
                    for metric in entry["metrics"]
                ],
            ),
        ]
        for warning in entry.get("warnings", []):
            lines.append(f"> Warning: {warning}")
        delta = entry.get("delta_vs_baseline") or {}
        if delta.get("strategies"):
            lines += [
                "",
                f"### Delta vs {delta['baseline']}",
                "",
                _table(
                    ["metric", *delta["strategies"]],
                    [
                        [
                            metric,
                            *[
                                _signed(values.get(metric))
                                for values in delta["strategies"].values()
                            ],
                        ]
                        for metric in entry["metrics"]
                    ],
                ),
            ]
        lines.append("")
    return "\n".join(lines)


def write_comparison(
    comparison: dict[str, Any],
    output_dir: Path,
    name: str | None = None,
) -> dict[str, Path]:
    """Write the comparison as JSON, tidy CSV, and Markdown."""
    stem = name or f"comparison_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
        "markdown": output_dir / f"{stem}.md",
    }
    paths["json"].write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(render_comparison_markdown(comparison), encoding="utf-8")
    _write_csv(comparison, paths["csv"])
    logger.info("Wrote comparison artifacts: %s", ", ".join(str(path) for path in paths.values()))
    return paths


def repo_root() -> Path:
    """Return the repository root that contains the eval package."""
    return Path(__file__).resolve().parents[2]


def sweep_strategies(
    *,
    dataset: Path,
    strategies: Sequence[str],
    output_dir: Path,
    extra_args: Sequence[str] = (),
) -> list[Path]:
    """Run a live evaluation per strategy in its own process and return report paths.

    Each strategy is a fresh process because the retrieval settings are cached at
    application start; changing them inside one process would silently reuse the
    first strategy.
    """
    root = repo_root()
    dataset = dataset.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir

    report_paths: list[Path] = []
    for strategy in strategies:
        run_name = f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{dataset.stem}_{strategy}"
        command = [
            sys.executable,
            "-m",
            "src.eval",
            "evaluate",
            "--dataset",
            str(dataset),
            "--mode",
            "live",
            "--output-dir",
            str(output_dir),
            "--run-name",
            run_name,
            *extra_args,
        ]
        logger.info("Sweeping strategy=%s -> %s", strategy, run_name)
        subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "RETRIEVAL_STRATEGY": strategy},
            check=True,
        )
        report_paths.append(output_dir / f"{run_name}.json")
    return report_paths


def _best(strategies: dict[str, Any], metric: str) -> str | None:
    scored = [
        (name, payload["metrics"][metric]["mean"], payload["metrics"][metric]["scored"])
        for name, payload in strategies.items()
        if payload["metrics"].get(metric, {}).get("mean") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[1], item[2]))[0]


def _deltas(entry: dict[str, Any]) -> dict[str, Any]:
    strategies = entry["strategies"]
    baseline = next(
        (name for name in DEFAULT_STRATEGIES if name in strategies),
        next(iter(sorted(strategies)), None),
    )
    if baseline is None:
        return {}
    baseline_metrics = strategies[baseline]["metrics"]

    deltas: dict[str, Any] = {"baseline": baseline, "strategies": {}}
    for name, payload in strategies.items():
        if name == baseline:
            continue
        deltas["strategies"][name] = {
            metric: round(payload["metrics"][metric]["mean"] - baseline_metrics[metric]["mean"], 4)
            for metric in entry["metrics"]
            if payload["metrics"].get(metric, {}).get("mean") is not None
            and baseline_metrics.get(metric, {}).get("mean") is not None
        }
    return deltas


def _setting_warnings(strategies: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("top_k", "judge"):
        values = {name: payload["settings"][field] for name, payload in strategies.items()}
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{name}={value}" for name, value in sorted(values.items()))
            warnings.append(f"strategies differ on {field}: {detail}")
    return warnings


def _write_csv(comparison: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "sample_count", "strategy", "metric", "mean", "scored", "best"])
        for entry in comparison["datasets"]:
            for strategy, payload in entry["strategies"].items():
                for metric in entry["metrics"]:
                    cell = payload["metrics"].get(metric) or {}
                    writer.writerow(
                        [
                            entry["dataset"],
                            entry["sample_count"],
                            strategy,
                            metric,
                            cell.get("mean", ""),
                            cell.get("scored", ""),
                            entry["best"].get(metric) or "",
                        ]
                    )


def _number(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def _signed(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "-"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
