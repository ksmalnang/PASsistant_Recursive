#!/usr/bin/env python3
"""Run a RAGAS evaluation and write the report with a timestamp+dataset-name prefix.

Output is automatically placed in ``src/eval/reports/`` using the naming
convention ``YYYYMMDD_HHMMSS_<dataset-stem>_ragas_report.json``.

Usage examples
--------------
# Fixture mode (uses stored answers — fast, no live pipeline):
    python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl

# Live mode (calls the actual RAG pipeline):
    python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live

# Extended metrics, custom retrieval depth:
    python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl \\
        --metrics-tier extended --k-eval 10

# Override the output directory:
    python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl \\
        --output-dir my_reports
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.ragas.evaluator import RagasEvaluator
from src.eval.ragas.models import RagasEvalConfig


def _build_output_path(dataset: Path, output_dir: Path) -> Path:
    """Return ``<output_dir>/YYYYMMDD_HHMMSS_<dataset-stem>_ragas_report.json``."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{dataset.stem}_ragas_report.json"
    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS eval and save report as timestamp+dataset-name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to RAGAS eval dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/eval/reports"),
        help="Directory to write the JSON report into",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "fixture"],
        default="fixture",
        help="'fixture' scores stored answers; 'live' calls the RAG pipeline",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to pre-computed responses JSON (fixture mode only)",
    )
    parser.add_argument(
        "--metrics-tier",
        choices=["core", "extended"],
        default="core",
        help="'core' evaluates 4 metrics; 'extended' evaluates 8 metrics",
    )
    parser.add_argument(
        "--k-eval",
        type=int,
        default=10,
        help="Retrieval depth — top-k documents fetched per question (live mode only)",
    )
    parser.add_argument(
        "--openrouter-min-interval-seconds",
        type=float,
        default=0.3,
        help="Minimum delay between consecutive OpenRouter requests",
    )
    parser.add_argument(
        "--openrouter-jitter-seconds",
        type=float,
        default=0.15,
        help="Random jitter added on top of the minimum interval",
    )
    parser.add_argument(
        "--openrouter-max-retries",
        type=int,
        default=6,
        help="Maximum retries for transient OpenRouter / Cloudflare failures",
    )
    parser.add_argument(
        "--openrouter-backoff-base-seconds",
        type=float,
        default=2.0,
        help="Base for exponential back-off between retries",
    )
    return parser.parse_args()


def _force_exit(sig: int, frame: object) -> None:  # noqa: ARG001
    print("\n[Interrupted] Forcing exit — killing all threads.", flush=True)
    os._exit(1)


def main() -> None:
    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    dataset: Path = args.dataset
    if not dataset.exists():
        logger.error("Dataset not found: %s", dataset)
        sys.exit(1)

    output_path = _build_output_path(dataset, args.output_dir)

    logger.info("Dataset    : %s", dataset)
    logger.info("Mode       : %s", args.mode)
    logger.info("Metrics    : %s", args.metrics_tier)
    logger.info("Report     : %s", output_path)

    config = RagasEvalConfig(
        dataset_path=dataset,
        output_path=output_path,
        metrics_tier=args.metrics_tier,
        mode=args.mode,
        fixture_path=args.fixture,
        k_eval=args.k_eval,
        evaluation_date=datetime.now(UTC).date().isoformat(),
        openrouter_min_interval_seconds=args.openrouter_min_interval_seconds,
        openrouter_jitter_seconds=args.openrouter_jitter_seconds,
        openrouter_max_retries=args.openrouter_max_retries,
        openrouter_backoff_base_seconds=args.openrouter_backoff_base_seconds,
    )

    evaluator = RagasEvaluator(config)
    report = asyncio.run(evaluator.run())

    print("\n=== RAGAS Evaluation Summary ===")
    print(f"Dataset    : {dataset.name}")
    print(f"Samples    : {report['sample_count']}")
    print(f"Metrics    : {report['metrics_tier']}")
    print("Aggregate scores:")
    for metric, score in report.get("aggregate_scores", {}).items():
        print(f"  {metric:40s}  {score:.4f}")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
