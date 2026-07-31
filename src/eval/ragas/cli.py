from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from datetime import UTC, datetime
from pathlib import Path

from src.eval.ragas.evaluator import RagasEvaluator
from src.eval.ragas.models import RagasEvalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS RAG evaluation on the PASsistant pipeline"
    )
    parser.add_argument("--dataset", required=True, help="Path to RAGAS eval dataset (JSONL)")
    parser.add_argument("--output", help="Path to write JSON report")
    parser.add_argument("--k-eval", type=int, default=5, help="Retrieval depth (default: 5)")
    parser.add_argument(
        "--mode",
        choices=["live", "fixture"],
        default="live",
        help="Evaluation mode (default: live)",
    )
    parser.add_argument("--fixture", help="Path to pre-computed responses JSON (fixture mode)")
    parser.add_argument(
        "--metrics-tier",
        choices=["core", "extended"],
        default="core",
        help="Which metric tier to evaluate (default: core)",
    )
    parser.add_argument(
        "--evaluation-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Evaluation date / report prefix in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--openrouter-min-interval-seconds",
        type=float,
        default=0.3,
        help="Minimum delay between OpenRouter requests.",
    )
    parser.add_argument(
        "--openrouter-jitter-seconds",
        type=float,
        default=0.15,
        help="Random jitter added to OpenRouter request pacing.",
    )
    parser.add_argument(
        "--openrouter-max-retries",
        type=int,
        default=6,
        help="Maximum retries for transient OpenRouter / Cloudflare failures.",
    )
    parser.add_argument(
        "--openrouter-backoff-base-seconds",
        type=float,
        default=2.0,
        help="Base exponential backoff in seconds for transient OpenRouter failures.",
    )
    return parser.parse_args()


def _parse_args() -> argparse.Namespace:
    return parse_args()


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

    args = parse_args()
    config = RagasEvalConfig(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output) if args.output else None,
        metrics_tier=args.metrics_tier,
        mode=args.mode,
        fixture_path=Path(args.fixture) if args.fixture else None,
        k_eval=args.k_eval,
        evaluation_date=args.evaluation_date,
        openrouter_min_interval_seconds=args.openrouter_min_interval_seconds,
        openrouter_jitter_seconds=args.openrouter_jitter_seconds,
        openrouter_max_retries=args.openrouter_max_retries,
        openrouter_backoff_base_seconds=args.openrouter_backoff_base_seconds,
    )

    evaluator = RagasEvaluator(config)
    report = asyncio.run(evaluator.run())

    print("\n=== RAGAS Evaluation Summary ===")
    print(f"Samples evaluated: {report['sample_count']}")
    print(f"Metrics tier:      {report['metrics_tier']}")
    print("Aggregate scores:")
    for metric, score in report.get("aggregate_scores", {}).items():
        print(f"  {metric:40s}  {score:.4f}")

    if config.output_path:
        print(f"\nFull report: {config.output_path}")
