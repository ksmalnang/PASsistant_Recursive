"""Command-line interface for dataset generation, evaluation, and comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.eval.compare import (
    DEFAULT_STRATEGIES,
    build_comparison,
    load_reports,
    render_comparison_markdown,
    repo_root,
    sweep_strategies,
    write_comparison,
)
from src.eval.dataset import DatasetError, dataset_stats, load_dataset
from src.eval.evaluate import run_evaluation
from src.eval.generate import build_dataset, default_responses_path
from src.eval.models import EvaluationConfig, OpenRouterSettings

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("src/eval/reports")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `python -m src.eval` and the `ragas-eval` console script."""
    _configure_logging()
    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)
    args = parse_args(argv)
    args.handler(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.eval",
        description="Generate RAGAS datasets, evaluate the pipeline, and compare strategies",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Score a dataset with RAGAS")
    _add_evaluate_args(evaluate)
    evaluate.set_defaults(handler=_run_evaluate)

    build = subparsers.add_parser(
        "build-dataset", help="Generate a dataset from indexed documents with RAGAS"
    )
    _add_build_args(build)
    build.set_defaults(handler=_run_build_dataset)

    validate = subparsers.add_parser("validate", help="Validate a dataset and print its stats")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.set_defaults(handler=_run_validate)

    compare = subparsers.add_parser("compare", help="Compare evaluation reports")
    compare.add_argument("reports", type=Path, nargs="+")
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compare.add_argument("--name", default=None, help="Comparison file stem")
    compare.set_defaults(handler=_run_compare)

    sweep = subparsers.add_parser(
        "sweep", help="Run live evaluation for every retrieval strategy, then compare"
    )
    _add_sweep_args(sweep)
    sweep.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    sweep.add_argument("--name", default=None, help="Comparison file stem")
    sweep.set_defaults(handler=_run_sweep)

    return parser.parse_args(argv)


def _add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)


def _add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    _add_shared_run_args(parser)
    parser.add_argument(
        "--mode",
        choices=("fixture", "live"),
        default="fixture",
        help="'fixture' scores a --fixture responses file; 'live' runs the pipeline (default: fixture)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Responses JSON {id: {response, retrieved_contexts}} for fixture mode",
    )
    parser.add_argument("--run-name", default=None, help="Report file stem")
    parser.add_argument(
        "--strategy",
        choices=DEFAULT_STRATEGIES,
        default=None,
        help="Override RETRIEVAL_STRATEGY for this run",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override RETRIEVAL_TOP_K")
    _add_openrouter_args(parser)


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    _add_shared_run_args(parser)
    _add_openrouter_args(parser)


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="Dataset JSONL path to write")
    parser.add_argument("--testset-size", type=int, default=20, help="Number of Q&A pairs")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Chunks to sample; default scales with --testset-size (4x, 12-200)",
    )
    parser.add_argument(
        "--per-document-limit", type=int, default=30, help="Max chunks per source document"
    )
    parser.add_argument("--collection", default=None, help="Qdrant collection override")
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Skip capturing pipeline responses to the sidecar file",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help="Sidecar path for captured responses (default: <output>.responses.json)",
    )
    parser.add_argument(
        "--dataset-version", default="v1", help="Value written to metadata.dataset_version"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    _add_openrouter_args(parser)


def _add_openrouter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-interval", type=float, default=None, dest="min_interval")
    parser.add_argument("--jitter", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--backoff", type=float, default=None)


def _run_evaluate(args: argparse.Namespace) -> None:
    if args.mode == "live" and args.fixture:
        raise SystemExit("--fixture only applies to --mode fixture")
    _apply_env_overrides(args)
    config = EvaluationConfig(
        dataset_path=args.dataset,
        mode=args.mode,
        fixture_path=args.fixture,
        output_dir=args.output_dir,
        run_name=args.run_name,
        openrouter=_openrouter_settings(args),
    )
    report = asyncio.run(run_evaluation(config))
    _print_report_summary(report)


def _run_build_dataset(args: argparse.Namespace) -> None:
    samples = build_dataset(
        output_path=args.output,
        testset_size=args.testset_size,
        sample_size=args.sample_size,
        per_document_limit=args.per_document_limit,
        collection=args.collection,
        capture_pipeline=not args.no_pipeline,
        responses_path=args.responses,
        dataset_version=args.dataset_version,
        seed=args.seed,
        openrouter=_openrouter_settings(args),
    )
    print(f"Wrote {len(samples)} samples to {args.output}")
    if not args.no_pipeline:
        print(f"Pipeline responses: {args.responses or default_responses_path(args.output)}")


def _run_validate(args: argparse.Namespace) -> None:
    try:
        samples = load_dataset(args.dataset)
    except DatasetError as exc:
        for error in exc.errors:
            print(error, file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(dataset_stats(samples), indent=2))


def _run_compare(args: argparse.Namespace) -> None:
    comparison = build_comparison(load_reports(args.reports))
    paths = write_comparison(comparison, args.output_dir, args.name)
    print(render_comparison_markdown(comparison))
    print("Comparison artifacts:")
    for kind, path in paths.items():
        print(f"  {kind:9s} {path}")


def _run_sweep(args: argparse.Namespace) -> None:
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root() / args.output_dir
    report_paths = sweep_strategies(
        dataset=args.dataset,
        strategies=args.strategies,
        output_dir=output_dir,
        extra_args=_passthrough_args(args),
    )
    comparison = build_comparison(load_reports(report_paths))
    paths = write_comparison(comparison, output_dir, args.name)
    print(render_comparison_markdown(comparison))
    print("Comparison artifacts:")
    for kind, path in paths.items():
        print(f"  {kind:9s} {path}")


def _apply_env_overrides(args: argparse.Namespace) -> None:
    """Apply retrieval overrides before any settings object is created."""
    strategy = getattr(args, "strategy", None)
    if strategy:
        os.environ["RETRIEVAL_STRATEGY"] = strategy
    top_k = getattr(args, "top_k", None)
    if top_k:
        os.environ["RETRIEVAL_TOP_K"] = str(top_k)


def _openrouter_settings(args: argparse.Namespace) -> OpenRouterSettings:
    from src.config import get_settings

    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise SystemExit("OPENAI_API_KEY or OPENROUTER_API_KEY must be set")

    base = OpenRouterSettings(api_key=api_key, base_url=settings.OPENAI_BASE_URL)
    overrides = {
        "min_interval_seconds": getattr(args, "min_interval", None),
        "jitter_seconds": getattr(args, "jitter", None),
        "max_retries": getattr(args, "max_retries", None),
        "backoff_base_seconds": getattr(args, "backoff", None),
    }
    return replace(base, **{key: value for key, value in overrides.items() if value is not None})


def _passthrough_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    for flag, attribute in (
        ("--min-interval", "min_interval"),
        ("--jitter", "jitter"),
        ("--max-retries", "max_retries"),
        ("--backoff", "backoff"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            extra += [flag, str(value)]
    return extra


def _print_report_summary(report: dict[str, Any]) -> None:
    run = report["run"]
    print(f"\n=== {run['name']} ===")
    print(
        f"dataset={run['dataset']['name']} samples={run['dataset']['sample_count']} "
        f"mode={run['mode']} strategy={run['retrieval']['strategy']}"
    )
    for metric, entry in report["aggregate"]["metrics"].items():
        mean = f"{entry['mean']:.4f}" if entry["mean"] is not None else "n/a"
        print(f"  {metric:24s} {mean}  ({entry['scored']}/{run['dataset']['sample_count']} scored)")
    usage = report["usage"]
    print(
        f"usage: {usage['request_count']} requests, {usage['total_tokens']} tokens, "
        f"${usage['cost_usd']:.4f}"
    )
    print("Artifacts:")
    for kind, path in report["artifacts"].items():
        print(f"  {kind:9s} {path}")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )


def _force_exit(signum: int, frame: Any) -> None:  # noqa: ARG001
    print("\n[Interrupted] Exiting.", flush=True)
    os._exit(130)
