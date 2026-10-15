"""Data types shared by dataset generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

Mode = Literal["fixture", "live"]


@dataclass(frozen=True)
class DatasetSample:
    """One evaluation sample in the testset schema.

    The testset carries only ground truth: the candidate answer and retrieved
    contexts are produced by the pipeline (live) or by a responses file (fixture).
    """

    id: str
    user_input: str
    reference: str
    reference_contexts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Pipeline output and RAGAS scores for one sample."""

    sample: DatasetSample
    response: str
    retrieved_contexts: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenRouterSettings:
    """Transport settings for the OpenRouter client."""

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    min_interval_seconds: float = 0.3
    jitter_seconds: float = 0.15
    max_retries: int = 6
    backoff_base_seconds: float = 2.0


@dataclass
class EvaluationConfig:
    """User-selected inputs for one evaluation run."""

    dataset_path: Path
    mode: Mode = "fixture"
    fixture_path: Path | None = None
    output_dir: Path = Path("src/eval/reports")
    run_name: str | None = None
    openrouter: OpenRouterSettings | None = None


@dataclass(frozen=True)
class RunContext:
    """Runtime facts recorded alongside the scores for reproducibility."""

    strategy: str
    top_k: int
    judge_model: str
    embedding_model: str
    pipeline_model: str
    started_at: datetime
    duration_seconds: float
