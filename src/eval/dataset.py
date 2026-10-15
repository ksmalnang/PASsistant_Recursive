"""Read, validate, and write evaluation testsets in JSONL form."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.eval.models import DatasetSample

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "user_input", "reference")
QUESTION_TYPES = {"simple", "multi_doc", "conditional"}
SCALAR_METADATA_FIELDS = (
    "topic",
    "question_type",
    "answerable",
    "author",
    "dataset_version",
    "source_doc_version",
)


class DatasetError(ValueError):
    """Raised when a testset file holds one or more invalid rows."""

    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        super().__init__(f"{path} has {len(errors)} invalid row(s):\n{preview}")


def load_dataset(path: Path) -> list[DatasetSample]:
    """Load every row of a testset, raising once if any row is invalid."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[DatasetSample] = []
    errors: list[str] = []
    for line_no, row, error in _iter_rows(path):
        if error is not None:
            errors.append(f"line {line_no}: {error}")
            continue
        try:
            samples.append(_parse_sample(row, line_no))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise DatasetError(path, errors)
    if not samples:
        raise DatasetError(path, ["dataset has no samples"])
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def write_dataset(path: Path, samples: list[DatasetSample]) -> None:
    """Write samples as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_sample_to_row(sample), ensure_ascii=False) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


def dataset_stats(samples: list[DatasetSample]) -> dict[str, Any]:
    """Summarise dataset shape and metadata coverage."""
    counts: dict[str, Counter[str]] = {field: Counter() for field in SCALAR_METADATA_FIELDS}
    for sample in samples:
        for field in SCALAR_METADATA_FIELDS:
            value = sample.metadata.get(field)
            if value is not None:
                counts[field][str(value)] += 1
    return {
        "samples": len(samples),
        "with_reference_contexts": sum(1 for sample in samples if sample.reference_contexts),
        "reference_contexts": sum(len(sample.reference_contexts) for sample in samples),
        **{field: dict(counts[field].most_common()) for field in SCALAR_METADATA_FIELDS},
    }


def _iter_rows(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield `(line_number, row, error)` for each non-empty line."""
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield line_no, json.loads(line), None
        except json.JSONDecodeError as exc:
            yield line_no, None, f"invalid JSON ({exc.msg})"


def _parse_sample(row: Any, line_no: int) -> DatasetSample:
    prefix = f"line {line_no}"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}: row must be a JSON object")

    for field in REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(f"{prefix}: missing required field '{field}'")

    return DatasetSample(
        id=_required_text(row["id"], f"{prefix}: id"),
        user_input=_required_text(row["user_input"], f"{prefix}: user_input"),
        reference=_required_text(row["reference"], f"{prefix}: reference"),
        reference_contexts=_parse_reference_contexts(row.get("reference_contexts"), prefix),
        metadata=_parse_metadata(row.get("metadata"), prefix),
    )


def _parse_reference_contexts(value: Any, prefix: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{prefix}: reference_contexts must be a list")
    contexts: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{prefix}: reference_contexts[{index}] must be a non-empty string")
        contexts.append(item.strip())
    return tuple(contexts)


def _parse_metadata(value: Any, prefix: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: metadata must be an object")

    question_type = value.get("question_type")
    if question_type is not None and question_type not in QUESTION_TYPES:
        raise ValueError(
            f"{prefix}: metadata.question_type must be one of {', '.join(sorted(QUESTION_TYPES))}"
        )

    answerable = value.get("answerable")
    if answerable is not None and not isinstance(answerable, bool):
        raise ValueError(f"{prefix}: metadata.answerable must be a boolean")

    source_doc_ids = value.get("source_doc_ids")
    if source_doc_ids is not None and (
        not isinstance(source_doc_ids, list)
        or not all(isinstance(item, str) for item in source_doc_ids)
    ):
        raise ValueError(f"{prefix}: metadata.source_doc_ids must be a list of strings")

    for field in ("topic", "author", "dataset_version", "source_doc_version"):
        raw = value.get(field)
        if raw is not None and not isinstance(raw, str):
            raise ValueError(f"{prefix}: metadata.{field} must be a string")

    return dict(value)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sample_to_row(sample: DatasetSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "user_input": sample.user_input,
        "reference": sample.reference,
        "reference_contexts": list(sample.reference_contexts),
        "metadata": sample.metadata,
    }
