"""Generate a testset from indexed documents with RAGAS and OpenRouter."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.eval.dataset import write_dataset
from src.eval.models import DatasetSample, OpenRouterSettings
from src.eval.openrouter import (
    OpenRouterClient,
    OpenRouterEmbeddings,
    OpenRouterRagasLLM,
    Usage,
    ragas_extra_body,
)
from src.eval.pipeline import LivePipeline
from src.eval.progress import Heartbeat

logger = logging.getLogger(__name__)

_SCROLL_PAGE_SIZE = 256
_MAX_SCROLLED_POINTS = 5000
DEFAULT_DATASET_VERSION = "v1"
GENERATOR_AUTHOR = "ragas"
# RAGAS runs several LLM calls per source chunk, so the chunk sample is what drives
# generation time. When --sample-size is omitted, scale it with the requested testset.
AUTO_SAMPLE_CHUNKS_PER_QUESTION = 4
AUTO_SAMPLE_MIN_CHUNKS = 12
AUTO_SAMPLE_MAX_CHUNKS = 200


def default_responses_path(output_path: Path) -> Path:
    """Return the sidecar path that stores captured pipeline outputs."""
    return output_path.with_suffix(".responses.json")


def build_dataset(
    *,
    output_path: Path,
    testset_size: int = 20,
    sample_size: int | None = None,
    per_document_limit: int = 30,
    collection: str | None = None,
    capture_pipeline: bool = True,
    responses_path: Path | None = None,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    seed: int = 42,
    openrouter: OpenRouterSettings | None = None,
) -> list[DatasetSample]:
    """Generate a testset, optionally capturing pipeline outputs to a sidecar file."""
    from qdrant_client import QdrantClient

    from src.config import get_settings
    from src.eval.evaluate import default_openrouter_settings

    settings = get_settings()
    qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    collection_name = collection or settings.QDRANT_COLLECTION_NAME

    chunks = sample_chunks(
        qdrant,
        collection_name,
        sample_size=_resolve_sample_size(sample_size, testset_size),
        per_document_limit=per_document_limit,
        seed=seed,
    )
    if not chunks:
        raise ValueError(f"No indexed chunks found in collection '{collection_name}'")
    _log_workload(len(chunks), testset_size)

    usage = Usage()
    client = OpenRouterClient(openrouter or default_openrouter_settings(), usage)
    samples = _generate_samples(chunks, client, settings, testset_size, dataset_version)

    if capture_pipeline:
        samples, responses = asyncio.run(_capture_pipeline(samples, usage))
        path = responses_path or default_responses_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote pipeline responses for %d samples to %s", len(responses), path)

    write_dataset(output_path, samples)
    return samples


def _resolve_sample_size(sample_size: int | None, testset_size: int) -> int:
    """Scale the chunk sample with the requested testset so small testsets stay cheap."""
    if sample_size is not None:
        return sample_size
    return min(
        AUTO_SAMPLE_MAX_CHUNKS,
        max(AUTO_SAMPLE_CHUNKS_PER_QUESTION * testset_size, AUTO_SAMPLE_MIN_CHUNKS),
    )


def _log_workload(chunk_count: int, testset_size: int) -> None:
    """RAGAS makes several LLM calls per chunk, so cost tracks chunks, not testset size."""
    logger.info(
        "Sampling %d chunks for %d testset samples; RAGAS runs roughly 3-5 LLM calls per chunk "
        "during analysis, so generation time scales with --sample-size, not --testset-size",
        chunk_count,
        testset_size,
    )
    if chunk_count > AUTO_SAMPLE_CHUNKS_PER_QUESTION * testset_size:
        logger.warning(
            "%d chunks for %d testset samples is a lot of analysis work; use a smaller "
            "--sample-size (e.g. 5-10) for a small testset",
            chunk_count,
            testset_size,
        )


def sample_chunks(
    qdrant: Any,
    collection: str,
    *,
    sample_size: int,
    per_document_limit: int,
    seed: int,
) -> list[Any]:
    """Sample chunks spread across documents so one PDF cannot dominate."""
    points = _scroll_all(qdrant, collection)
    by_document: dict[str, list[Any]] = defaultdict(list)
    for point in points:
        by_document[str(point.payload.get("document_id", "unknown"))].append(point)

    rng = random.Random(seed)
    document_ids = list(by_document)
    rng.shuffle(document_ids)

    sampled: list[Any] = []
    for document_id in document_ids:
        chunks = list(by_document[document_id])
        rng.shuffle(chunks)
        sampled.extend(chunks[:per_document_limit])
        if len(sampled) >= sample_size:
            break

    sampled = sampled[:sample_size]
    logger.info(
        "Sampled %d chunks from %d of %d documents in %s",
        len(sampled),
        len({point.payload.get("document_id") for point in sampled}),
        len(document_ids),
        collection,
    )
    return sampled


def _scroll_all(qdrant: Any, collection: str) -> list[Any]:
    points: list[Any] = []
    offset: Any = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None or len(points) >= _MAX_SCROLLED_POINTS:
            return points


def _generate_samples(
    chunks: list[Any],
    client: OpenRouterClient,
    settings: Any,
    testset_size: int,
    dataset_version: str,
) -> list[DatasetSample]:
    from langchain_core.documents import Document
    from ragas.testset import TestsetGenerator

    documents = [
        Document(
            page_content=str(point.payload.get("text") or ""),
            metadata={
                "document_id": point.payload.get("document_id"),
                "filename": point.payload.get("filename"),
            },
        )
        for point in chunks
    ]
    documents = [document for document in documents if document.page_content.strip()]

    generator = TestsetGenerator(
        llm=OpenRouterRagasLLM(
            client,
            settings.RAGAS_LLM_MODEL or settings.LLM_MODEL,
            extra_body=ragas_extra_body("llm"),
        ),
        embedding_model=OpenRouterEmbeddings(
            client,
            settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL,
            extra_body=ragas_extra_body("embedding"),
        ),
    )
    logger.info(
        "Generating %d testset samples from %d chunks; RAGAS prints its own progress bars",
        testset_size,
        len(documents),
    )
    with Heartbeat(f"RAGAS testset generation ({testset_size} samples)"):
        testset = generator.generate_with_langchain_docs(documents, testset_size=testset_size)

    doc_ids_by_text = {
        _normalize(str(point.payload.get("text") or "")): str(point.payload.get("document_id"))
        for point in chunks
        if point.payload.get("document_id")
    }
    samples = [
        sample
        for index, item in enumerate(testset.to_list(), start=1)
        if (sample := _to_sample(item, index, doc_ids_by_text, dataset_version)) is not None
    ]
    logger.info("Generated %d usable samples", len(samples))
    return samples


async def _capture_pipeline(
    samples: list[DatasetSample],
    usage: Usage,
) -> tuple[list[DatasetSample], dict[str, dict[str, Any]]]:
    """Run the live pipeline and return the samples plus a responses sidecar."""
    results = await LivePipeline(usage=usage).run(samples)
    captured: list[DatasetSample] = []
    responses: dict[str, dict[str, Any]] = {}
    for sample, result in zip(samples, results, strict=True):
        metadata = dict(sample.metadata)
        if result.failures.get("pipeline"):
            metadata["pipeline_failure"] = result.failures["pipeline"]
        elif not result.retrieved_contexts:
            metadata["pipeline_failure"] = "retrieval returned no contexts"
        captured.append(replace(sample, metadata=metadata))
        responses[sample.id] = {
            "response": result.response,
            "retrieved_contexts": result.retrieved_contexts,
        }
    return captured, responses


def _to_sample(
    item: dict[str, Any],
    index: int,
    doc_ids_by_text: dict[str, str],
    dataset_version: str,
) -> DatasetSample | None:
    raw_question = item.get("user_input")
    raw_answer = item.get("reference") or item.get("response")
    if not isinstance(raw_question, str) or not isinstance(raw_answer, str):
        return None
    question = raw_question.strip()
    reference = raw_answer.strip()
    if not question or not reference:
        return None

    reference_contexts = tuple(str(text) for text in (item.get("reference_contexts") or []))
    doc_ids = sorted(
        {
            doc_ids_by_text[normalized]
            for text in reference_contexts
            if (normalized := _normalize(text)) in doc_ids_by_text
        }
    )
    metadata: dict[str, Any] = {
        "question_type": _question_type(item.get("synthesizer_name")),
        "answerable": True,
        "source_doc_ids": doc_ids,
        "author": GENERATOR_AUTHOR,
        "dataset_version": dataset_version,
    }
    return DatasetSample(
        id=f"q_{index:03d}",
        user_input=question,
        reference=reference,
        reference_contexts=reference_contexts,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _question_type(synthesizer_name: Any) -> str | None:
    name = str(synthesizer_name or "").lower()
    if "multi_hop" in name:
        return "multi_doc"
    if "single_hop" in name:
        return "simple"
    return None


def _normalize(text: str) -> str:
    return " ".join(text.split())
