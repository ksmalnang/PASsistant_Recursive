"""Produce per-sample responses and contexts for evaluation."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from src.eval.models import DatasetSample, SampleResult
from src.eval.openrouter import Usage, UsageTrackingChatModel
from src.eval.progress import Progress

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[\d+\]")
_SOURCES_SECTION_RE = re.compile(
    r"\n\s*(?:Sources|Sumber|References|Daftar Pustaka|Referensi)\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_citations(text: str) -> str:
    """Remove inline citation markers and any trailing sources block before scoring."""
    return _CITATION_RE.sub("", _SOURCES_SECTION_RE.sub("", text)).strip()


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    """Load pre-computed responses keyed by sample id."""
    import json

    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixture file must be a JSON object mapping sample ids to outputs")
    logger.info("Loaded fixtures for %d samples from %s", len(payload), path)
    return payload


def apply_fixtures(
    samples: list[DatasetSample],
    fixtures: dict[str, dict[str, Any]],
) -> list[SampleResult]:
    """Build results from a responses file; the testset itself stores no answers."""
    missing = sorted(sample.id for sample in samples if sample.id not in fixtures)
    if missing:
        raise ValueError(
            "Fixture mode needs a response per sample in --fixture; missing: "
            + ", ".join(missing[:10])
        )

    results: list[SampleResult] = []
    for sample in samples:
        fixture = fixtures[sample.id]
        if not isinstance(fixture, dict):
            raise ValueError(f"Fixture for {sample.id} must be a JSON object")
        results.append(
            SampleResult(
                sample=sample,
                response=str(fixture.get("response") or ""),
                retrieved_contexts=[
                    str(text) for text in (fixture.get("retrieved_contexts") or [])
                ],
            )
        )
    return results


class LivePipeline:
    """Run the project's retrieval and response pipeline for each sample."""

    def __init__(self, usage: Usage | None = None):
        self._usage = usage if usage is not None else Usage()
        self._retrieval_node: Any = None
        self._response_service: Any = None

    async def run(self, samples: list[DatasetSample]) -> list[SampleResult]:
        self._initialize()
        progress = Progress(len(samples), "pipeline")
        results: list[SampleResult] = []
        for sample in samples:
            started_at = time.perf_counter()
            try:
                results.append(await self._run_sample(sample))
            except Exception as exc:
                logger.warning(
                    "Pipeline failed for %s after %.1fs: %s",
                    sample.id,
                    time.perf_counter() - started_at,
                    exc,
                )
                results.append(
                    SampleResult(
                        sample=sample,
                        response="",
                        retrieved_contexts=[],
                        failures={"pipeline": str(exc)},
                    )
                )
            finally:
                progress.advance(sample.id)
        return results

    def _initialize(self) -> None:
        from src.clients.llm import get_response_llm, get_rewrite_llm
        from src.services.response_generation import ResponseGenerationService
        from src.utils.nodes.retrieval import RetrievalNode
        from src.utils.vector_store import VectorStoreTools

        retriever = VectorStoreTools()
        llm = get_response_llm()
        if llm is None:
            raise RuntimeError(
                "Live evaluation needs a response LLM. Set OPENAI_API_KEY (the OpenRouter key) "
                "so the pipeline generates real answers instead of its fallback text."
            )
        rewrite_llm = get_rewrite_llm()
        self._retrieval_node = RetrievalNode(
            retriever=retriever,
            llm_provider=(
                (lambda: UsageTrackingChatModel(rewrite_llm, self._usage))
                if rewrite_llm is not None
                else (lambda: None)
            ),
        )
        provider = lambda: UsageTrackingChatModel(llm, self._usage)  # noqa: E731
        self._response_service = ResponseGenerationService(llm_provider=provider)
        logger.info(
            "Live pipeline ready: strategy=%s, top_k=%d",
            retriever.retrieval_strategy,
            self._retrieval_node.top_k,
        )

    async def _run_sample(self, sample: DatasetSample) -> SampleResult:
        from langchain_core.messages import HumanMessage

        from src.utils.state import AgentState

        state = AgentState(
            messages=[HumanMessage(content=sample.user_input)],
            requires_retrieval=True,
            retrieval_query=sample.user_input,
            current_intent="query_document",
        )
        state = state.model_copy(update=await self._retrieval_node.run(state))
        contexts = [
            str(chunk.get("text") or chunk.get("child_text") or "")
            for chunk in state.retrieved_chunks
        ]
        response = self._response_service.generate(state)
        return SampleResult(
            sample=sample,
            response=response.get("draft_response") or "",
            retrieved_contexts=[context for context in contexts if context],
        )
