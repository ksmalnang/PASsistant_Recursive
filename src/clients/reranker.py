"""Cross-encoder reranker client."""

from typing import Any

import httpx

from src.config import get_settings

_reranker: Any | None = None


class RemoteReranker:
    """Call an HTTP rerank API using model, base URL, and API key settings."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ):
        self.endpoint = self._build_endpoint(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        """Return reranker scores aligned to the input document order."""
        response = httpx.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self._parse_scores(response.json(), len(documents))

    def _parse_scores(self, payload: Any, document_count: int) -> list[float]:
        """Parse common rerank API response shapes into input-order scores."""
        if isinstance(payload, dict):
            if isinstance(payload.get("scores"), list):
                return [float(score) for score in payload["scores"]]

            results = payload.get("results") or payload.get("data")
            if isinstance(results, list):
                return self._parse_ranked_results(results, document_count)

        if isinstance(payload, list):
            return self._parse_ranked_results(payload, document_count)

        raise RuntimeError("Unsupported reranker response format")

    def _parse_ranked_results(self, results: list[Any], document_count: int) -> list[float]:
        """Parse result objects that may include original document indexes."""
        indexed_scores: list[float | None] = [None] * document_count
        ordered_scores: list[float] = []
        has_indexes = False

        for result in results:
            if not isinstance(result, dict):
                ordered_scores.append(float(result))
                continue

            score = result.get("relevance_score", result.get("score"))
            if score is None:
                raise RuntimeError("Reranker response result is missing a score")

            index = result.get("index")
            if isinstance(index, int):
                if not 0 <= index < document_count:
                    raise RuntimeError(f"Reranker response index is out of range: {index}")
                indexed_scores[index] = float(score)
                has_indexes = True
            else:
                ordered_scores.append(float(score))

        if has_indexes:
            if any(score is None for score in indexed_scores):
                raise RuntimeError("Reranker response did not include a score for every document")
            return [float(score) for score in indexed_scores if score is not None]

        return ordered_scores

    def _build_endpoint(self, base_url: str) -> str:
        """Normalize a base URL into a rerank endpoint URL."""
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/rerank"):
            return trimmed
        return f"{trimmed}/rerank"


def get_reranker() -> Any:
    """Return the process-wide reranker: remote when configured, else local FastEmbed."""
    global _reranker
    if _reranker is None:
        settings = get_settings()
        if not settings.RERANKER_MODEL:
            raise ValueError("RERANKER_MODEL is required when RETRIEVAL_STRATEGY=reranker")
        if settings.RERANKER_BASE_URL:
            if not settings.RERANKER_API_KEY:
                raise ValueError("RERANKER_API_KEY is required when RERANKER_BASE_URL is set")
            _reranker = RemoteReranker(
                base_url=settings.RERANKER_BASE_URL,
                api_key=settings.RERANKER_API_KEY,
                model=settings.RERANKER_MODEL,
            )
        else:
            _reranker = _build_local_reranker(settings.RERANKER_MODEL)
    return _reranker


def _build_local_reranker(model_name: str) -> Any:
    """Build a local FastEmbed cross-encoder when no remote endpoint is configured."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "FastEmbed reranker support is not installed. Install qdrant-client[fastembed]."
        ) from exc
    return TextCrossEncoder(model_name=model_name)
