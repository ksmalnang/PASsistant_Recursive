"""Cross-encoder reranker client."""

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

_reranker: Any | None = None


class RemoteReranker:
    """Call an HTTP rerank API using model, base URL, and API key settings."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        fallback: Any | None = None,
        provider_name: str = "remote",
    ):
        self.endpoint = self._build_endpoint(base_url)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback
        self.provider_name = provider_name

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        """Return reranker scores aligned to the input document order."""
        if not documents:
            return []

        try:
            return self._call_remote(query=query, documents=documents)
        except Exception as exc:
            if self.fallback is not None:
                fallback_name = getattr(
                    self.fallback,
                    "provider_name",
                    getattr(self.fallback, "model", type(self.fallback).__name__),
                )
                logger.warning(
                    "Primary reranker (%s, model=%s) failed: %s. Falling back to %s.",
                    self.provider_name,
                    self.model,
                    exc,
                    fallback_name,
                )
                try:
                    return self._call_fallback(query=query, documents=documents)
                except Exception as fallback_exc:
                    logger.error(
                        "Fallback reranker (%s) also failed: %s. Original error: %s",
                        fallback_name,
                        fallback_exc,
                        exc,
                    )
                    raise RuntimeError(
                        f"Both primary reranker ({self.provider_name}) and fallback reranker "
                        f"({fallback_name}) failed. Primary error: {exc}. Fallback error: {fallback_exc}"
                    ) from fallback_exc
            raise

    def _call_remote(self, *, query: str, documents: list[str]) -> list[float]:
        """Send rerank request to the remote HTTP endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider_name == "openrouter":
            headers["X-Title"] = "PASsistant"

        response = httpx.post(
            self.endpoint,
            headers=headers,
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

    def _call_fallback(self, *, query: str, documents: list[str]) -> list[float]:
        """Invoke fallback reranker (remote or local)."""
        fallback = self.fallback
        if fallback is not None and hasattr(fallback, "rerank"):
            try:
                return [
                    float(s)
                    for s in fallback.rerank(query=query, documents=documents)
                ]
            except TypeError:
                results = fallback.rerank(query, documents)
                return [float(getattr(r, "score", r)) for r in results]
        raise RuntimeError("Configured fallback reranker does not support rerank method")

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
    """Return the process-wide reranker: OpenRouter primary with Jina fallback, or configured provider."""
    global _reranker
    if _reranker is not None:
        return _reranker

    settings = get_settings()
    model = getattr(settings, "RERANKER_MODEL", None)
    if not model:
        raise ValueError("RERANKER_MODEL is required when RETRIEVAL_STRATEGY=reranker")

    provider = getattr(settings, "RERANKER_PROVIDER", "openrouter") or "openrouter"
    fallback_provider = getattr(settings, "RERANKER_FALLBACK_PROVIDER", "jina") or "jina"

    if provider == "local":
        _reranker = _build_local_reranker(model)
        return _reranker

    # Determine fallback reranker (e.g. Jina)
    fallback_reranker = None
    if fallback_provider == "jina":
        jina_key = (
            getattr(settings, "RERANKER_FALLBACK_API_KEY", None)
            or getattr(settings, "JINA_API_KEY", None)
            or (
                getattr(settings, "RERANKER_API_KEY", None)
                if (getattr(settings, "RERANKER_API_KEY", "") or "").startswith("jina_")
                else None
            )
        )
        if jina_key:
            jina_base_url = (
                getattr(settings, "RERANKER_FALLBACK_BASE_URL", None)
                or getattr(settings, "JINA_BASE_URL", None)
                or (
                    getattr(settings, "RERANKER_BASE_URL", None)
                    if "jina" in (getattr(settings, "RERANKER_BASE_URL", "") or "")
                    else None
                )
                or "https://api.jina.ai/v1"
            )
            jina_model = (
                getattr(settings, "RERANKER_FALLBACK_MODEL", None)
                or getattr(settings, "JINA_RERANKER_MODEL", None)
                or "jina-reranker-v3"
            )
            fallback_reranker = RemoteReranker(
                base_url=jina_base_url,
                api_key=jina_key,
                model=jina_model,
                provider_name="jina",
            )
    elif fallback_provider == "local":
        try:
            fallback_reranker = _build_local_reranker(model)
        except Exception as exc:
            logger.debug("FastEmbed local reranker not available for fallback: %s", exc)

    # Determine primary reranker
    if provider == "openrouter":
        openrouter_key = (
            (
                getattr(settings, "RERANKER_API_KEY", None)
                if not (getattr(settings, "RERANKER_API_KEY", "") or "").startswith("jina_")
                else None
            )
            or getattr(settings, "OPENROUTER_API_KEY", None)
            or getattr(settings, "OPENAI_API_KEY", None)
        )
        openrouter_base_url = (
            (
                getattr(settings, "RERANKER_BASE_URL", None)
                if "jina" not in (getattr(settings, "RERANKER_BASE_URL", "") or "")
                else None
            )
            or getattr(settings, "OPENROUTER_BASE_URL", None)
            or getattr(settings, "OPENAI_BASE_URL", None)
            or "https://openrouter.ai/api/v1"
        )
        if openrouter_key:
            _reranker = RemoteReranker(
                base_url=openrouter_base_url,
                api_key=openrouter_key,
                model=model,
                provider_name="openrouter",
                fallback=fallback_reranker,
            )
            return _reranker
        elif fallback_reranker is not None:
            logger.warning(
                "OpenRouter API key is missing for reranking; using fallback %s directly.",
                fallback_reranker.provider_name,
            )
            _reranker = fallback_reranker
            return _reranker
        else:
            try:
                _reranker = _build_local_reranker(model)
                return _reranker
            except Exception as exc:
                raise ValueError(
                    "OPENAI_API_KEY (or RERANKER_API_KEY) is required when using OpenRouter reranker"
                ) from exc

    # Custom or direct remote endpoint
    base_url = getattr(settings, "RERANKER_BASE_URL", None)
    api_key = getattr(settings, "RERANKER_API_KEY", None)
    if base_url:
        if not api_key:
            raise ValueError("RERANKER_API_KEY is required when RERANKER_BASE_URL is set")
        _reranker = RemoteReranker(
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider_name=provider,
            fallback=fallback_reranker,
        )
        return _reranker

    _reranker = _build_local_reranker(model)
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
