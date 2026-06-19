"""Search, result hydration, and ranking helpers for vector retrieval."""

import hashlib
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, SparseVector

from src.utils.cache import RedisCache
from src.utils.state import DocumentType

logger = logging.getLogger(__name__)
RetrievalStrategy = Literal["similarity", "rrf", "reranker"]


class SearchOperations:
    """Dense, hybrid, and reranked retrieval helpers for flat recursive chunks."""

    _RRF_RELATIVE_KEEP_RATIO = 0.5
    _RERANKER_FINAL_SCORE_THRESHOLD = 0.0
    _SEARCH_RESULT_SCHEMA_VERSION = "score-fields-v2"
    cache: RedisCache
    retrieval_strategy: RetrievalStrategy
    reranker_candidate_multiplier: int
    reranker_model: str | None
    reranker_base_url: str | None
    bm25_vector_name: str
    collection_name: str
    _RRF_RANK_CONSTANT: int

    if TYPE_CHECKING:

        def _get_embeddings(self) -> Embeddings: ...
        def _ensure_collection_for_vector_size(self, vector_size: int) -> None: ...
        def _get_reranker(self) -> Any: ...
        def _supports_bm25_vectors(self) -> bool: ...
        def _get_bm25_cache_signature(self) -> str: ...
        def _build_bm25_vector(
            self, text: str, *, is_query: bool = False
        ) -> SparseVector | None: ...
        def _query_points(self, **kwargs: Any) -> list[Any]: ...
        def _rrf_weight(self, rank: int) -> float: ...

    async def search_similar(
        self,
        query: str,
        document_type: DocumentType | None = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> list[dict]:
        """
        Retrieve relevant indexed document context using the configured strategy.

        Args:
            query: Search query text
            document_type: Filter by document type
            top_k: Maximum number of retrieved chunk results to return
            score_threshold: Dense-stage minimum score used only by pure similarity retrieval

        Returns:
            Flat list of retrieved chunk results
        """
        query_embedding = await self._get_embeddings().aembed_query(query)
        self._ensure_collection_for_vector_size(len(query_embedding))
        cache_key = self._build_search_cache_key(
            query=query,
            document_type=document_type,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        cached_results = self.cache.get_json(cache_key)
        if cached_results is not None:
            logger.debug("Returning cached retrieval results for %s", cache_key)
            return cached_results

        query_filter = self._build_document_filter(document_type)
        results = self._search_candidates(
            query=query,
            query_embedding=query_embedding,
            query_filter=query_filter,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        hydrated = self._hydrate_results(results)
        ordered = sorted(
            hydrated.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        reranked = self._rerank_results(query, ordered)
        filtered = self._filter_final_results(reranked)
        limited = filtered[:top_k]
        self.cache.set_json(cache_key, limited)
        return limited

    def _hydrate_results(self, results: list[Any]) -> dict[str, dict[str, Any]]:
        """Build a flat map of result ID → chunk result."""
        hydrated: dict[str, dict[str, Any]] = {}
        for result in results:
            payload = dict(self._get_result_payload(result))
            result_id = str(self._get_result_id(result))
            result_score = self._get_result_score(result)
            vector_score = self._get_optional_result_score(result, "dense_score")
            bm25_score = self._get_optional_result_score(result, "bm25_score")
            rrf_score = self._get_optional_result_score(result, "rrf_score")
            hydrated[result_id] = {
                "chunk_id": payload.get("chunk_id", result_id),
                "text": payload.get("text", ""),
                "final_score": result_score,
                "retrieval_score": result_score,
                "score": result_score,
                "vector_score": vector_score if vector_score is not None else result_score,
                "bm25_score": bm25_score,
                "rrf_score": rrf_score,
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "doc_title": payload.get("doc_title"),
                "document_type": payload.get("document_type"),
                "chunk_type": payload.get("chunk_type"),
                "source_locations": payload.get("source_locations", []),
            }
        return hydrated

    def _build_document_filter(self, document_type: DocumentType | None) -> Filter | None:
        """Build an optional filter for a specific document type."""
        if not document_type:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=document_type.value),
                )
            ]
        )

    def _resolve_candidate_limit(self, top_k: int) -> int:
        """Choose how many first-stage hits to fetch before fusion or reranking."""
        multiplier = 3
        if self.retrieval_strategy in {"rrf", "reranker"}:
            multiplier = self.reranker_candidate_multiplier
        return max(top_k * multiplier, top_k)

    def _rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply a second-stage cross-encoder reranker when configured."""
        if self.retrieval_strategy != "reranker" or len(results) < 2:
            return results

        documents = [self._build_reranker_document(result) for result in results]
        scores = self._score_reranker_documents(query=query, documents=documents)
        reranked: list[dict[str, Any]] = []
        for index, reranker_score in sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        ):
            result = dict(results[index])
            result["reranker_score"] = float(reranker_score)
            result["final_score"] = float(reranker_score)
            result["score"] = result["final_score"]
            reranked.append(result)
        return reranked

    def _filter_final_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply strategy-aware post-ranking filtering without harming first-stage recall."""
        if not results:
            return []

        if self.retrieval_strategy == "reranker":
            return [
                result
                for result in results
                if float(result.get("reranker_score", result.get("score", 0.0)))
                > self._RERANKER_FINAL_SCORE_THRESHOLD
            ]

        if self.retrieval_strategy == "rrf":
            top_score = float(results[0].get("rrf_score") or results[0].get("score") or 0.0)
            if top_score <= 0.0:
                return []
            return [
                result
                for result in results
                if float(result.get("rrf_score") or result.get("score") or 0.0)
                >= top_score * self._RRF_RELATIVE_KEEP_RATIO
            ]

        return results

    def _score_reranker_documents(self, query: str, documents: list[str]) -> list[float]:
        """Score documents with the configured reranker model."""
        scores = [
            float(score)
            for score in self._get_reranker().rerank(
                query=query,
                documents=documents,
            )
        ]
        if len(scores) != len(documents):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores: "
                f"expected={len(documents)}, received={len(scores)}"
            )
        return scores

    def _build_reranker_document(self, result: dict[str, Any]) -> str:
        """Build the text payload passed to the cross-encoder reranker."""
        return result.get("text", "")

    def _build_search_cache_key(
        self,
        query: str,
        document_type: DocumentType | None,
        top_k: int,
        score_threshold: float,
    ) -> str:
        """Build a stable Redis key for retrieval results."""
        digest = hashlib.sha256(
            (
                f"{query}|{document_type.value if document_type else 'all'}|"
                f"{top_k}|{score_threshold}|{self.retrieval_strategy}|"
                f"{self._SEARCH_RESULT_SCHEMA_VERSION}|"
                f"{self.reranker_model or 'none'}|{self.reranker_base_url or 'local'}|"
                f"{self.reranker_candidate_multiplier}|"
                f"{self._supports_bm25_vectors()}|{self._RRF_RANK_CONSTANT}|"
                f"{self._get_bm25_cache_signature()}"
            ).encode()
        ).hexdigest()
        return f"search:{digest}"

    def _search_candidates(
        self,
        query: str,
        query_embedding: list[float],
        query_filter: Filter | None,
        top_k: int,
        score_threshold: float,
    ) -> list[Any]:
        """Run the configured first-stage retriever.

        `score_threshold` is applied only in similarity mode. RRF gathers dense and
        BM25 candidates without an absolute score cutoff, then filters by relative
        fused score. Reranker mode also skips first-stage score filtering so the
        second-stage cross-encoder can decide the final ranking.
        """
        if self.retrieval_strategy in {"rrf", "reranker"} and self._supports_bm25_vectors():
            bm25_query = self._build_bm25_vector(query, is_query=True)
            if bm25_query is not None:
                return self._rrf_search(
                    query_embedding=query_embedding,
                    bm25_query=bm25_query,
                    query_filter=query_filter,
                    top_k=top_k,
                )
        candidate_threshold = None if self.retrieval_strategy == "reranker" else score_threshold
        return self._dense_search(
            query_embedding=query_embedding,
            query_filter=query_filter,
            top_k=top_k,
            score_threshold=candidate_threshold,
        )

    def _dense_search(
        self,
        query_embedding: list[float],
        query_filter: Filter | None,
        top_k: int,
        score_threshold: float | None,
    ) -> list[Any]:
        """Run a dense-vector nearest-neighbor search."""
        return self._query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=self._resolve_candidate_limit(top_k),
            score_threshold=score_threshold,
            with_payload=True,
        )

    def _rrf_search(
        self,
        query_embedding: list[float],
        bm25_query: SparseVector,
        query_filter: Filter | None,
        top_k: int,
    ) -> list[Any]:
        """Fuse dense and BM25-ranked candidates with Reciprocal Rank Fusion."""
        candidate_limit = self._resolve_candidate_limit(top_k)
        dense_points = self._query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )
        bm25_points = self._query_points(
            collection_name=self.collection_name,
            query=bm25_query,
            using=self.bm25_vector_name,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )
        return self._fuse_with_rrf(
            dense_points=dense_points,
            bm25_points=bm25_points,
            limit=candidate_limit,
        )

    def _fuse_with_rrf(
        self,
        dense_points: list[Any],
        bm25_points: list[Any],
        limit: int,
    ) -> list[Any]:
        """Merge dense and BM25 rankings using the standard RRF formula."""
        fused_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        point_by_id: dict[str, Any] = {}

        for rank, point in enumerate(dense_points, start=1):
            point_id = str(self._get_result_id(point))
            fused_scores[point_id] = fused_scores.get(point_id, 0.0) + self._rrf_weight(rank)
            dense_scores[point_id] = self._get_result_score(point)
            point_by_id.setdefault(point_id, point)

        for rank, point in enumerate(bm25_points, start=1):
            point_id = str(self._get_result_id(point))
            fused_scores[point_id] = fused_scores.get(point_id, 0.0) + self._rrf_weight(rank)
            bm25_scores[point_id] = self._get_result_score(point)
            point_by_id.setdefault(point_id, point)

        fused: list[Any] = []
        for point_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
            source = point_by_id[point_id]
            fused.append(
                SimpleNamespace(
                    id=self._get_result_id(source),
                    score=float(score),
                    payload=dict(self._get_result_payload(source)),
                    dense_score=dense_scores.get(point_id),
                    bm25_score=bm25_scores.get(point_id),
                    rrf_score=float(score),
                )
            )
        return fused[:limit]

    def _get_result_id(self, result: Any) -> Any:
        """Read a point identifier from either a ScoredPoint or a test double."""
        return result.id

    def _get_result_score(self, result: Any) -> float:
        """Read the main ranking score from either a ScoredPoint or a test double."""
        return float(result.score)

    def _get_optional_result_score(self, result: Any, field: str) -> float | None:
        """Read an auxiliary ranking score when it is available."""
        value = getattr(result, field, None)
        if value is None:
            return None
        return float(value)

    def _get_result_payload(self, result: Any) -> dict[str, Any]:
        """Read a result payload from either a ScoredPoint or a test double."""
        return result.payload or {}
