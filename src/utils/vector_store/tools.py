"""Qdrant-backed retrieval helpers."""

from typing import Any

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from src.clients.embeddings import get_embeddings
from src.clients.qdrant import get_qdrant_client
from src.clients.redis import RedisCache, get_cache
from src.clients.reranker import RemoteReranker
from src.config import get_settings
from src.utils.vector_store.bm25 import TOKEN_PATTERN, BM25VectorOperations
from src.utils.vector_store.collection import CollectionOperations
from src.utils.vector_store.search import SearchOperations


class VectorStoreTools(
    SearchOperations,
    CollectionOperations,
    BM25VectorOperations,
):
    """
    Vector-store facade for retrieval.

    Handles embeddings and retrieval strategies including dense similarity,
    RRF hybrid retrieval, and optional second-stage reranking.
    """

    _BM25_VECTOR_NAME = "bm25"
    _RRF_RANK_CONSTANT = 60
    _TOKEN_PATTERN = TOKEN_PATTERN

    def __init__(
        self,
        client: QdrantClient | None = None,
        cache: RedisCache | None = None,
    ):
        settings = get_settings()
        self.client = client or get_qdrant_client()
        self.cache = cache or get_cache()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.VECTOR_SIZE
        self.retrieval_strategy = settings.RETRIEVAL_STRATEGY
        self.reranker_model = settings.RERANKER_MODEL
        self.reranker_base_url = settings.RERANKER_BASE_URL
        self.reranker_api_key = settings.RERANKER_API_KEY
        self.reranker_candidate_multiplier = max(
            settings.RERANKER_CANDIDATE_MULTIPLIER,
            1,
        )
        self.reranker: Any | None = None
        self.bm25_vector_name = self._BM25_VECTOR_NAME
        self.bm25_vectors_enabled: bool | None = None
        self._rrf_warning_emitted = False

    def _get_embeddings(self) -> Embeddings:
        """Return the shared embeddings client, created lazily on first use."""
        return get_embeddings()

    def _get_reranker(self) -> Any:
        """Create the reranker lazily so startup stays cheap when it is disabled."""
        if self.reranker is None:
            if not self.reranker_model:
                raise ValueError("RERANKER_MODEL is required when RETRIEVAL_STRATEGY=reranker")
            if self.reranker_base_url:
                if not self.reranker_api_key:
                    raise ValueError("RERANKER_API_KEY is required when RERANKER_BASE_URL is set")
                self.reranker = RemoteReranker(
                    base_url=self.reranker_base_url,
                    api_key=self.reranker_api_key,
                    model=self.reranker_model,
                )
                return self.reranker

            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "FastEmbed reranker support is not installed. Install qdrant-client[fastembed]."
                ) from exc
            self.reranker = TextCrossEncoder(model_name=self.reranker_model)
        return self.reranker
