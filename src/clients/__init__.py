"""Process-wide clients for external systems."""

import logging

from src.clients.decision import (
    JevDecisionClient,
    close_decision_client,
    get_decision_client,
)
from src.clients.embeddings import get_embeddings
from src.clients.llm import get_llm
from src.clients.ocr import close_zai_client, get_zai_client
from src.clients.qdrant import close_qdrant_client, get_qdrant_client
from src.clients.redis import RedisCache, close_cache, get_cache
from src.clients.reranker import RemoteReranker, get_reranker
from src.clients.student_records import (
    InMemoryStudentRecordRepository,
    get_student_repository,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InMemoryStudentRecordRepository",
    "JevDecisionClient",
    "RedisCache",
    "RemoteReranker",
    "close_all_clients",
    "close_cache",
    "close_decision_client",
    "close_qdrant_client",
    "close_zai_client",
    "get_cache",
    "get_decision_client",
    "get_embeddings",
    "get_llm",
    "get_qdrant_client",
    "get_reranker",
    "get_student_repository",
    "get_zai_client",
]


def close_all_clients() -> None:
    """Close every process-wide client. Safe to call more than once."""
    for close in (close_qdrant_client, close_cache, close_zai_client, close_decision_client):
        try:
            close()
        except Exception as exc:
            logger.warning("Shutdown cleanup failed in %s: %s", close.__name__, exc)
