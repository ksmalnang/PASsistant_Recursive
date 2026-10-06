"""Health endpoints."""

import logging

from fastapi import APIRouter
from qdrant_client import QdrantClient

from src.api.models import DependencyHealthResponse, HealthResponse
from src.clients.redis import get_cache
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def _check_redis_health() -> DependencyHealthResponse:
    """Return Redis connectivity status."""
    if not settings.REDIS_URL:
        return DependencyHealthResponse(
            status="disabled", detail="REDIS_URL is not configured."
        )

    client = get_cache().client
    if client is None:
        return DependencyHealthResponse(
            status="disabled", detail="Redis client is unavailable."
        )

    try:
        client.ping()
    except Exception:
        logger.warning("Redis health check failed", exc_info=True)
        return DependencyHealthResponse(
            status="unhealthy", detail="Redis did not respond to ping."
        )
    return DependencyHealthResponse(status="healthy", detail="Redis responded to ping.")


def _check_qdrant_health() -> DependencyHealthResponse:
    """Return Qdrant connectivity status."""
    try:
        QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=1,
        ).get_collections()
    except Exception:
        logger.warning("Qdrant health check failed", exc_info=True)
        return DependencyHealthResponse(
            status="unhealthy", detail="Qdrant did not respond."
        )
    return DependencyHealthResponse(
        status="healthy", detail="Qdrant responded to collection listing."
    )


# Plain `def` on purpose: both checks are blocking calls, so FastAPI runs this
# in a threadpool instead of freezing the event loop (Qdrant can take ~1s).
@router.get("/health", summary="Health check")
def health_check() -> HealthResponse:
    """Return the current API health status and runtime environment metadata."""
    redis_status = _check_redis_health()
    qdrant_status = _check_qdrant_health()
    unhealthy = "unhealthy" in (redis_status.status, qdrant_status.status)
    return HealthResponse(
        status="degraded" if unhealthy else "healthy",
        version="0.1.0",
        environment=settings.APP_ENV,
        redis=redis_status,
        qdrant=qdrant_status,
    )
