"""Health endpoints."""

import logging

from fastapi import APIRouter, HTTPException, status
from qdrant_client import QdrantClient

from src.api.models import DependencyHealthResponse, ErrorResponse, HealthResponse
from src.clients.redis import get_cache
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def _check_redis_health() -> DependencyHealthResponse:
    """Return Redis connectivity status."""
    if not settings.REDIS_URL:
        return DependencyHealthResponse(
            status="disabled",
            detail="REDIS_URL is not configured.",
        )

    cache = get_cache()
    if cache.client is None:
        return DependencyHealthResponse(
            status="disabled",
            detail="Redis client is unavailable.",
        )

    try:
        cache.client.ping()
        return DependencyHealthResponse(
            status="healthy",
            detail="Redis responded to ping.",
        )
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc, exc_info=True)
        return DependencyHealthResponse(
            status="unhealthy",
            detail=str(exc),
        )


def _check_qdrant_health() -> DependencyHealthResponse:
    """Return Qdrant connectivity status."""
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=1,
        )
        client.get_collections()
        return DependencyHealthResponse(
            status="healthy",
            detail="Qdrant responded to collection listing.",
        )
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc, exc_info=True)
        return DependencyHealthResponse(
            status="unhealthy",
            detail=str(exc),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Return the current API health status and runtime environment metadata.",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "The service could not produce a health response.",
        }
    },
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        redis_status = _check_redis_health()
        qdrant_status = _check_qdrant_health()
        overall_status = (
            "healthy"
            if redis_status.status != "unhealthy" and qdrant_status.status != "unhealthy"
            else "degraded"
        )
        return HealthResponse(
            status=overall_status,
            version="0.1.0",
            environment=settings.APP_ENV,
            redis=redis_status,
            qdrant=qdrant_status,
        )
    except Exception as exc:
        logger.error("Health endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate health response.",
        ) from exc
