"""Redis-backed cache client helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from redis import Redis
from redis.exceptions import RedisError

from src.config import get_settings

logger = logging.getLogger(__name__)
_cache_instance: RedisCache | None = None


class RedisCache:
    """Small Redis wrapper for JSON and set-based cache operations."""

    def __init__(
        self,
        redis_url: str | None,
        key_prefix: str,
        default_ttl_seconds: int,
    ):
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = key_prefix.strip(":")
        self.client: Redis | None = None

        if redis_url:
            self.client = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )

    @property
    def enabled(self) -> bool:
        """Whether Redis is configured for this process."""
        return self.client is not None

    def get_json(self, key: str) -> Any | None:
        """Get a JSON value from cache."""
        if not self.client:
            return None
        try:
            cached = self.client.get(self._key(key))
        except RedisError as exc:
            logger.warning("Redis get failed for key %s: %s", key, exc)
            return None

        if cached is None:
            return None
        if not isinstance(cached, (str, bytes, bytearray)):
            logger.warning("Redis returned non-JSON-compatible payload for key %s", key)
            return None
        return json.loads(cached)

    def set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a JSON value in cache."""
        if not self.client:
            return
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True)
        expiry = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        try:
            if expiry <= 0:
                self.client.set(self._key(key), payload)
            else:
                self.client.setex(self._key(key), expiry, payload)
        except RedisError as exc:
            logger.warning("Redis set failed for key %s: %s", key, exc)

    def add_to_set(
        self,
        key: str,
        *values: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Add members to a Redis set."""
        if not self.client or not values:
            return
        expiry = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        namespaced_key = self._key(key)
        try:
            self.client.sadd(namespaced_key, *values)
            if expiry > 0:
                self.client.expire(namespaced_key, expiry)
        except RedisError as exc:
            logger.warning("Redis sadd failed for key %s: %s", key, exc)

    def get_set_members(self, key: str) -> set[str]:
        """Read all members from a Redis set."""
        if not self.client:
            return set()
        try:
            members = cast(set[Any], self.client.smembers(self._key(key)))
            return {str(value) for value in members}
        except RedisError as exc:
            logger.warning("Redis smembers failed for key %s: %s", key, exc)
            return set()

    def delete(self, key: str) -> None:
        """Delete a single cache key."""
        self.delete_many([key])

    def delete_many(self, keys: list[str]) -> None:
        """Delete multiple cache keys."""
        if not self.client or not keys:
            return
        try:
            self.client.delete(*[self._key(key) for key in keys])
        except RedisError as exc:
            logger.warning("Redis delete failed for keys %s: %s", keys, exc)

    def delete_prefix(self, prefix: str) -> None:
        """Delete all cache keys under a logical prefix."""
        if not self.client:
            return
        try:
            keys = list(self.client.scan_iter(match=f"{self._key(prefix)}*"))
            if keys:
                self.client.delete(*keys)
        except RedisError as exc:
            logger.warning("Redis prefix delete failed for %s: %s", prefix, exc)

    def _key(self, key: str) -> str:
        """Build the fully qualified Redis key."""
        return f"{self.key_prefix}:{key}"


def get_cache() -> RedisCache:
    """Get the configured process-wide Redis cache client wrapper."""
    global _cache_instance
    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = RedisCache(
            redis_url=settings.REDIS_URL,
            key_prefix=settings.REDIS_KEY_PREFIX,
            default_ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS,
        )
    return _cache_instance


def close_cache() -> None:
    """Close the process-wide cache client if one was created."""
    if _cache_instance is not None and _cache_instance.client is not None:
        _cache_instance.client.close()
