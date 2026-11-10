"""Redis response cache for the scoring endpoint."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal, Protocol

import redis.asyncio as redis

from artifact import Bundle
from predictor import ScoreResult

logger = logging.getLogger(__name__)

CacheStatus = Literal["HIT", "MISS", "BYPASS"]

KEY_PREFIX = "score"


class CacheSettings(Protocol):
    """Configuration the cache needs.

    A protocol rather than an import of the settings class, matching how
    `predictor.ThresholdSettings` keeps the layer below HTTP independent.
    """

    redis_url: str | None
    cache_ttl_seconds: int


def model_version(bundle: Bundle) -> str:
    """Return a stable identifier for the artifact behind the cache.

    Derived from the whole metadata dictionary rather than one field: a
    development artifact may carry no training timestamp, and two such
    artifacts must still produce different keys.

    Args:
        bundle: Loaded artifact.

    Returns:
        A hex digest identifying this artifact.
    """
    canonical = json.dumps(bundle.metadata, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def cache_key(version: str, payload: dict[str, Any]) -> str:
    """Build the cache key for one request.

    The key covers the whole payload and the model version. Without the
    version a retrained model would serve its predecessor's answers.

    Args:
        version: Identifier from `model_version`.
        payload: Validated request fields.

    Returns:
        The Redis key.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}:{version}:{digest}"


class ScoreCache:
    """Read-through cache for scoring results.

    Attributes:
        client: Redis client, or `None` when the cache is disabled.
        version: Model version included in every key.
        ttl_seconds: Expiry applied on write.
    """

    def __init__(
        self,
        client: redis.Redis | None,
        version: str,
        ttl_seconds: int,
    ) -> None:
        """Take a client and the key parameters.

        Args:
            client: Redis client, or `None` to disable the cache.
            version: Identifier from `model_version`.
            ttl_seconds: Expiry applied on write.
        """
        self.client = client
        self.version = version
        self.ttl_seconds = ttl_seconds
        self._unreachable_logged = False

    @property
    def enabled(self) -> bool:
        """Whether a client is attached."""
        return self.client is not None

    @classmethod
    def disabled(cls) -> ScoreCache:
        """Build a cache that always reports BYPASS."""
        return cls(client=None, version="", ttl_seconds=0)

    @classmethod
    def from_settings(cls, settings: CacheSettings, bundle: Bundle) -> ScoreCache:
        """Build the cache from configuration, degrading when unconfigured.

        Args:
            settings: Service settings carrying the URL and the TTL.
            bundle: Loaded artifact, used for the model version.

        Returns:
            A connected cache, or a disabled one when no URL is configured.
        """
        if not settings.redis_url:
            logger.warning("CS_REDIS_URL not set; serving without a cache.")
            return cls.disabled()
        pool = redis.ConnectionPool.from_url(settings.redis_url)
        return cls(
            client=redis.Redis.from_pool(pool),
            version=model_version(bundle),
            ttl_seconds=settings.cache_ttl_seconds,
        )

    def _degrade(self, exc: Exception) -> None:
        """Log an unreachable cache once per process."""
        if not self._unreachable_logged:
            logger.warning("Cache unreachable, serving uncached: %s", exc)
            self._unreachable_logged = True

    async def get(
        self, payload: dict[str, Any]
    ) -> tuple[ScoreResult | None, CacheStatus]:
        """Look one payload up.

        Args:
            payload: Validated request fields.

        Returns:
            The stored result and HIT, or `None` with MISS or BYPASS.
        """
        if self.client is None:
            return None, "BYPASS"
        try:
            raw = await self.client.get(cache_key(self.version, payload))
        except redis.RedisError as exc:
            self._degrade(exc)
            return None, "BYPASS"
        if raw is None:
            return None, "MISS"
        return ScoreResult(**json.loads(raw)), "HIT"

    async def set(self, payload: dict[str, Any], result: ScoreResult) -> None:
        """Store one result under the TTL.

        Args:
            payload: Validated request fields.
            result: Result to store.
        """
        if self.client is None:
            return
        body = json.dumps(
            {
                "probability": result.probability,
                "decision": result.decision,
                "threshold": result.threshold,
            }
        )
        try:
            await self.client.set(
                cache_key(self.version, payload), body, ex=self.ttl_seconds
            )
        except redis.RedisError as exc:
            self._degrade(exc)

    async def aclose(self) -> None:
        """Close the client and its pool."""
        if self.client is not None:
            await self.client.aclose()
