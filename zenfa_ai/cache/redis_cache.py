"""Redis caching layer for Zenfa AI build results.

Caches BuildResponse objects keyed by a hash of the request parameters.
Falls back gracefully when Redis is unavailable — the engine works
without caching, just slower.

Cache key strategy:
  zenfa:build:{sha256(purpose + budget_min + budget_max + sorted_component_ids + vendor + prefs)}

TTL: Configurable, default 30 minutes (prices/stock change frequently).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CACHE_PREFIX = "zenfa:build:"
DEFAULT_TTL = int(os.getenv("ZENFA_CACHE_TTL", "1800"))  # 30 minutes
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ──────────────────────────────────────────────
# Cache Key Generation
# ──────────────────────────────────────────────


def build_cache_key(
    purpose: str,
    budget_min: int,
    budget_max: int,
    component_ids: List[int],
    vendor_filter: Optional[str] = None,
    preferences: Optional[dict] = None,
) -> str:
    """Generate a deterministic cache key from request parameters.

    The key is a SHA-256 hash of the canonical request representation.
    Same inputs always produce the same key.
    """
    canonical = {
        "purpose": purpose,
        "budget_min": budget_min,
        "budget_max": budget_max,
        "component_ids": sorted(component_ids),
        "vendor_filter": vendor_filter or "",
        "preferences": json.dumps(preferences or {}, sort_keys=True),
    }
    raw = json.dumps(canonical, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{digest}"


def request_to_cache_key(request) -> str:
    """Extract cache key from a BuildRequest object."""
    component_ids = [c.id for c in request.components]
    prefs = None
    if request.preferences:
        prefs = request.preferences.model_dump() if hasattr(request.preferences, "model_dump") else {}

    purpose = request.purpose.value if hasattr(request.purpose, "value") else str(request.purpose)

    return build_cache_key(
        purpose=purpose,
        budget_min=request.budget_min,
        budget_max=request.budget_max,
        component_ids=component_ids,
        vendor_filter=request.vendor_filter,
        preferences=prefs,
    )


# ──────────────────────────────────────────────
# Redis Cache Client
# ──────────────────────────────────────────────


class BuildCache:
    """Redis-backed cache for build responses.

    Gracefully degrades when Redis is unavailable — all methods
    return None / silently fail instead of raising exceptions.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        self.ttl = ttl
        self._redis = None
        self._redis_url = redis_url
        self._available = False

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            # Test connection
            await self._redis.ping()
            self._available = True
            logger.info("Redis cache connected: %s", self._redis_url)
            return True

        except ImportError:
            logger.warning(
                "redis package not installed. "
                "Install with: pip install redis[hiredis]"
            )
            return False

        except Exception as e:
            logger.warning("Redis unavailable: %s — caching disabled", e)
            self._available = False
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._available = False

    @property
    def available(self) -> bool:
        """Whether Redis is connected and working."""
        return self._available

    async def get(self, key: str) -> Optional[str]:
        """Get a cached response by key. Returns None on miss or error."""
        if not self._available:
            return None
        try:
            data = await self._redis.get(key)
            if data:
                logger.debug("Cache HIT: %s", key)
            else:
                logger.debug("Cache MISS: %s", key)
            return data
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Cache a response. Returns True if successful."""
        if not self._available:
            return False
        try:
            await self._redis.set(key, value, ex=ttl or self.ttl)
            logger.debug("Cache SET: %s (TTL: %ds)", key, ttl or self.ttl)
            return True
        except Exception as e:
            logger.warning("Cache set failed: %s", e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cached entry. Returns True if successful."""
        if not self._available:
            return False
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed: %s", e)
            return False

    async def clear_all(self) -> int:
        """Clear all Zenfa build cache entries. Returns count deleted."""
        if not self._available:
            return 0
        try:
            keys = []
            async for key in self._redis.scan_iter(f"{CACHE_PREFIX}*"):
                keys.append(key)
            if keys:
                await self._redis.delete(*keys)
            logger.info("Cleared %d cache entries", len(keys))
            return len(keys)
        except Exception as e:
            logger.warning("Cache clear failed: %s", e)
            return 0

    async def stats(self) -> dict:
        """Get basic cache stats."""
        if not self._available:
            return {"available": False, "keys": 0}
        try:
            count = 0
            async for _ in self._redis.scan_iter(f"{CACHE_PREFIX}*"):
                count += 1
            return {"available": True, "keys": count, "ttl": self.ttl}
        except Exception as e:
            return {"available": False, "error": str(e)}


# ──────────────────────────────────────────────
# In-Memory Fallback Cache (for tests / no Redis)
# ──────────────────────────────────────────────


class InMemoryCache(BuildCache):
    """Simple dict-based cache for testing and fallback.

    No TTL enforcement — entries persist until cleared.
    """

    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        super().__init__(ttl=ttl)
        self._store: dict[str, str] = {}
        self._available = True

    async def connect(self) -> bool:
        self._available = True
        return True

    async def disconnect(self) -> None:
        self._store.clear()
        self._available = False

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        self._store[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self._store.pop(key, None)
        return True

    async def clear_all(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    async def stats(self) -> dict:
        return {"available": True, "keys": len(self._store), "ttl": self.ttl}
