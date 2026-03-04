"""API key authentication and rate limiting for the vendor gateway.

Only applies to /v1/* routes. The internal gateway (/internal/*)
is unauthenticated (service-to-service behind a firewall).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader


# ──────────────────────────────────────────────
# API Key Validation
# ──────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

from zenfa_ai.cache.redis_cache import BuildCache

# Simulated DB call – In production this would query PostgreSQL.
# For B2B keys, we will cache the result in Redis.
async def _fetch_api_key_from_db(api_key: str) -> bool:
    """Mock database check."""
    raw = os.getenv("ZENFA_API_KEYS", "")
    valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
    return api_key in valid_keys


async def _is_valid_api_key(api_key: str, cache: Optional[BuildCache]) -> bool:
    """Check if API key is valid using Redis cache as first layer."""
    if not cache or not cache._available:
        return await _fetch_api_key_from_db(api_key)

    cache_key = f"auth:apikey:{api_key}"
    
    # 1. Check Cache
    cached_status = await cache.get(cache_key)
    if cached_status is not None:
        return cached_status == "valid"

    # 2. Cache Miss -> Check DB
    is_valid = await _fetch_api_key_from_db(api_key)
    
    # 3. Store in cache (cache for 5 minutes)
    await cache.set(cache_key, "valid" if is_valid else "invalid", ttl=300)
    
    return is_valid


async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """FastAPI dependency — validates X-API-Key header against Redis/DB.

    Raises 401 if missing, 403 if invalid.
    """
    
    origin = request.headers.get("origin")
    if origin:
        # Simple domain verification rule (in reality this would be stored alongside the API Key in the DB)
        allowed_origins = os.getenv("CORS_ORIGINS", "").split(",")
        allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
        
        if origin not in allowed_origins and allowed_origins:
           raise HTTPException(
               status_code=403,
               detail="Invalid Origin for API Key",
           )
           
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
        )

    # In app.py, the cache is attached to app state during lifespan
    cache: Optional[BuildCache] = getattr(request.app.state, "cache", None)

    is_valid = await _is_valid_api_key(api_key, cache)

    if not is_valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


# ──────────────────────────────────────────────
# Rate Limiting (in-memory, per API key)
# ──────────────────────────────────────────────

# Simple sliding window counter for fallback memory cache
_rate_limits: dict[str, list[float]] = defaultdict(list)

# Default: 60 requests per minute per key
RATE_LIMIT_WINDOW = 60  # seconds


def _get_rate_limit_max() -> int:
    """Read rate limit max at call time (allows test patching)."""
    return int(os.getenv("ZENFA_RATE_LIMIT_MAX", "60"))


async def _redis_sliding_window(cache: BuildCache, api_key: str, now: float, window: int, limit: int) -> bool:
    """Execute Redis sliding window rate limit using an atomic pipeline.
    
    Returns True if allowed, False if limit exceeded.
    """
    key = f"ratelimit:{api_key}"
    window_start = now - window
    
    # Use the raw redis client for advanced commands
    redis_client = cache._redis
    
    async with redis_client.pipeline(transaction=True) as pipe:
        # ZREMRANGEBYSCORE key -inf (now - window)
        pipe.zremrangebyscore(key, "-inf", window_start) # Remove old requests
        # ZADD key score member
        pipe.zadd(key, {str(now): now})                  # Add current request 
        # ZCARD key
        pipe.zcard(key)                                  # Count requests in window
        # EXPIRE key window
        pipe.expire(key, window)                         # Set TTL to clean up idle keys
        
        results = await pipe.execute()
    
    count_in_window = results[2]
    return count_in_window <= limit


async def check_rate_limit(
    request: Request,
    api_key: str = Depends(require_api_key),
) -> str:
    """FastAPI dependency — rate limits by API key.

    Uses a robust Redis sliding window counter. Falls back to memory if cache not available.
    """
    now = time.monotonic()
    rate_limit_max = _get_rate_limit_max()
    
    # 1. Fetch the cache from the app state
    cache: Optional[BuildCache] = getattr(request.app.state, "cache", None)

    # 2. Redis Rate Limiting
    if cache and cache._available:
        # Convert monotonic to unix epoch for redis absolute scoring
        unix_now = time.time()
        is_allowed = await _redis_sliding_window(
            cache=cache, 
            api_key=api_key, 
            now=unix_now, 
            window=RATE_LIMIT_WINDOW, 
            limit=rate_limit_max
        )
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {rate_limit_max} requests per minute.",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )
        return api_key

    # 3. Fallback: Memory Rate Limiting
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limits[api_key] = [
        t for t in _rate_limits[api_key] if t > window_start
    ]

    if len(_rate_limits[api_key]) >= rate_limit_max:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rate_limit_max} requests per minute.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )

    _rate_limits[api_key].append(now)
    return api_key


def reset_rate_limits() -> None:
    """Reset all rate limit counters (for testing)."""
    _rate_limits.clear()


def reset_api_keys() -> None:
    """Reset cached API keys (for testing)."""
    # For testing, we just clear the environment variable cache
    if "ZENFA_API_KEYS" in os.environ:
        os.environ["ZENFA_API_KEYS"] = ""
