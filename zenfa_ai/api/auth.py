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

# In production, these come from a database or secrets manager
# For now, env var with comma-separated keys
_VALID_KEYS: Optional[set[str]] = None


def _load_valid_keys() -> set[str]:
    """Load valid API keys from environment."""
    global _VALID_KEYS
    if _VALID_KEYS is None:
        raw = os.getenv("ZENFA_API_KEYS", "")
        _VALID_KEYS = {k.strip() for k in raw.split(",") if k.strip()}
    return _VALID_KEYS


async def require_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """FastAPI dependency — validates X-API-Key header.

    Raises 401 if missing, 403 if invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
        )

    valid_keys = _load_valid_keys()
    if not valid_keys:
        # No keys configured — reject all vendor requests
        raise HTTPException(
            status_code=503,
            detail="Vendor API not configured. Contact admin.",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


# ──────────────────────────────────────────────
# Rate Limiting (in-memory, per API key)
# ──────────────────────────────────────────────

# Simple sliding window counter
_rate_limits: dict[str, list[float]] = defaultdict(list)

# Default: 60 requests per minute per key
RATE_LIMIT_WINDOW = 60  # seconds


def _get_rate_limit_max() -> int:
    """Read rate limit max at call time (allows test patching)."""
    return int(os.getenv("ZENFA_RATE_LIMIT_MAX", "60"))


async def check_rate_limit(
    request: Request,
    api_key: str = Depends(require_api_key),
) -> str:
    """FastAPI dependency — rate limits by API key.

    Uses a simple sliding window counter. In production, use Redis.
    """
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW
    rate_limit_max = _get_rate_limit_max()

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
    global _VALID_KEYS
    _VALID_KEYS = None
