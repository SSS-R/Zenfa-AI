"""Tests for the Redis caching layer.

Uses InMemoryCache — no Redis server needed.
"""

from __future__ import annotations

import pytest

from zenfa_ai.cache.redis_cache import (
    BuildCache,
    InMemoryCache,
    build_cache_key,
    request_to_cache_key,
)
from zenfa_ai.models.build import BuildRequest
from zenfa_ai.models.components import ComponentWithPrice


# ──────────────────────────────────────────────
# Cache Key Tests
# ──────────────────────────────────────────────


class TestCacheKey:
    def test_same_inputs_same_key(self):
        """Identical inputs should produce identical cache keys."""
        key1 = build_cache_key("gaming", 80000, 100000, [1, 2, 3])
        key2 = build_cache_key("gaming", 80000, 100000, [1, 2, 3])
        assert key1 == key2

    def test_different_purpose_different_key(self):
        key1 = build_cache_key("gaming", 80000, 100000, [1, 2, 3])
        key2 = build_cache_key("office", 80000, 100000, [1, 2, 3])
        assert key1 != key2

    def test_different_budget_different_key(self):
        key1 = build_cache_key("gaming", 80000, 100000, [1, 2, 3])
        key2 = build_cache_key("gaming", 80000, 120000, [1, 2, 3])
        assert key1 != key2

    def test_component_order_irrelevant(self):
        """Component IDs should be sorted — order shouldn't matter."""
        key1 = build_cache_key("gaming", 80000, 100000, [1, 3, 2])
        key2 = build_cache_key("gaming", 80000, 100000, [3, 1, 2])
        assert key1 == key2

    def test_key_has_prefix(self):
        key = build_cache_key("gaming", 80000, 100000, [1])
        assert key.startswith("zenfa:build:")

    def test_request_to_cache_key(self):
        """Should extract key from a BuildRequest object."""
        components = [
            ComponentWithPrice(
                id=1, name="AMD Ryzen 5 7600", slug="amd-ryzen-5-7600",
                component_type="cpu", brand="AMD", performance_score=72,
                price_bdt=22500, vendor_name="StarTech", in_stock=True,
                specs={"socket": "AM5"},
            ),
        ]
        request = BuildRequest(
            budget_min=80000, budget_max=100000,
            purpose="gaming", components=components,
        )
        key = request_to_cache_key(request)
        assert key.startswith("zenfa:build:")

        # Same request should produce same key
        key2 = request_to_cache_key(request)
        assert key == key2


# ──────────────────────────────────────────────
# InMemoryCache Tests
# ──────────────────────────────────────────────


class TestInMemoryCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = InMemoryCache()
        await cache.connect()

        await cache.set("test:key", '{"result": "ok"}')
        val = await cache.get("test:key")
        assert val == '{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_get_miss(self):
        cache = InMemoryCache()
        await cache.connect()

        val = await cache.get("nonexistent")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete(self):
        cache = InMemoryCache()
        await cache.connect()

        await cache.set("test:key", "value")
        await cache.delete("test:key")
        val = await cache.get("test:key")
        assert val is None

    @pytest.mark.asyncio
    async def test_clear_all(self):
        cache = InMemoryCache()
        await cache.connect()

        await cache.set("zenfa:build:a", "1")
        await cache.set("zenfa:build:b", "2")
        count = await cache.clear_all()
        assert count == 2

        val = await cache.get("zenfa:build:a")
        assert val is None

    @pytest.mark.asyncio
    async def test_stats(self):
        cache = InMemoryCache()
        await cache.connect()

        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        stats = await cache.stats()
        assert stats["available"] is True
        assert stats["keys"] == 2

    @pytest.mark.asyncio
    async def test_disconnect(self):
        cache = InMemoryCache()
        await cache.connect()
        assert cache.available is True

        await cache.disconnect()
        assert cache.available is False


# ──────────────────────────────────────────────
# BuildCache Graceful Degradation
# ──────────────────────────────────────────────


class TestBuildCacheDegradation:
    @pytest.mark.asyncio
    async def test_operations_when_unavailable(self):
        """All operations should return None/False when not connected."""
        cache = BuildCache(redis_url="redis://nonexistent:9999")
        # Don't connect — simulate Redis being down

        val = await cache.get("test")
        assert val is None

        result = await cache.set("test", "value")
        assert result is False

        result = await cache.delete("test")
        assert result is False

        count = await cache.clear_all()
        assert count == 0

    @pytest.mark.asyncio
    async def test_available_property(self):
        cache = BuildCache()
        assert cache.available is False  # Not connected yet
