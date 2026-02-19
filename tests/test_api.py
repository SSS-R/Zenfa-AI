"""Tests for the dual-gateway API (internal + vendor).

Uses FastAPI's TestClient — no real LLM or server needed.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from zenfa_ai.api.app import create_app
from zenfa_ai.api.auth import reset_api_keys, reset_rate_limits


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


SAMPLE_COMPONENTS = [
    {
        "id": 1, "name": "AMD Ryzen 5 7600", "slug": "amd-ryzen-5-7600",
        "component_type": "cpu", "brand": "AMD", "performance_score": 72,
        "price_bdt": 22500, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"socket": "AM5", "tdp": 65, "integrated_graphics": True},
    },
    {
        "id": 3, "name": "MSI B650M MORTAR WIFI", "slug": "msi-b650m-mortar-wifi",
        "component_type": "motherboard", "brand": "MSI", "performance_score": 65,
        "price_bdt": 18500, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"socket": "AM5", "form_factor": "mATX", "ram_type": "DDR5",
                  "ram_slots": 4, "max_ram_gb": 128},
    },
    {
        "id": 7, "name": "G.Skill DDR5 16GB", "slug": "gskill-ddr5",
        "component_type": "ram", "brand": "G.Skill", "performance_score": 75,
        "price_bdt": 5800, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"ram_type": "DDR5", "capacity_gb": 16, "modules": 1},
    },
    {
        "id": 6, "name": "MSI RTX 4060", "slug": "msi-rtx-4060",
        "component_type": "gpu", "brand": "NVIDIA", "performance_score": 68,
        "price_bdt": 37000, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550},
    },
    {
        "id": 10, "name": "Corsair RM750e", "slug": "corsair-rm750e",
        "component_type": "psu", "brand": "Corsair", "performance_score": 70,
        "price_bdt": 9500, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"wattage": 750},
    },
    {
        "id": 9, "name": "Samsung 980 PRO 1TB", "slug": "samsung-980-pro",
        "component_type": "storage", "brand": "Samsung", "performance_score": 80,
        "price_bdt": 8500, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"storage_type": "NVMe", "capacity_gb": 1000},
    },
    {
        "id": 11, "name": "NZXT H5 Flow", "slug": "nzxt-h5-flow",
        "component_type": "case", "brand": "NZXT", "performance_score": 65,
        "price_bdt": 8500, "vendor_name": "StarTech", "in_stock": True,
        "specs": {"max_gpu_length_mm": 365,
                  "form_factor_support": ["ATX", "mATX", "ITX"]},
    },
]


BUILD_REQUEST = {
    "budget_min": 80000,
    "budget_max": 150000,
    "purpose": "gaming",
    "components": SAMPLE_COMPONENTS,
}


@pytest.fixture
def client():
    """Create a test client with no LLM (knapsack-only mode)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_auth_state():
    """Reset auth caches between tests."""
    reset_api_keys()
    reset_rate_limits()
    yield
    reset_api_keys()
    reset_rate_limits()


# ──────────────────────────────────────────────
# Root Tests
# ──────────────────────────────────────────────


class TestRoot:
    def test_root_returns_gateway_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "Zenfa AI"
        assert "internal" in data["gateways"]
        assert "vendor" in data["gateways"]


# ──────────────────────────────────────────────
# Internal Gateway Tests (PC Lagbe?)
# ──────────────────────────────────────────────


class TestInternalGateway:
    def test_health(self, client):
        resp = client.get("/internal/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["gateway"] == "internal"

    def test_build_no_auth_required(self, client):
        """Internal gateway should work without X-API-Key."""
        resp = client.post("/internal/build", json=BUILD_REQUEST)
        assert resp.status_code == 200
        data = resp.json()
        assert data["build"]["total_price"] > 0
        assert len(data["build"]["components"]) > 0

    def test_build_response_structure(self, client):
        """Response should have build, quality, explanation, metadata."""
        resp = client.post("/internal/build", json=BUILD_REQUEST)
        data = resp.json()
        assert "build" in data
        assert "quality" in data
        assert "explanation" in data
        assert "metadata" in data
        assert data["metadata"]["engine_version"] == "0.1.0"

    def test_compatibility_check(self, client):
        """Compatibility check should validate builds."""
        build = {
            "components": [
                {"id": 1, "name": "AMD Ryzen 5 7600", "component_type": "cpu",
                 "price_bdt": 22500, "vendor_name": "StarTech",
                 "specs": {"socket": "AM5", "tdp": 65}},
                {"id": 3, "name": "MSI B650M MORTAR WIFI",
                 "component_type": "motherboard", "price_bdt": 18500,
                 "vendor_name": "StarTech",
                 "specs": {"socket": "AM5", "form_factor": "mATX",
                          "ram_type": "DDR5"}},
            ],
            "total_price": 41000,
            "remaining_budget": 39000,
            "compatibility_verified": False,
        }
        resp = client.post("/internal/compatibility/check", json=build)
        assert resp.status_code == 200
        data = resp.json()
        assert data["compatible"] is True  # AM5 CPU with AM5 mobo = OK


# ──────────────────────────────────────────────
# Vendor Gateway Tests (B2B)
# ──────────────────────────────────────────────


class TestVendorGateway:
    def test_health_requires_auth(self, client):
        """Vendor health check requires API key."""
        resp = client.get("/v1/health")
        assert resp.status_code == 401

    @patch.dict(os.environ, {"ZENFA_API_KEYS": "test-key-123"})
    def test_health_with_valid_key(self, client):
        resp = client.get(
            "/v1/health",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["gateway"] == "vendor"

    @patch.dict(os.environ, {"ZENFA_API_KEYS": "test-key-123"})
    def test_invalid_key_rejected(self, client):
        resp = client.get(
            "/v1/health",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    @patch.dict(os.environ, {"ZENFA_API_KEYS": "test-key-123"})
    def test_build_with_auth(self, client):
        """Vendor build endpoint should work with valid API key."""
        resp = client.post(
            "/v1/build",
            json=BUILD_REQUEST,
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["build"]["total_price"] > 0

    def test_build_without_auth(self, client):
        """Vendor build endpoint should reject requests without API key."""
        resp = client.post("/v1/build", json=BUILD_REQUEST)
        assert resp.status_code == 401

    @patch.dict(os.environ, {"ZENFA_API_KEYS": "test-key-123",
                              "ZENFA_RATE_LIMIT_MAX": "3"})
    def test_rate_limiting(self, client):
        """Should rate limit after exceeding max requests."""
        headers = {"X-API-Key": "test-key-123"}

        # First 3 requests should succeed
        for _ in range(3):
            resp = client.get("/v1/health", headers=headers)
            assert resp.status_code == 200

        # 4th should be rate limited
        resp = client.get("/v1/health", headers=headers)
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]
