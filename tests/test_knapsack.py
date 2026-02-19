"""Tests for the Knapsack optimizer."""

import pytest

from zenfa_ai.engine.knapsack import (
    LLMSuggestion,
    SuggestionStatus,
    generate_build,
)
from zenfa_ai.models.build import BuildRequest
from zenfa_ai.models.components import (
    BuildPreferences,
    ComponentWithPrice,
    Purpose,
)


# ──────────────────────────────────────────────
# Sample BD Market Data (from AGENT_BRIEF.md)
# ──────────────────────────────────────────────

SAMPLE_COMPONENTS = [
    ComponentWithPrice(
        id=1, name="AMD Ryzen 5 7600", slug="amd-ryzen-5-7600",
        component_type="cpu", brand="AMD", performance_score=72,
        price_bdt=22500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "AM5", "core_count": 6, "thread_count": 12,
               "base_clock_ghz": 3.8, "boost_clock_ghz": 5.1, "tdp": 65,
               "integrated_graphics": True, "igpu_name": "Radeon Graphics"},
    ),
    ComponentWithPrice(
        id=2, name="Intel Core i5-13400F", slug="intel-core-i5-13400f",
        component_type="cpu", brand="Intel", performance_score=70,
        price_bdt=19500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "LGA1700", "core_count": 10, "thread_count": 16,
               "base_clock_ghz": 2.5, "boost_clock_ghz": 4.6, "tdp": 65,
               "integrated_graphics": False},
    ),
    ComponentWithPrice(
        id=3, name="MSI B650M MORTAR WIFI", slug="msi-b650m-mortar-wifi",
        component_type="motherboard", brand="MSI", performance_score=65,
        price_bdt=18500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "AM5", "form_factor": "mATX", "ram_type": "DDR5",
               "ram_slots": 4, "max_ram_gb": 128, "chipset": "B650",
               "pcie_x16_slots": 1, "m2_slots": 2},
    ),
    ComponentWithPrice(
        id=4, name="Gigabyte B550M DS3H", slug="gigabyte-b550m-ds3h",
        component_type="motherboard", brand="Gigabyte", performance_score=55,
        price_bdt=9200, vendor_name="StarTech", in_stock=True,
        specs={"socket": "AM4", "form_factor": "mATX", "ram_type": "DDR4",
               "ram_slots": 2, "max_ram_gb": 64, "chipset": "B550",
               "pcie_x16_slots": 1, "m2_slots": 1},
    ),
    ComponentWithPrice(
        id=13, name="MSI PRO B660M-A DDR4", slug="msi-pro-b660m-a-ddr4",
        component_type="motherboard", brand="MSI", performance_score=60,
        price_bdt=12500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "LGA1700", "form_factor": "mATX", "ram_type": "DDR4",
               "ram_slots": 2, "max_ram_gb": 64, "chipset": "B660",
               "pcie_x16_slots": 1, "m2_slots": 2},
    ),
    ComponentWithPrice(
        id=5, name="XFX Speedster SWFT 210 RX 7700 XT", slug="xfx-rx-7700-xt",
        component_type="gpu", brand="AMD", performance_score=82,
        price_bdt=58000, vendor_name="StarTech", in_stock=True,
        specs={"vram_gb": 12, "length_mm": 322, "recommended_psu_wattage": 600},
    ),
    ComponentWithPrice(
        id=6, name="MSI GeForce RTX 4060 VENTUS 2X", slug="msi-rtx-4060",
        component_type="gpu", brand="NVIDIA", performance_score=68,
        price_bdt=37000, vendor_name="StarTech", in_stock=True,
        specs={"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550},
    ),
    ComponentWithPrice(
        id=7, name="G.Skill Trident Z5 RGB 16GB DDR5 6000MHz",
        slug="gskill-trident-z5-ddr5",
        component_type="ram", brand="G.Skill", performance_score=75,
        price_bdt=5800, vendor_name="StarTech", in_stock=True,
        specs={"ram_type": "DDR5", "capacity_gb": 16, "speed_mhz": 6000,
               "modules": 1, "cas_latency": 36, "rgb": True},
    ),
    ComponentWithPrice(
        id=8, name="Corsair Vengeance LPX 16GB DDR4 3200MHz",
        slug="corsair-vengeance-lpx-ddr4",
        component_type="ram", brand="Corsair", performance_score=60,
        price_bdt=3600, vendor_name="StarTech", in_stock=True,
        specs={"ram_type": "DDR4", "capacity_gb": 16, "speed_mhz": 3200,
               "modules": 2, "cas_latency": 16, "rgb": False},
    ),
    ComponentWithPrice(
        id=9, name="Samsung 980 PRO 1TB NVMe", slug="samsung-980-pro-1tb",
        component_type="storage", brand="Samsung", performance_score=80,
        price_bdt=8500, vendor_name="StarTech", in_stock=True,
        specs={"storage_type": "NVMe", "capacity_gb": 1000},
    ),
    ComponentWithPrice(
        id=10, name="Corsair RM750e 750W 80+ Gold", slug="corsair-rm750e",
        component_type="psu", brand="Corsair", performance_score=70,
        price_bdt=9500, vendor_name="StarTech", in_stock=True,
        specs={"wattage": 750, "efficiency_rating": "80+ Gold", "modular": True},
    ),
    ComponentWithPrice(
        id=11, name="NZXT H5 Flow", slug="nzxt-h5-flow",
        component_type="case", brand="NZXT", performance_score=65,
        price_bdt=8500, vendor_name="StarTech", in_stock=True,
        specs={"max_gpu_length_mm": 365, "max_cpu_cooler_height_mm": 165,
               "psu_support": "ATX", "form_factor_support": ["ATX", "mATX", "ITX"]},
    ),
    ComponentWithPrice(
        id=12, name="DeepCool AK400", slug="deepcool-ak400",
        component_type="cooler", brand="DeepCool", performance_score=60,
        price_bdt=2800, vendor_name="StarTech", in_stock=True,
        specs={"cooler_type": "Air", "fan_size_mm": 120,
               "tdp_capacity_watts": 220, "socket_support": ["AM5", "AM4", "LGA1700"]},
    ),
]


def _make_request(
    budget_min: int = 80000,
    budget_max: int = 80000,
    purpose: str = "gaming",
    components: list | None = None,
    preferences: BuildPreferences | None = None,
    vendor_filter: str | None = None,
) -> BuildRequest:
    """Create a BuildRequest with defaults."""
    return BuildRequest(
        budget_min=budget_min,
        budget_max=budget_max,
        purpose=purpose,
        components=components or SAMPLE_COMPONENTS,
        preferences=preferences,
        vendor_filter=vendor_filter,
    )


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


class TestBuildGeneration:
    def test_generates_valid_build_under_budget(self):
        """Build total should not exceed budget_max."""
        request = _make_request(budget_max=80000)
        build, _ = generate_build(request)
        assert build.total_price <= 80000
        assert build.remaining_budget >= 0

    def test_includes_required_components(self):
        """Build should include CPU, motherboard, RAM, storage, PSU, case."""
        request = _make_request(budget_max=150000)
        build, _ = generate_build(request)
        types = {c.component_type for c in build.components}
        for required in ["cpu", "motherboard", "ram", "storage", "psu", "case"]:
            assert required in types, f"Missing required: {required}"

    def test_compatibility_verified(self):
        """Generated build should pass compatibility checks."""
        request = _make_request(budget_max=150000)
        build, _ = generate_build(request)
        assert build.compatibility_verified is True

    def test_gaming_prioritizes_gpu(self):
        """Gaming builds should include a GPU."""
        request = _make_request(purpose="gaming", budget_max=150000)
        build, _ = generate_build(request)
        types = {c.component_type for c in build.components}
        assert "gpu" in types


class TestBudgetRange:
    def test_fixed_budget(self):
        """When min == max, treated as fixed budget."""
        request = _make_request(budget_min=80000, budget_max=80000)
        assert request.is_fixed_budget is True
        build, _ = generate_build(request)
        assert build.total_price <= 80000

    def test_budget_range(self):
        """When min < max, optimizer targets within the range."""
        request = _make_request(budget_min=60000, budget_max=100000)
        assert request.is_fixed_budget is False
        build, _ = generate_build(request)
        assert build.total_price <= 100000


class TestBudgetValidation:
    def test_budget_min_greater_than_max_raises(self):
        """budget_min > budget_max should raise ValueError."""
        with pytest.raises(ValueError, match="budget_min"):
            _make_request(budget_min=100000, budget_max=50000)


class TestLLMSuggestions:
    def test_apply_swap_suggestion(self):
        """LLM suggestion to swap GPU should be applied if available."""
        request = _make_request(budget_max=150000)
        suggestion = LLMSuggestion(
            action="swap",
            component_category="gpu",
            current_component="RTX 4060",
            suggested_alternatives=["RX 7700 XT"],
            reason="better value for gaming",
        )
        build, results = generate_build(request, llm_suggestions=[suggestion])

        # Check the suggestion was applied
        applied = [r for r in results if r.status == SuggestionStatus.APPLIED]
        assert len(applied) >= 1

    def test_locked_component_not_swapped(self):
        """Locked components should not be swapped (oscillation prevention)."""
        request = _make_request(budget_max=150000)
        suggestion = LLMSuggestion(
            action="swap",
            component_category="gpu",
            suggested_alternatives=["RX 7700 XT"],
        )
        _, results = generate_build(
            request,
            llm_suggestions=[suggestion],
            locked_components={"gpu"},
        )
        incompatible = [r for r in results if r.status == SuggestionStatus.INCOMPATIBLE]
        assert len(incompatible) >= 1

    def test_unavailable_suggestion_reported(self):
        """Suggestion for a non-existent component should be reported."""
        request = _make_request(budget_max=150000)
        suggestion = LLMSuggestion(
            action="swap",
            component_category="gpu",
            suggested_alternatives=["RTX 5090 Nonexistent"],
        )
        build, results = generate_build(request, llm_suggestions=[suggestion])
        unavailable = [r for r in results if r.status == SuggestionStatus.UNAVAILABLE]
        assert len(unavailable) >= 1


class TestPreferences:
    def test_brand_preference_applied(self):
        """Brand preference should influence component selection."""
        prefs = BuildPreferences(prefer_brand="AMD")
        request = _make_request(budget_max=150000, preferences=prefs)
        build, _ = generate_build(request)

        # CPU should be AMD (Ryzen 5 7600) when preference is AMD
        cpu = next((c for c in build.components if c.component_type == "cpu"), None)
        assert cpu is not None
        assert "AMD" in cpu.name or "Ryzen" in cpu.name


class TestVendorFilter:
    def test_vendor_filter_applied(self):
        """Only components from the specified vendor should be used."""
        request = _make_request(budget_max=150000, vendor_filter="StarTech")
        build, _ = generate_build(request)
        for c in build.components:
            assert c.vendor_name == "StarTech"
