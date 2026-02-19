"""Tests for the 9 hardware compatibility rules."""

import pytest

from zenfa_ai.engine.compatibility import (
    CompatibilityResult,
    check_case_gpu_length,
    check_compatibility,
    check_cooler_cpu_socket,
    check_cooler_cpu_tdp,
    check_cpu_motherboard_socket,
    check_motherboard_case_form_factor,
    check_psu_gpu_wattage,
    check_ram_motherboard_capacity,
    check_ram_motherboard_slots,
    check_ram_motherboard_type,
)
from zenfa_ai.models.build import CandidateBuild, SelectedComponent


# ──────────────────────────────────────────────
# Test Fixtures — Helper builders
# ──────────────────────────────────────────────


def _make(component_type: str, name: str = "Test", specs: dict | None = None, **kw):
    """Quick SelectedComponent factory."""
    return SelectedComponent(
        id=kw.get("id", 1),
        name=name,
        component_type=component_type,
        price_bdt=kw.get("price_bdt", 10000),
        vendor_name=kw.get("vendor_name", "StarTech"),
        specs=specs or {},
    )


# ──────────────────────────────────────────────
# Rule 1: CPU ↔ Motherboard socket
# ──────────────────────────────────────────────


class TestRule1CpuMotherboardSocket:
    def test_matching_socket_passes(self):
        cpu = _make("cpu", "Ryzen 5 7600", {"socket": "AM5"})
        mobo = _make("motherboard", "B650M", {"socket": "AM5"})
        assert check_cpu_motherboard_socket(cpu, mobo) is None

    def test_mismatched_socket_fails(self):
        cpu = _make("cpu", "Ryzen 5 7600", {"socket": "AM5"})
        mobo = _make("motherboard", "B560M", {"socket": "LGA1200"})
        v = check_cpu_motherboard_socket(cpu, mobo)
        assert v is not None
        assert v.rule == "cpu_motherboard_socket"

    def test_missing_socket_skipped(self):
        cpu = _make("cpu", "Unknown CPU", {})
        mobo = _make("motherboard", "Unknown Mobo", {"socket": "AM5"})
        assert check_cpu_motherboard_socket(cpu, mobo) is None


# ──────────────────────────────────────────────
# Rule 2: RAM ↔ Motherboard DDR type
# ──────────────────────────────────────────────


class TestRule2RamMotherboardType:
    def test_matching_ddr_passes(self):
        ram = _make("ram", "DDR5 Kit", {"ram_type": "DDR5"})
        mobo = _make("motherboard", "B650M", {"ram_type": "DDR5"})
        assert check_ram_motherboard_type(ram, mobo) is None

    def test_mismatched_ddr_fails(self):
        ram = _make("ram", "DDR4 Kit", {"ram_type": "DDR4"})
        mobo = _make("motherboard", "B650M", {"ram_type": "DDR5"})
        v = check_ram_motherboard_type(ram, mobo)
        assert v is not None
        assert "DDR4" in v.message


# ──────────────────────────────────────────────
# Rule 3: PSU wattage ≥ GPU requirement
# ──────────────────────────────────────────────


class TestRule3PsuGpuWattage:
    def test_sufficient_psu_passes(self):
        psu = _make("psu", "750W", {"wattage": 750})
        gpu = _make("gpu", "RTX 4060", {"recommended_psu_wattage": 550})
        assert check_psu_gpu_wattage(psu, gpu) is None

    def test_exact_wattage_passes(self):
        psu = _make("psu", "550W", {"wattage": 550})
        gpu = _make("gpu", "RTX 4060", {"recommended_psu_wattage": 550})
        assert check_psu_gpu_wattage(psu, gpu) is None

    def test_insufficient_psu_fails(self):
        psu = _make("psu", "450W", {"wattage": 450})
        gpu = _make("gpu", "RX 7700 XT", {"recommended_psu_wattage": 600})
        v = check_psu_gpu_wattage(psu, gpu)
        assert v is not None
        assert "450W" in v.message


# ──────────────────────────────────────────────
# Rule 4: Case GPU clearance
# ──────────────────────────────────────────────


class TestRule4CaseGpuLength:
    def test_gpu_fits_passes(self):
        case = _make("case", "H5 Flow", {"max_gpu_length_mm": 365})
        gpu = _make("gpu", "RTX 4060", {"length_mm": 240})
        assert check_case_gpu_length(case, gpu) is None

    def test_gpu_too_long_fails(self):
        case = _make("case", "Tiny Case", {"max_gpu_length_mm": 200})
        gpu = _make("gpu", "Big GPU", {"length_mm": 322})
        v = check_case_gpu_length(case, gpu)
        assert v is not None
        assert "200mm" in v.message


# ──────────────────────────────────────────────
# Rule 5: Motherboard form factor ↔ Case support
# ──────────────────────────────────────────────


class TestRule5MotherboardCaseFormFactor:
    def test_supported_form_factor_passes(self):
        mobo = _make("motherboard", "B650M", {"form_factor": "mATX"})
        case = _make("case", "H5 Flow", {"form_factor_support": ["ATX", "mATX", "ITX"]})
        assert check_motherboard_case_form_factor(mobo, case) is None

    def test_unsupported_form_factor_fails(self):
        mobo = _make("motherboard", "X670E", {"form_factor": "E-ATX"})
        case = _make("case", "Small Case", {"form_factor_support": ["mATX", "ITX"]})
        v = check_motherboard_case_form_factor(mobo, case)
        assert v is not None
        assert "E-ATX" in v.message


# ──────────────────────────────────────────────
# Rule 6: Cooler ↔ CPU socket
# ──────────────────────────────────────────────


class TestRule6CoolerCpuSocket:
    def test_supported_socket_passes(self):
        cooler = _make("cooler", "AK400", {"socket_support": ["AM5", "AM4", "LGA1700"]})
        cpu = _make("cpu", "Ryzen 5 7600", {"socket": "AM5"})
        assert check_cooler_cpu_socket(cooler, cpu) is None

    def test_unsupported_socket_fails(self):
        cooler = _make("cooler", "Old Cooler", {"socket_support": ["AM4", "LGA1200"]})
        cpu = _make("cpu", "Ryzen 5 7600", {"socket": "AM5"})
        v = check_cooler_cpu_socket(cooler, cpu)
        assert v is not None
        assert "AM5" in v.message


# ──────────────────────────────────────────────
# Rule 7: Cooler TDP ≥ CPU TDP
# ──────────────────────────────────────────────


class TestRule7CoolerCpuTdp:
    def test_sufficient_tdp_passes(self):
        cooler = _make("cooler", "AK400", {"tdp_capacity_watts": 220})
        cpu = _make("cpu", "Ryzen 5 7600", {"tdp": 65})
        assert check_cooler_cpu_tdp(cooler, cpu) is None

    def test_insufficient_tdp_fails(self):
        cooler = _make("cooler", "Tiny Cooler", {"tdp_capacity_watts": 50})
        cpu = _make("cpu", "i9-14900K", {"tdp": 253})
        v = check_cooler_cpu_tdp(cooler, cpu)
        assert v is not None
        assert "50W" in v.message

    def test_missing_tdp_skipped(self):
        cooler = _make("cooler", "Stock Cooler", {})
        cpu = _make("cpu", "Ryzen 5 7600", {"tdp": 65})
        assert check_cooler_cpu_tdp(cooler, cpu) is None


# ──────────────────────────────────────────────
# Rule 8: Total RAM ≤ Motherboard max
# ──────────────────────────────────────────────


class TestRule8RamMotherboardCapacity:
    def test_within_capacity_passes(self):
        ram = _make("ram", "16GB Kit", {"modules": 2, "capacity_gb": 16})
        mobo = _make("motherboard", "B650M", {"max_ram_gb": 128})
        assert check_ram_motherboard_capacity(ram, mobo) is None

    def test_exceeds_capacity_fails(self):
        ram = _make("ram", "64GB Kit", {"modules": 4, "capacity_gb": 32})
        mobo = _make("motherboard", "Budget Board", {"max_ram_gb": 64})
        v = check_ram_motherboard_capacity(ram, mobo)
        assert v is not None
        assert "128GB" in v.message  # 4 × 32 = 128


# ──────────────────────────────────────────────
# Rule 9: RAM modules ≤ Motherboard slots
# ──────────────────────────────────────────────


class TestRule9RamMotherboardSlots:
    def test_within_slots_passes(self):
        ram = _make("ram", "Dual Kit", {"modules": 2})
        mobo = _make("motherboard", "B650M", {"ram_slots": 4})
        assert check_ram_motherboard_slots(ram, mobo) is None

    def test_exceeds_slots_fails(self):
        ram = _make("ram", "Quad Kit", {"modules": 4})
        mobo = _make("motherboard", "Budget Board", {"ram_slots": 2})
        v = check_ram_motherboard_slots(ram, mobo)
        assert v is not None
        assert "4" in v.message


# ──────────────────────────────────────────────
# Full build compatibility check
# ──────────────────────────────────────────────


class TestFullCompatibilityCheck:
    def test_compatible_build_passes(self):
        """A fully compatible build should pass all rules."""
        build = CandidateBuild(
            components=[
                _make("cpu", "Ryzen 5 7600", {
                    "socket": "AM5", "tdp": 65
                }),
                _make("motherboard", "B650M MORTAR", {
                    "socket": "AM5", "form_factor": "mATX",
                    "ram_type": "DDR5", "ram_slots": 4, "max_ram_gb": 128,
                }),
                _make("ram", "DDR5 16GB", {
                    "ram_type": "DDR5", "capacity_gb": 16,
                    "modules": 1,
                }),
                _make("gpu", "RX 7700 XT", {
                    "vram_gb": 12, "length_mm": 322,
                    "recommended_psu_wattage": 600,
                }),
                _make("psu", "RM750e", {"wattage": 750}),
                _make("case", "H5 Flow", {
                    "max_gpu_length_mm": 365,
                    "form_factor_support": ["ATX", "mATX", "ITX"],
                }),
                _make("cooler", "AK400", {
                    "socket_support": ["AM5", "AM4", "LGA1700"],
                    "tdp_capacity_watts": 220,
                }),
            ],
            total_price=79200,
            remaining_budget=800,
        )
        result = check_compatibility(build)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_incompatible_build_detects_violations(self):
        """A build with socket + DDR mismatches should flag both."""
        build = CandidateBuild(
            components=[
                _make("cpu", "Ryzen 5 7600", {"socket": "AM5"}),
                _make("motherboard", "B560M", {
                    "socket": "LGA1200", "ram_type": "DDR4",
                    "ram_slots": 2, "max_ram_gb": 64,
                    "form_factor": "mATX",
                }),
                _make("ram", "DDR5 16GB", {
                    "ram_type": "DDR5", "capacity_gb": 16, "modules": 1,
                }),
                _make("psu", "500W", {"wattage": 500}),
                _make("gpu", "RTX 4060", {
                    "length_mm": 240, "recommended_psu_wattage": 550,
                }),
                _make("case", "H5 Flow", {
                    "max_gpu_length_mm": 365,
                    "form_factor_support": ["ATX", "mATX"],
                }),
            ],
            total_price=60000,
            remaining_budget=20000,
        )
        result = check_compatibility(build)
        assert result.passed is False
        rule_names = [v.rule for v in result.violations]
        assert "cpu_motherboard_socket" in rule_names  # AM5 ≠ LGA1200
        assert "ram_motherboard_type" in rule_names    # DDR5 ≠ DDR4
        assert "psu_gpu_wattage" in rule_names         # 500W < 550W

    def test_build_without_optional_components(self):
        """Build without cooler/GPU should still pass applicable rules."""
        build = CandidateBuild(
            components=[
                _make("cpu", "Ryzen 5 5600G", {
                    "socket": "AM4", "tdp": 65,
                    "integrated_graphics": True,
                }),
                _make("motherboard", "B550M", {
                    "socket": "AM4", "form_factor": "mATX",
                    "ram_type": "DDR4", "ram_slots": 2, "max_ram_gb": 64,
                }),
                _make("ram", "DDR4 16GB", {
                    "ram_type": "DDR4", "capacity_gb": 16, "modules": 2,
                }),
                _make("storage", "500GB NVMe", {
                    "storage_type": "NVMe", "capacity_gb": 500,
                }),
                _make("psu", "450W", {"wattage": 450}),
                _make("case", "Budget Case", {
                    "max_gpu_length_mm": 300,
                    "form_factor_support": ["mATX", "ITX"],
                }),
            ],
            total_price=35000,
            remaining_budget=5000,
        )
        result = check_compatibility(build)
        assert result.passed is True
