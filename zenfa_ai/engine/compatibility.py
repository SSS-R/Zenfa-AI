"""Hardware compatibility rules — 9 hard constraints for PC builds.

Every build MUST pass all rules. A violation is a boolean fail that
disqualifies the component combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zenfa_ai.models.build import CandidateBuild, SelectedComponent


# ──────────────────────────────────────────────
# Result Types
# ──────────────────────────────────────────────


@dataclass
class Violation:
    """A single compatibility violation."""

    rule: str
    message: str
    components_involved: List[str] = field(default_factory=list)


@dataclass
class CompatibilityResult:
    """Result of a full compatibility check."""

    passed: bool
    violations: List[Violation] = field(default_factory=list)


# ──────────────────────────────────────────────
# Helper — extract component by type
# ──────────────────────────────────────────────


def _get_component(
    components: List[SelectedComponent], component_type: str
) -> Optional[SelectedComponent]:
    """Return the first component matching the given type, or None."""
    for c in components:
        if c.component_type == component_type:
            return c
    return None


def _get_spec(component: SelectedComponent, key: str, default: Any = None) -> Any:
    """Safely get a spec value from a component."""
    return component.specs.get(key, default)


# ──────────────────────────────────────────────
# Individual Rule Checkers
# ──────────────────────────────────────────────


def check_cpu_motherboard_socket(
    cpu: SelectedComponent, motherboard: SelectedComponent
) -> Optional[Violation]:
    """RULE 1: CPU.socket == Motherboard.socket"""
    cpu_socket = _get_spec(cpu, "socket")
    mobo_socket = _get_spec(motherboard, "socket")

    if cpu_socket is None or mobo_socket is None:
        return None  # Can't check if data is missing

    if cpu_socket != mobo_socket:
        return Violation(
            rule="cpu_motherboard_socket",
            message=(
                f"CPU socket ({cpu_socket}) does not match "
                f"motherboard socket ({mobo_socket})"
            ),
            components_involved=[cpu.name, motherboard.name],
        )
    return None


def check_ram_motherboard_type(
    ram: SelectedComponent, motherboard: SelectedComponent
) -> Optional[Violation]:
    """RULE 2: RAM.ram_type == Motherboard.ram_type"""
    ram_type = _get_spec(ram, "ram_type")
    mobo_ram_type = _get_spec(motherboard, "ram_type")

    if ram_type is None or mobo_ram_type is None:
        return None

    if ram_type != mobo_ram_type:
        return Violation(
            rule="ram_motherboard_type",
            message=(
                f"RAM type ({ram_type}) does not match "
                f"motherboard RAM type ({mobo_ram_type})"
            ),
            components_involved=[ram.name, motherboard.name],
        )
    return None


def check_psu_gpu_wattage(
    psu: SelectedComponent, gpu: SelectedComponent
) -> Optional[Violation]:
    """RULE 3: PSU.wattage >= GPU.recommended_psu_wattage"""
    psu_wattage = _get_spec(psu, "wattage")
    gpu_psu_req = _get_spec(gpu, "recommended_psu_wattage")

    if psu_wattage is None or gpu_psu_req is None:
        return None

    if psu_wattage < gpu_psu_req:
        return Violation(
            rule="psu_gpu_wattage",
            message=(
                f"PSU wattage ({psu_wattage}W) is below "
                f"GPU requirement ({gpu_psu_req}W)"
            ),
            components_involved=[psu.name, gpu.name],
        )
    return None


def check_case_gpu_length(
    case: SelectedComponent, gpu: SelectedComponent
) -> Optional[Violation]:
    """RULE 4: Casing.max_gpu_length_mm >= GPU.length_mm"""
    case_max = _get_spec(case, "max_gpu_length_mm")
    gpu_len = _get_spec(gpu, "length_mm")

    if case_max is None or gpu_len is None:
        return None

    if case_max < gpu_len:
        return Violation(
            rule="case_gpu_length",
            message=(
                f"Case max GPU length ({case_max}mm) is too short "
                f"for GPU ({gpu_len}mm)"
            ),
            components_involved=[case.name, gpu.name],
        )
    return None


def check_motherboard_case_form_factor(
    motherboard: SelectedComponent, case: SelectedComponent
) -> Optional[Violation]:
    """RULE 5: Motherboard.form_factor IN Casing.form_factor_support"""
    mobo_ff = _get_spec(motherboard, "form_factor")
    case_ff_support = _get_spec(case, "form_factor_support")

    if mobo_ff is None or case_ff_support is None:
        return None

    if mobo_ff not in case_ff_support:
        return Violation(
            rule="motherboard_case_form_factor",
            message=(
                f"Motherboard form factor ({mobo_ff}) is not supported "
                f"by case (supports: {case_ff_support})"
            ),
            components_involved=[motherboard.name, case.name],
        )
    return None


def check_cooler_cpu_socket(
    cooler: SelectedComponent, cpu: SelectedComponent
) -> Optional[Violation]:
    """RULE 6: CPU.socket IN CPUCooler.socket_support"""
    cpu_socket = _get_spec(cpu, "socket")
    cooler_sockets = _get_spec(cooler, "socket_support")

    if cpu_socket is None or cooler_sockets is None:
        return None

    if cpu_socket not in cooler_sockets:
        return Violation(
            rule="cooler_cpu_socket",
            message=(
                f"CPU socket ({cpu_socket}) is not supported "
                f"by cooler (supports: {cooler_sockets})"
            ),
            components_involved=[cooler.name, cpu.name],
        )
    return None


def check_cooler_cpu_tdp(
    cooler: SelectedComponent, cpu: SelectedComponent
) -> Optional[Violation]:
    """RULE 7: CPUCooler.tdp_capacity_watts >= CPU.tdp (if both exist)"""
    cooler_tdp = _get_spec(cooler, "tdp_capacity_watts")
    cpu_tdp = _get_spec(cpu, "tdp")

    if cooler_tdp is None or cpu_tdp is None:
        return None

    if cooler_tdp < cpu_tdp:
        return Violation(
            rule="cooler_cpu_tdp",
            message=(
                f"Cooler TDP capacity ({cooler_tdp}W) is below "
                f"CPU TDP ({cpu_tdp}W)"
            ),
            components_involved=[cooler.name, cpu.name],
        )
    return None


def check_ram_motherboard_capacity(
    ram: SelectedComponent, motherboard: SelectedComponent
) -> Optional[Violation]:
    """RULE 8: RAM.modules * RAM.capacity_gb <= Motherboard.max_ram_gb"""
    modules = _get_spec(ram, "modules")
    capacity = _get_spec(ram, "capacity_gb")
    max_ram = _get_spec(motherboard, "max_ram_gb")

    if modules is None or capacity is None or max_ram is None:
        return None

    total_ram = modules * capacity
    if total_ram > max_ram:
        return Violation(
            rule="ram_motherboard_capacity",
            message=(
                f"Total RAM ({total_ram}GB = {modules}×{capacity}GB) exceeds "
                f"motherboard maximum ({max_ram}GB)"
            ),
            components_involved=[ram.name, motherboard.name],
        )
    return None


def check_ram_motherboard_slots(
    ram: SelectedComponent, motherboard: SelectedComponent
) -> Optional[Violation]:
    """RULE 9: RAM.modules <= Motherboard.ram_slots"""
    modules = _get_spec(ram, "modules")
    slots = _get_spec(motherboard, "ram_slots")

    if modules is None or slots is None:
        return None

    if modules > slots:
        return Violation(
            rule="ram_motherboard_slots",
            message=(
                f"RAM modules ({modules}) exceed "
                f"motherboard RAM slots ({slots})"
            ),
            components_involved=[ram.name, motherboard.name],
        )
    return None


# ──────────────────────────────────────────────
# Main Compatibility Check
# ──────────────────────────────────────────────


def check_compatibility(build: CandidateBuild) -> CompatibilityResult:
    """Run all 9 compatibility rules against a candidate build.

    Returns a CompatibilityResult with passed=True if all checks pass,
    or passed=False with a list of violations.
    """
    violations: List[Violation] = []
    components = build.components

    cpu = _get_component(components, "cpu")
    motherboard = _get_component(components, "motherboard")
    ram = _get_component(components, "ram")
    gpu = _get_component(components, "gpu")
    psu = _get_component(components, "psu")
    case = _get_component(components, "case")
    cooler = _get_component(components, "cooler")

    # Rule 1: CPU ↔ Motherboard socket
    if cpu and motherboard:
        v = check_cpu_motherboard_socket(cpu, motherboard)
        if v:
            violations.append(v)

    # Rule 2: RAM ↔ Motherboard DDR type
    if ram and motherboard:
        v = check_ram_motherboard_type(ram, motherboard)
        if v:
            violations.append(v)

    # Rule 3: PSU wattage ≥ GPU requirement
    if psu and gpu:
        v = check_psu_gpu_wattage(psu, gpu)
        if v:
            violations.append(v)

    # Rule 4: Case GPU clearance
    if case and gpu:
        v = check_case_gpu_length(case, gpu)
        if v:
            violations.append(v)

    # Rule 5: Motherboard form factor ↔ Case support
    if motherboard and case:
        v = check_motherboard_case_form_factor(motherboard, case)
        if v:
            violations.append(v)

    # Rule 6: Cooler ↔ CPU socket
    if cooler and cpu:
        v = check_cooler_cpu_socket(cooler, cpu)
        if v:
            violations.append(v)

    # Rule 7: Cooler TDP ≥ CPU TDP
    if cooler and cpu:
        v = check_cooler_cpu_tdp(cooler, cpu)
        if v:
            violations.append(v)

    # Rule 8: Total RAM ≤ Motherboard max
    if ram and motherboard:
        v = check_ram_motherboard_capacity(ram, motherboard)
        if v:
            violations.append(v)

    # Rule 9: RAM modules ≤ Motherboard slots
    if ram and motherboard:
        v = check_ram_motherboard_slots(ram, motherboard)
        if v:
            violations.append(v)

    return CompatibilityResult(
        passed=len(violations) == 0,
        violations=violations,
    )
