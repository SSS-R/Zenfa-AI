"""Knapsack optimizer — generates optimal PC builds within budget.

Uses a greedy approach with purpose-based weights and compatibility
filtering. Supports applying LLM suggestions with status tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from zenfa_ai.engine.compatibility import check_compatibility
from zenfa_ai.engine.weights import get_weight
from zenfa_ai.models.build import (
    BuildRequest,
    CandidateBuild,
    SelectedComponent,
)
from zenfa_ai.models.components import ComponentType, ComponentWithPrice, Purpose


# ──────────────────────────────────────────────
# Suggestion Application Types
# ──────────────────────────────────────────────


class SuggestionStatus(str, Enum):
    """Result status when applying an LLM suggestion."""

    APPLIED = "APPLIED"
    UNAVAILABLE = "UNAVAILABLE"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    OVER_BUDGET = "OVER_BUDGET"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass
class SuggestionResult:
    """Tracks the result of applying one LLM suggestion."""

    suggestion: str
    status: SuggestionStatus
    note: str = ""


@dataclass
class LLMSuggestion:
    """A single suggestion from the LLM evaluator."""

    action: str  # "swap", "upgrade", "downgrade", "remove", "add"
    component_category: str  # "gpu", "cpu", etc.
    current_component: Optional[str] = None
    suggested_alternatives: List[str] = field(default_factory=list)
    reason: str = ""
    priority: str = "medium"  # "high", "medium", "low"


# ──────────────────────────────────────────────
# Required component slots per build
# ──────────────────────────────────────────────

# Always required
REQUIRED_TYPES = ["cpu", "motherboard", "ram", "storage", "psu", "case"]

# Conditionally required (GPU needed unless iGPU + office)
CONDITIONAL_GPU_TYPES = {"office"}


def _needs_gpu(purpose: Purpose, components: List[ComponentWithPrice]) -> bool:
    """Determine if a dedicated GPU is needed.

    GPU can be skipped if purpose is 'office' AND the selected CPU
    has integrated graphics.
    """
    if purpose.value not in CONDITIONAL_GPU_TYPES:
        return True
    return True  # We'll check after CPU selection


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────


def _score_component(
    component: ComponentWithPrice,
    purpose: Purpose,
) -> float:
    """Score a component using its performance score weighted by purpose.

    Returns: weighted_score (higher = more desirable).
    """
    weight = get_weight(purpose, component.component_type)
    return component.performance_score * weight


def _value_score(component: ComponentWithPrice, purpose: Purpose) -> float:
    """Score a component by value = performance per taka.

    Higher score = better value for money.
    """
    if component.price_bdt <= 0:
        return 0.0
    perf = _score_component(component, purpose)
    return (perf / component.price_bdt) * 10000  # Normalize


# ──────────────────────────────────────────────
# Component Grouping & Filtering
# ──────────────────────────────────────────────


def _group_by_type(
    components: List[ComponentWithPrice],
) -> Dict[str, List[ComponentWithPrice]]:
    """Group components by their component_type (always plain string keys)."""
    groups: Dict[str, List[ComponentWithPrice]] = {}
    for c in components:
        ct = c.component_type.value if hasattr(c.component_type, "value") else str(c.component_type)
        groups.setdefault(ct, []).append(c)
    return groups


def _filter_in_stock(
    components: List[ComponentWithPrice],
) -> List[ComponentWithPrice]:
    """Keep only in-stock components."""
    return [c for c in components if c.in_stock]


def _filter_by_vendor(
    components: List[ComponentWithPrice],
    vendor: Optional[str],
) -> List[ComponentWithPrice]:
    """Filter components by vendor name if specified."""
    if vendor is None:
        return components
    return [c for c in components if c.vendor_name == vendor]


def _filter_by_preferences(
    components: List[ComponentWithPrice],
    preferences: Any,
    component_type: str,
) -> List[ComponentWithPrice]:
    """Apply user preferences as soft filters.

    If filtering produces 0 results, returns the original list (soft constraint).
    """
    if preferences is None:
        return components

    filtered = components

    # Brand preference
    if preferences.prefer_brand is not None:
        brand_matches = [
            c for c in filtered if c.brand == preferences.prefer_brand
        ]
        if brand_matches:
            filtered = brand_matches

    # Storage minimum
    if component_type == "storage" and preferences.min_storage_gb:
        storage_matches = [
            c for c in filtered
            if c.specs.get("capacity_gb", 0) >= preferences.min_storage_gb
        ]
        if storage_matches:
            filtered = storage_matches

    # RGB preference
    if preferences.prefer_rgb and component_type in ("ram", "case_fan"):
        rgb_matches = [c for c in filtered if c.specs.get("rgb", False)]
        if rgb_matches:
            filtered = rgb_matches

    return filtered


# ──────────────────────────────────────────────
# Compatibility-Aware Selection
# ──────────────────────────────────────────────


def _is_compatible_pair(
    selected: Dict[str, SelectedComponent],
    candidate: ComponentWithPrice,
    component_type: str,
) -> bool:
    """Quick pairwise compatibility check before adding to build.

    Checks the most critical rules to avoid assembling invalid builds.
    """
    specs = candidate.specs

    if component_type == "motherboard" and "cpu" in selected:
        cpu_socket = selected["cpu"].specs.get("socket")
        mobo_socket = specs.get("socket")
        if cpu_socket and mobo_socket and cpu_socket != mobo_socket:
            return False

    if component_type == "cpu" and "motherboard" in selected:
        mobo_socket = selected["motherboard"].specs.get("socket")
        cpu_socket = specs.get("socket")
        if mobo_socket and cpu_socket and mobo_socket != cpu_socket:
            return False

    if component_type == "ram" and "motherboard" in selected:
        mobo_ram = selected["motherboard"].specs.get("ram_type")
        ram_type = specs.get("ram_type")
        if mobo_ram and ram_type and mobo_ram != ram_type:
            return False

    if component_type == "psu" and "gpu" in selected:
        gpu_req = selected["gpu"].specs.get("recommended_psu_wattage")
        psu_watt = specs.get("wattage")
        if gpu_req and psu_watt and psu_watt < gpu_req:
            return False

    if component_type == "case" and "gpu" in selected:
        gpu_len = selected["gpu"].specs.get("length_mm")
        case_max = specs.get("max_gpu_length_mm")
        if gpu_len and case_max and case_max < gpu_len:
            return False

    if component_type == "case" and "motherboard" in selected:
        mobo_ff = selected["motherboard"].specs.get("form_factor")
        case_ff = specs.get("form_factor_support")
        if mobo_ff and case_ff and mobo_ff not in case_ff:
            return False

    if component_type == "cooler" and "cpu" in selected:
        cpu_socket = selected["cpu"].specs.get("socket")
        cooler_sockets = specs.get("socket_support")
        if cpu_socket and cooler_sockets and cpu_socket not in cooler_sockets:
            return False

    return True


def _to_selected(c: ComponentWithPrice) -> SelectedComponent:
    """Convert a ComponentWithPrice to a SelectedComponent."""
    return SelectedComponent(
        id=c.id,
        name=c.name,
        component_type=c.component_type if isinstance(c.component_type, str) else c.component_type.value,
        price_bdt=c.price_bdt,
        vendor_name=c.vendor_name,
        vendor_url=c.vendor_url,
        specs=c.specs,
    )


# ──────────────────────────────────────────────
# Main Build Generator
# ──────────────────────────────────────────────

# Selection order — dependencies first
SELECTION_ORDER = [
    "cpu",          # Select CPU first (determines socket/platform)
    "motherboard",  # Must match CPU socket
    "ram",          # Must match motherboard DDR type
    "gpu",          # GPU next (biggest budget item for gaming)
    "psu",          # Must meet GPU requirements
    "storage",      # Flexible
    "case",         # Must fit motherboard + GPU
    "cooler",       # Optional, must match CPU socket
]


def generate_build(
    request: BuildRequest,
    llm_suggestions: Optional[List[LLMSuggestion]] = None,
    locked_components: Optional[set[str]] = None,
) -> tuple[CandidateBuild, List[SuggestionResult]]:
    """Generate an optimized PC build within budget.

    Args:
        request: The build request with budget range, purpose, and component catalog.
        llm_suggestions: Optional suggestions from the LLM to apply.
        locked_components: Component categories that cannot be swapped (oscillation prevention).

    Returns:
        A tuple of (CandidateBuild, list of SuggestionResult).
    """
    budget_max = request.budget_max
    purpose = request.purpose
    preferences = request.preferences
    suggestion_results: List[SuggestionResult] = []

    # Filter and prepare components
    available = _filter_in_stock(request.components)
    available = _filter_by_vendor(available, request.vendor_filter)
    grouped = _group_by_type(available)

    # Track selected components and budget usage
    selected: Dict[str, SelectedComponent] = {}
    total_price = 0

    # Apply LLM suggestions (swap specific components)
    suggestion_map: Dict[str, List[str]] = {}  # category → preferred names
    if llm_suggestions:
        for sug in llm_suggestions:
            cat = sug.component_category
            if locked_components and cat in locked_components:
                suggestion_results.append(SuggestionResult(
                    suggestion=f"{sug.action} {cat}",
                    status=SuggestionStatus.INCOMPATIBLE,
                    note=f"Component '{cat}' is locked to prevent oscillation",
                ))
                continue
            if sug.suggested_alternatives:
                suggestion_map[cat] = sug.suggested_alternatives

    # Select components in dependency order
    for comp_type in SELECTION_ORDER:
        # Skip GPU for office builds with iGPU
        if comp_type == "gpu" and purpose == Purpose.OFFICE:
            # Check if selected CPU has integrated graphics
            if "cpu" in selected and selected["cpu"].specs.get("integrated_graphics"):
                continue

        # Skip cooler — optional, only add if budget allows after essentials
        if comp_type == "cooler":
            continue  # Handle cooler separately below

        candidates = grouped.get(comp_type, [])
        if not candidates:
            continue

        # Apply preference filtering
        candidates = _filter_by_preferences(candidates, preferences, comp_type)

        # If LLM suggested specific components for this category, try them first
        if comp_type in suggestion_map:
            suggested_names = suggestion_map[comp_type]
            suggested = [
                c for c in candidates
                if any(alt.lower() in c.name.lower() for alt in suggested_names)
            ]
            if suggested:
                # Try to use a suggested component
                for sc in sorted(suggested, key=lambda c: _value_score(c, purpose), reverse=True):
                    if sc.price_bdt + total_price <= budget_max and _is_compatible_pair(selected, sc, comp_type):
                        selected[comp_type] = _to_selected(sc)
                        total_price += sc.price_bdt
                        suggestion_results.append(SuggestionResult(
                            suggestion=f"swap {comp_type} to {sc.name}",
                            status=SuggestionStatus.APPLIED,
                            note=f"Available at {sc.vendor_name} for {sc.price_bdt:,}৳",
                        ))
                        break
                else:
                    # Matched by name but all failed budget/compatibility
                    for alt in suggested_names:
                        matching = [c for c in candidates if alt.lower() in c.name.lower()]
                        if matching:
                            can_afford = any(c.price_bdt + total_price <= budget_max for c in matching)
                            if not can_afford:
                                suggestion_results.append(SuggestionResult(
                                    suggestion=f"swap {comp_type} to {alt}",
                                    status=SuggestionStatus.OVER_BUDGET,
                                    note="Would exceed budget",
                                ))
                            else:
                                suggestion_results.append(SuggestionResult(
                                    suggestion=f"swap {comp_type} to {alt}",
                                    status=SuggestionStatus.INCOMPATIBLE,
                                    note="Fails compatibility check with current build",
                                ))
            else:
                # No candidates matched the suggested names at all
                for alt in suggested_names:
                    # Check full unfiltered catalog (might be out of stock)
                    full_match = any(
                        alt.lower() in c.name.lower()
                        for c in (grouped.get(comp_type, []))
                    )
                    if full_match:
                        suggestion_results.append(SuggestionResult(
                            suggestion=f"swap {comp_type} to {alt}",
                            status=SuggestionStatus.OUT_OF_STOCK,
                            note="Found but currently out of stock",
                        ))
                    else:
                        suggestion_results.append(SuggestionResult(
                            suggestion=f"swap {comp_type} to {alt}",
                            status=SuggestionStatus.UNAVAILABLE,
                            note="Not found in BD market",
                        ))
            if comp_type in selected:
                continue  # Already selected via suggestion

        # Default: pick best value component that fits budget and is compatible
        compatible_candidates = [
            c for c in candidates
            if c.price_bdt + total_price <= budget_max
            and _is_compatible_pair(selected, c, comp_type)
        ]

        if not compatible_candidates:
            continue

        # Sort by value score (performance per taka), pick the best
        best = max(compatible_candidates, key=lambda c: _value_score(c, purpose))
        selected[comp_type] = _to_selected(best)
        total_price += best.price_bdt

    # Optional: add cooler if budget allows and CPU TDP > 65W
    cooler_candidates = grouped.get("cooler", [])
    if cooler_candidates and "cpu" in selected:
        cpu_tdp = selected["cpu"].specs.get("tdp", 65)
        if cpu_tdp > 65 or (budget_max - total_price > 2000):
            cooler_candidates = _filter_by_preferences(
                cooler_candidates, preferences, "cooler"
            )
            compatible_coolers = [
                c for c in cooler_candidates
                if c.price_bdt + total_price <= budget_max
                and _is_compatible_pair(selected, c, "cooler")
            ]
            if compatible_coolers:
                best_cooler = max(
                    compatible_coolers,
                    key=lambda c: _value_score(c, purpose),
                )
                selected["cooler"] = _to_selected(best_cooler)
                total_price += best_cooler.price_bdt

    # Build result
    build = CandidateBuild(
        components=list(selected.values()),
        total_price=total_price,
        remaining_budget=budget_max - total_price,
        compatibility_verified=False,
    )

    # Final compatibility verification
    compat_result = check_compatibility(build)
    build.compatibility_verified = compat_result.passed

    return build, suggestion_results
