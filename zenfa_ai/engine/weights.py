"""Purpose-based component weight allocations.

Each purpose defines how much budget priority each component category gets.
Higher weight = more budget allocated = higher-tier component selected.
"""

from __future__ import annotations

from zenfa_ai.models.components import ComponentType, Purpose

# ──────────────────────────────────────────────
# Weight Profiles
# ──────────────────────────────────────────────

# Weights per component category for each purpose.
# Values should sum to ~1.0 for each purpose.
PURPOSE_WEIGHTS: dict[Purpose, dict[str, float]] = {
    Purpose.GAMING: {
        ComponentType.GPU: 0.40,
        ComponentType.CPU: 0.25,
        ComponentType.RAM: 0.15,
        ComponentType.STORAGE: 0.10,
        ComponentType.PSU: 0.05,
        ComponentType.MOTHERBOARD: 0.05,
    },
    Purpose.EDITING: {
        ComponentType.CPU: 0.35,
        ComponentType.RAM: 0.30,
        ComponentType.GPU: 0.20,
        ComponentType.STORAGE: 0.10,
        ComponentType.PSU: 0.03,
        ComponentType.MOTHERBOARD: 0.02,
    },
    Purpose.OFFICE: {
        ComponentType.CPU: 0.30,
        ComponentType.RAM: 0.25,
        ComponentType.STORAGE: 0.25,
        ComponentType.GPU: 0.10,
        ComponentType.PSU: 0.05,
        ComponentType.MOTHERBOARD: 0.05,
    },
    Purpose.GENERAL: {
        ComponentType.CPU: 0.25,
        ComponentType.GPU: 0.25,
        ComponentType.RAM: 0.20,
        ComponentType.STORAGE: 0.15,
        ComponentType.PSU: 0.08,
        ComponentType.MOTHERBOARD: 0.07,
    },
}

# Default weight for categories not listed (case, cooler, fan)
DEFAULT_WEIGHT: float = 0.02


def get_weight(purpose: Purpose, component_type: str) -> float:
    """Get the weight for a component type given the build purpose."""
    weights = PURPOSE_WEIGHTS.get(purpose, PURPOSE_WEIGHTS[Purpose.GENERAL])
    return weights.get(component_type, DEFAULT_WEIGHT)
