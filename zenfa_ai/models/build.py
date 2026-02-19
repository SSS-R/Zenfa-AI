"""Build request, response, and intermediate models for Zenfa AI Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from zenfa_ai.models.components import (
    BuildPreferences,
    ComponentType,
    ComponentWithPrice,
    Purpose,
)


# ──────────────────────────────────────────────
# Build Request
# ──────────────────────────────────────────────


class BuildRequest(BaseModel):
    """Input contract — sent by project-zenfa to zenfa-ai.

    Budget is a range. When budget_min == budget_max the optimizer
    treats it as a single fixed budget.
    """

    budget_min: int = Field(ge=0, description="Minimum budget in BDT (৳)")
    budget_max: int = Field(ge=0, description="Maximum budget in BDT (৳)")
    purpose: Purpose

    # Component catalog — the entire available market
    components: List[ComponentWithPrice]

    # Optional vendor filtering (for API customers)
    vendor_filter: Optional[str] = None

    # Optional user preferences (all predefined selectable options)
    preferences: Optional[BuildPreferences] = None

    @model_validator(mode="after")
    def _validate_budget_range(self) -> "BuildRequest":
        if self.budget_min > self.budget_max:
            raise ValueError(
                f"budget_min ({self.budget_min}) cannot exceed "
                f"budget_max ({self.budget_max})"
            )
        return self

    @property
    def budget(self) -> int:
        """Effective budget — max of the range (used by the optimizer)."""
        return self.budget_max

    @property
    def is_fixed_budget(self) -> bool:
        """True when the user specified a single exact budget."""
        return self.budget_min == self.budget_max


# ──────────────────────────────────────────────
# Internal / Intermediate Models
# ──────────────────────────────────────────────


class SelectedComponent(BaseModel):
    """A component chosen for the final build."""

    id: int
    name: str
    component_type: str  # Plain string (e.g., "cpu", "gpu") for easy comparison
    price_bdt: int
    vendor_name: str
    vendor_url: str = ""
    specs: Dict[str, Any] = Field(default_factory=dict)


class CandidateBuild(BaseModel):
    """In-progress build produced by the Knapsack engine.

    Used internally during the agentic loop — not returned to callers.
    """

    components: List[SelectedComponent] = Field(default_factory=list)
    total_price: int = 0
    remaining_budget: int = 0
    compatibility_verified: bool = False


# ──────────────────────────────────────────────
# Build Response
# ──────────────────────────────────────────────


class FinalBuild(BaseModel):
    """The final component list with pricing summary."""

    components: List[SelectedComponent]
    total_price: int
    remaining_budget: int


class BuildQuality(BaseModel):
    """Quality metrics for the generated build."""

    score: float = Field(ge=0.0, le=10.0)
    scores_breakdown: Dict[str, int] = Field(default_factory=dict)
    iterations_used: int = 1
    time_taken_seconds: float = 0.0


class BuildExplanation(BaseModel):
    """Human-readable explanation of the build decisions."""

    summary: str = ""
    per_component: Dict[str, str] = Field(default_factory=dict)
    trade_offs: str = ""
    upgrade_path: str = ""


class BuildMetadata(BaseModel):
    """Engine metadata attached to every response."""

    engine_version: str = "0.1.0"
    llm_model: str = "knapsack-only"
    fallback_used: bool = False
    cached: bool = False


class BuildResponse(BaseModel):
    """Top-level response returned by the /build endpoint."""

    build: FinalBuild
    quality: BuildQuality
    explanation: BuildExplanation
    metadata: BuildMetadata
