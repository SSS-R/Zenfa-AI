"""Pydantic schemas for LLM evaluator I/O.

Defines the structured response format that the LLM must return,
and the scoring rubric contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Suggestion(BaseModel):
    """A single improvement suggestion from the LLM."""

    action: str = Field(
        description="Action type: 'swap', 'upgrade', 'downgrade', 'remove', 'add'"
    )
    component_category: str = Field(
        description="Component category: 'cpu', 'gpu', 'ram', etc."
    )
    current_component: Optional[str] = Field(
        default=None,
        description="Name of the current component to replace",
    )
    suggested_alternatives: List[str] = Field(
        default_factory=list,
        description="Ordered list of suggested alternatives (best first)",
    )
    reason: str = Field(default="", description="Why this change improves the build")
    priority: str = Field(
        default="medium",
        description="Priority: 'high', 'medium', 'low'",
    )


class ScoreBreakdown(BaseModel):
    """The 5-category scoring rubric."""

    performance_match: int = Field(
        ge=0, le=3,
        description="Does the build match the purpose? (0-3 pts)",
    )
    value_score: int = Field(
        ge=0, le=3,
        description="Best performance-per-taka for each component? (0-3 pts)",
    )
    build_balance: int = Field(
        ge=0, le=2,
        description="Is the build balanced? No bottlenecks? (0-2 pts)",
    )
    future_proofing: int = Field(
        ge=0, le=1,
        description="Will this stay relevant for 2-3 years? (0-1 pt)",
    )
    community_trust: int = Field(
        ge=0, le=1,
        description="Well-reviewed, no known issues? (0-1 pt)",
    )

    @property
    def total(self) -> float:
        """Sum of all rubric scores (0-10)."""
        return (
            self.performance_match
            + self.value_score
            + self.build_balance
            + self.future_proofing
            + self.community_trust
        )


class EvaluationResponse(BaseModel):
    """Structured response expected from the LLM evaluator.

    The LLM must return JSON matching this schema exactly.
    """

    scores: ScoreBreakdown = Field(
        description="Detailed scoring per rubric category"
    )
    final_score: float = Field(
        ge=0.0, le=10.0,
        description="Overall build score (0-10)",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of the score",
    )
    suggestions: List[Suggestion] = Field(
        default_factory=list,
        description="Improvement suggestions (empty if score >= 8.5)",
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Known issues with any component (thermal, driver, DOA rate)",
    )
    approved: bool = Field(
        description="True if score >= 8.5 and no critical red flags",
    )


class ExplanationResponse(BaseModel):
    """Structured response for the post-loop explanation generation."""

    summary: str = Field(
        description="2-3 sentence overview of the build and why it's a good choice"
    )
    per_component: Dict[str, str] = Field(
        default_factory=dict,
        description="Component category → 1-sentence justification",
    )
    trade_offs: str = Field(
        default="",
        description="What was sacrificed to stay within budget",
    )
    upgrade_path: str = Field(
        default="",
        description="Suggested next upgrade when budget allows",
    )
