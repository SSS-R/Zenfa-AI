"""Loop state management for the agentic negotiation loop.

Tracks iteration history, timing, builds, scores, LLM state,
and oscillation detection across the Knapsack ⇄ LLM loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from zenfa_ai.evaluator.schemas import Suggestion
from zenfa_ai.engine.knapsack import SuggestionResult
from zenfa_ai.models.build import CandidateBuild


# ──────────────────────────────────────────────
# Default loop parameters (overridable via config)
# ──────────────────────────────────────────────

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_TIME_SECONDS = 120.0
DEFAULT_TARGET_SCORE = 8.5
DEFAULT_MINIMUM_SCORE = 7.0
DEFAULT_OPTIMIZATION_WINDOW = 90.0  # seconds

# Budget thresholds
LOW_BUDGET_THRESHOLD = 25000  # Skip LLM below this
HIGH_BUDGET_THRESHOLD = 500000  # Add diminishing returns guidance


@dataclass
class LoopState:
    """Mutable state for the agentic loop.

    Tracks everything needed by the decision gate and the orchestrator.
    """

    # Request context
    budget_min: int = 0
    budget_max: int = 0
    purpose: str = "general"

    # Timing
    start_time: float = field(default_factory=time.monotonic)
    max_time: float = DEFAULT_MAX_TIME_SECONDS

    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    # Build history (for "return best" decision)
    builds: List[CandidateBuild] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    # Current state
    current_build: Optional[CandidateBuild] = None
    current_score: float = 0.0
    best_build: Optional[CandidateBuild] = None
    best_score: float = 0.0

    # LLM state
    llm_suggestions: List[Suggestion] = field(default_factory=list)
    suggestion_history: List[SuggestionResult] = field(default_factory=list)
    unavailable_components: Set[str] = field(default_factory=set)
    llm_failed: bool = False
    llm_model_used: str = ""
    used_fallback: bool = False

    # Oscillation prevention
    locked_components: Set[str] = field(default_factory=set)
    component_swap_history: Dict[str, List[str]] = field(default_factory=dict)

    # ── Computed Properties ──

    @property
    def elapsed_time(self) -> float:
        """Seconds since loop start."""
        return time.monotonic() - self.start_time

    @property
    def score_improving(self) -> bool:
        """Whether the last score was better than the one before."""
        if len(self.scores) < 2:
            return True  # Assume improving on first iteration
        return self.scores[-1] > self.scores[-2]

    @property
    def is_low_budget(self) -> bool:
        """Budget too low for a meaningful build — skip LLM."""
        return self.budget_max < LOW_BUDGET_THRESHOLD

    @property
    def is_high_budget(self) -> bool:
        """Very high budget — add diminishing returns guidance."""
        return self.budget_max > HIGH_BUDGET_THRESHOLD

    # ── State Updates ──

    def record_build(
        self,
        build: CandidateBuild,
        score: float,
    ) -> None:
        """Record a build and its score, updating best if applicable."""
        self.current_build = build
        self.current_score = score
        self.builds.append(build)
        self.scores.append(score)
        self.iteration_count += 1

        if score > self.best_score or self.best_build is None:
            self.best_build = build
            self.best_score = score

    def record_suggestions(
        self,
        suggestions: List[Suggestion],
        results: List[SuggestionResult],
    ) -> None:
        """Record LLM suggestions and their application results."""
        self.llm_suggestions = suggestions
        self.suggestion_history = results

        # Track unavailable components to avoid re-suggesting
        for r in results:
            status_str = r.status.value if hasattr(r.status, "value") else str(r.status)
            if status_str in ("unavailable", "out_of_stock"):
                # Extract component name from suggestion text
                self.unavailable_components.add(r.suggestion)

    def detect_oscillation(
        self,
        suggestions: List[Suggestion],
    ) -> Set[str]:
        """Detect A→B→A swap oscillations and lock affected categories.

        Returns newly locked categories.
        """
        newly_locked: Set[str] = set()

        for s in suggestions:
            cat = s.component_category
            if cat in self.locked_components:
                continue  # Already locked

            # Track swap history for this category
            if cat not in self.component_swap_history:
                self.component_swap_history[cat] = []

            # Check if suggested alternative was a "current" in a previous iteration
            for alt in s.suggested_alternatives:
                if alt in self.component_swap_history.get(cat, []):
                    # Oscillation detected: we swapped FROM this component before
                    self.locked_components.add(cat)
                    newly_locked.add(cat)
                    break

            # Record current component as swapped-from
            if s.current_component:
                self.component_swap_history.setdefault(cat, []).append(
                    s.current_component
                )

        return newly_locked
