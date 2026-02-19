"""Decision gate for the agentic negotiation loop.

Implements the exit/continue logic from the Agentic Loop spec.
"""

from __future__ import annotations

from enum import Enum

from zenfa_ai.orchestrator.state import (
    DEFAULT_MINIMUM_SCORE,
    DEFAULT_OPTIMIZATION_WINDOW,
    DEFAULT_TARGET_SCORE,
    LoopState,
)


class Decision(str, Enum):
    """What the loop should do next."""

    RETURN_CURRENT = "return_current"
    """Return the current build as final."""

    RETURN_BEST = "return_best"
    """Return the highest-scored build from any iteration."""

    RETURN_KNAPSACK_ONLY = "return_knapsack_only"
    """Return the Knapsack build without LLM evaluation."""

    CONTINUE = "continue"
    """Run another Knapsack → LLM iteration."""

    CONTINUE_OPTIMIZING = "continue_optimizing"
    """Score is good but time remains — try for even better."""


def should_continue(
    state: LoopState,
    target_score: float = DEFAULT_TARGET_SCORE,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    optimization_window: float = DEFAULT_OPTIMIZATION_WINDOW,
) -> Decision:
    """The decision gate — determines whether to loop, exit, or fallback.

    Decision tree (evaluated top to bottom):
    1. Time exceeded → RETURN_BEST
    2. Max iterations → RETURN_BEST
    3. LLM failed → RETURN_KNAPSACK_ONLY
    4. Score >= target AND within optimization window → CONTINUE_OPTIMIZING
    5. Score >= target AND past window → RETURN_CURRENT
    6. Score >= minimum AND still improving → CONTINUE
    7. Score >= minimum AND plateaued → RETURN_CURRENT
    8. Score < minimum → CONTINUE (keep trying)
    """
    # Hard exits
    if state.elapsed_time > state.max_time:
        return Decision.RETURN_BEST

    if state.iteration_count >= state.max_iterations:
        return Decision.RETURN_BEST

    if state.llm_failed:
        return Decision.RETURN_KNAPSACK_ONLY

    # Score-based decisions
    if state.current_score >= target_score:
        if state.elapsed_time < optimization_window:
            return Decision.CONTINUE_OPTIMIZING
        return Decision.RETURN_CURRENT

    if state.current_score >= minimum_score:
        if state.score_improving:
            return Decision.CONTINUE
        return Decision.RETURN_CURRENT

    # Score too low — keep trying
    return Decision.CONTINUE
