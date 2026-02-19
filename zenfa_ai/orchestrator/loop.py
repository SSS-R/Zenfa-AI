"""Agentic negotiation loop orchestrator.

Wires together:
  Phase 1 (Knapsack) → Phase 2 (LLM Evaluate) → Phase 3 (Decision Gate)

Handles oscillation prevention, graceful degradation, budget edge cases,
and post-loop explanation generation.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from zenfa_ai.engine.knapsack import (
    LLMSuggestion,
    SuggestionResult,
    generate_build,
)
from zenfa_ai.evaluator.llm_client import BaseLLMClient, LLMError
from zenfa_ai.evaluator.prompts import (
    EXPLANATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_user_prompt,
)
from zenfa_ai.evaluator.schemas import EvaluationResponse, ExplanationResponse, Suggestion
from zenfa_ai.models.build import (
    BuildExplanation,
    BuildMetadata,
    BuildQuality,
    BuildRequest,
    BuildResponse,
    CandidateBuild,
    FinalBuild,
)
from zenfa_ai.orchestrator.decisions import Decision, should_continue
from zenfa_ai.orchestrator.state import (
    LOW_BUDGET_THRESHOLD,
    LoopState,
)

logger = logging.getLogger(__name__)


async def run_build_loop(
    request: BuildRequest,
    llm_client: Optional[BaseLLMClient] = None,
    max_iterations: int = 5,
    max_time: float = 120.0,
    target_score: float = 8.5,
    minimum_score: float = 7.0,
    optimization_window: float = 90.0,
) -> BuildResponse:
    """Run the full agentic build optimization loop.

    Flow:
    1. Generate initial build via Knapsack
    2. Evaluate with LLM → get score + suggestions
    3. Decision gate → continue or return
    4. Apply suggestions via Knapsack → loop back to step 2
    5. Post-loop: generate explanation

    Args:
        request: Build request with budget, purpose, components.
        llm_client: LLM client (None = knapsack-only mode).
        max_iterations: Maximum loop iterations.
        max_time: Maximum wall-clock time in seconds.
        target_score: Score threshold for approval.
        minimum_score: Minimum acceptable score.
        optimization_window: Time window for optimization attempts.

    Returns:
        Complete BuildResponse with build, quality, explanation, metadata.
    """
    state = LoopState(
        budget_min=request.budget_min,
        budget_max=request.budget_max,
        purpose=request.purpose if isinstance(request.purpose, str) else request.purpose.value,
        max_iterations=max_iterations,
        max_time=max_time,
    )

    # ── Budget edge case: too low ──
    if state.is_low_budget or llm_client is None:
        return await _knapsack_only_build(request, state, llm_client is None)

    # ── Main loop ──
    llm_suggestions: List[LLMSuggestion] = []
    suggestion_results: List[SuggestionResult] = []
    last_evaluation: Optional[EvaluationResponse] = None

    while True:
        # ── Phase 1: Knapsack Generation ──
        logger.info(
            "Loop iteration %d — generating build", state.iteration_count + 1
        )

        build, knapsack_results = generate_build(
            request,
            llm_suggestions=llm_suggestions if llm_suggestions else None,
            locked_components=state.locked_components if state.locked_components else None,
        )
        suggestion_results = knapsack_results

        # ── Phase 2: LLM Evaluation ──
        try:
            # Build available components context for the LLM
            available = _get_available_components(request, state)

            user_prompt = build_user_prompt(
                build=build,
                purpose=state.purpose,
                budget_min=state.budget_min,
                budget_max=state.budget_max,
                iteration=state.iteration_count + 1,
                suggestion_results=suggestion_results if state.iteration_count > 0 else None,
                available_categories=available,
            )

            evaluation = await llm_client.evaluate(SYSTEM_PROMPT, user_prompt)
            last_evaluation = evaluation

            # Record in state
            state.record_build(build, evaluation.final_score)
            state.llm_model_used = llm_client.model
            if hasattr(llm_client, "used_fallback"):
                state.used_fallback = llm_client.used_fallback

            logger.info(
                "Iteration %d score: %.1f (best: %.1f)",
                state.iteration_count,
                evaluation.final_score,
                state.best_score,
            )

        except LLMError as e:
            logger.error("LLM evaluation failed: %s", e)
            state.llm_failed = True
            # Record the build even without LLM score
            state.record_build(build, 0.0)

        # ── Phase 3: Decision Gate ──
        decision = should_continue(
            state,
            target_score=target_score,
            minimum_score=minimum_score,
            optimization_window=optimization_window,
        )

        logger.info("Decision: %s", decision.value)

        if decision in (
            Decision.RETURN_CURRENT,
            Decision.RETURN_BEST,
            Decision.RETURN_KNAPSACK_ONLY,
        ):
            break

        # ── Prepare next iteration ──
        if last_evaluation and last_evaluation.suggestions:
            # Detect oscillation before applying suggestions
            newly_locked = state.detect_oscillation(last_evaluation.suggestions)
            if newly_locked:
                logger.info("Oscillation detected — locked: %s", newly_locked)

            # Convert LLM suggestions to knapsack format
            llm_suggestions = _convert_suggestions(
                last_evaluation.suggestions,
                state.locked_components,
            )

            # Record suggestions
            state.record_suggestions(
                last_evaluation.suggestions,
                suggestion_results,
            )
        else:
            llm_suggestions = []

    # ── Select final build ──
    if decision == Decision.RETURN_BEST and state.best_build:
        final_build = state.best_build
        final_score = state.best_score
    elif state.current_build:
        final_build = state.current_build
        final_score = state.current_score
    else:
        # Shouldn't happen, but fallback
        final_build = build
        final_score = 0.0

    # ── Post-loop: Generate explanation ──
    explanation = await _generate_explanation(
        llm_client, final_build, state, final_score
    )

    # ── Assemble response ──
    return _build_response(
        final_build, final_score, last_evaluation, explanation, state
    )


async def _knapsack_only_build(
    request: BuildRequest,
    state: LoopState,
    no_llm: bool,
) -> BuildResponse:
    """Generate a knapsack-only build without LLM evaluation."""
    build, _ = generate_build(request)
    state.record_build(build, 0.0)

    reason = "No LLM configured" if no_llm else f"Budget below {LOW_BUDGET_THRESHOLD:,}৳"
    logger.info("Knapsack-only build: %s", reason)

    explanation = BuildExplanation(
        summary=f"Knapsack-optimized build ({reason}). "
                "No LLM evaluation was performed.",
        per_component={},
        trade_offs="LLM evaluation skipped — build is optimized for best "
                   "value within budget constraints only.",
        upgrade_path="",
    )

    return _build_response(
        build=build,
        final_score=0.0,
        evaluation=None,
        explanation=explanation,
        state=state,
    )


async def _generate_explanation(
    llm_client: BaseLLMClient,
    build: CandidateBuild,
    state: LoopState,
    final_score: float,
) -> BuildExplanation:
    """Generate a customer-friendly explanation via LLM."""
    try:
        prompt = build_explanation_prompt(
            build=build,
            purpose=state.purpose,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            final_score=final_score,
        )

        resp = await llm_client.generate_explanation(
            EXPLANATION_SYSTEM_PROMPT, prompt
        )

        return BuildExplanation(
            summary=resp.summary,
            per_component=resp.per_component,
            trade_offs=resp.trade_offs,
            upgrade_path=resp.upgrade_path,
        )

    except LLMError as e:
        logger.warning("Explanation generation failed: %s", e)
        return BuildExplanation(
            summary="Build optimized for best performance within budget.",
            per_component={},
            trade_offs="",
            upgrade_path="",
        )


def _convert_suggestions(
    suggestions: List[Suggestion],
    locked: set[str],
) -> List[LLMSuggestion]:
    """Convert LLM Suggestions to the knapsack's LLMSuggestion format."""
    result = []
    for s in suggestions:
        if s.component_category in locked:
            continue  # Skip locked categories
        result.append(
            LLMSuggestion(
                action=s.action,
                component_category=s.component_category,
                current_component=s.current_component,
                suggested_alternatives=s.suggested_alternatives,
                reason=s.reason,
            )
        )
    return result


def _get_available_components(
    request: BuildRequest,
    state: LoopState,
) -> dict[str, List[str]]:
    """Get available in-stock component names grouped by category."""
    available: dict[str, List[str]] = {}
    for c in request.components:
        if not c.in_stock:
            continue
        ct = c.component_type.value if hasattr(c.component_type, "value") else str(c.component_type)
        available.setdefault(ct, []).append(c.name)
    return available


def _build_response(
    build: CandidateBuild,
    final_score: float,
    evaluation: Optional[EvaluationResponse],
    explanation: BuildExplanation,
    state: LoopState,
) -> BuildResponse:
    """Assemble the final BuildResponse."""
    # Build quality
    scores_breakdown = {}
    if evaluation:
        scores_breakdown = {
            "performance_match": evaluation.scores.performance_match,
            "value_score": evaluation.scores.value_score,
            "build_balance": evaluation.scores.build_balance,
            "future_proofing": evaluation.scores.future_proofing,
            "community_trust": evaluation.scores.community_trust,
        }

    quality = BuildQuality(
        score=final_score,
        scores_breakdown=scores_breakdown,
        iterations_used=state.iteration_count,
        time_taken_seconds=round(state.elapsed_time, 2),
    )

    # Final build
    final = FinalBuild(
        components=build.components,
        total_price=build.total_price,
        remaining_budget=build.remaining_budget,
    )

    # Metadata
    metadata = BuildMetadata(
        engine_version="0.1.0",
        llm_model=state.llm_model_used or "knapsack-only",
        fallback_used=state.used_fallback,
        cached=False,
    )

    return BuildResponse(
        build=final,
        quality=quality,
        explanation=explanation,
        metadata=metadata,
    )
