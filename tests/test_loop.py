"""Tests for the agentic loop orchestrator.

All LLM calls are mocked — tests verify the loop logic,
decision gate, oscillation prevention, and fallback behavior.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zenfa_ai.evaluator.llm_client import (
    BaseLLMClient,
    GeminiClient,
    LLMError,
    LLMTimeoutError,
)
from zenfa_ai.evaluator.schemas import (
    EvaluationResponse,
    ExplanationResponse,
    ScoreBreakdown,
    Suggestion,
)
from zenfa_ai.models.build import BuildRequest
from zenfa_ai.models.components import ComponentWithPrice
from zenfa_ai.orchestrator.decisions import Decision, should_continue
from zenfa_ai.orchestrator.loop import run_build_loop
from zenfa_ai.orchestrator.state import LoopState


# ──────────────────────────────────────────────
# Test Data
# ──────────────────────────────────────────────

SAMPLE_COMPONENTS = [
    ComponentWithPrice(
        id=1, name="AMD Ryzen 5 7600", slug="amd-ryzen-5-7600",
        component_type="cpu", brand="AMD", performance_score=72,
        price_bdt=22500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "AM5", "tdp": 65, "integrated_graphics": True},
    ),
    ComponentWithPrice(
        id=3, name="MSI B650M MORTAR WIFI", slug="msi-b650m-mortar-wifi",
        component_type="motherboard", brand="MSI", performance_score=65,
        price_bdt=18500, vendor_name="StarTech", in_stock=True,
        specs={"socket": "AM5", "form_factor": "mATX", "ram_type": "DDR5",
               "ram_slots": 4, "max_ram_gb": 128},
    ),
    ComponentWithPrice(
        id=7, name="G.Skill DDR5 16GB", slug="gskill-ddr5",
        component_type="ram", brand="G.Skill", performance_score=75,
        price_bdt=5800, vendor_name="StarTech", in_stock=True,
        specs={"ram_type": "DDR5", "capacity_gb": 16, "modules": 1},
    ),
    ComponentWithPrice(
        id=6, name="MSI RTX 4060", slug="msi-rtx-4060",
        component_type="gpu", brand="NVIDIA", performance_score=68,
        price_bdt=37000, vendor_name="StarTech", in_stock=True,
        specs={"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550},
    ),
    ComponentWithPrice(
        id=10, name="Corsair RM750e", slug="corsair-rm750e",
        component_type="psu", brand="Corsair", performance_score=70,
        price_bdt=9500, vendor_name="StarTech", in_stock=True,
        specs={"wattage": 750},
    ),
    ComponentWithPrice(
        id=9, name="Samsung 980 PRO 1TB", slug="samsung-980-pro",
        component_type="storage", brand="Samsung", performance_score=80,
        price_bdt=8500, vendor_name="StarTech", in_stock=True,
        specs={"storage_type": "NVMe", "capacity_gb": 1000},
    ),
    ComponentWithPrice(
        id=11, name="NZXT H5 Flow", slug="nzxt-h5-flow",
        component_type="case", brand="NZXT", performance_score=65,
        price_bdt=8500, vendor_name="StarTech", in_stock=True,
        specs={"max_gpu_length_mm": 365, "form_factor_support": ["ATX", "mATX", "ITX"]},
    ),
]


def _make_request(budget_max: int = 150000) -> BuildRequest:
    return BuildRequest(
        budget_min=80000, budget_max=budget_max,
        purpose="gaming", components=SAMPLE_COMPONENTS,
    )


def _high_score_eval(**overrides) -> EvaluationResponse:
    """Build a high-score evaluation response."""
    data = {
        "scores": ScoreBreakdown(
            performance_match=3, value_score=3,
            build_balance=2, future_proofing=1, community_trust=1,
        ),
        "final_score": 9.0,
        "reasoning": "Great build",
        "suggestions": [],
        "red_flags": [],
        "approved": True,
    }
    data.update(overrides)
    return EvaluationResponse(**data)


def _low_score_eval(**overrides) -> EvaluationResponse:
    """Build a low-score evaluation with suggestions."""
    data = {
        "scores": ScoreBreakdown(
            performance_match=2, value_score=1,
            build_balance=1, future_proofing=1, community_trust=1,
        ),
        "final_score": 6.0,
        "reasoning": "GPU underpowered",
        "suggestions": [
            Suggestion(
                action="swap", component_category="gpu",
                current_component="RTX 4060",
                suggested_alternatives=["RX 7700 XT"],
                reason="Better gaming value", priority="high",
            )
        ],
        "red_flags": [],
        "approved": False,
    }
    data.update(overrides)
    return EvaluationResponse(**data)


def _mock_explanation() -> ExplanationResponse:
    return ExplanationResponse(
        summary="A well-balanced gaming build.",
        per_component={"cpu": "Ryzen 5 7600 is great for gaming."},
        trade_offs="DDR5 instead of DDR4",
        upgrade_path="Add more RAM",
    )


# ──────────────────────────────────────────────
# Decision Gate Tests
# ──────────────────────────────────────────────


class TestDecisionGate:
    def test_score_meets_target_within_window(self):
        """Score >= 8.5 within optimization window → CONTINUE_OPTIMIZING."""
        state = LoopState(start_time=time.monotonic())
        state.record_build(MagicMock(), 9.0)
        assert should_continue(state) == Decision.CONTINUE_OPTIMIZING

    def test_score_meets_target_past_window(self):
        """Score >= 8.5 past optimization window → RETURN_CURRENT."""
        state = LoopState(start_time=time.monotonic() - 100)
        state.record_build(MagicMock(), 9.0)
        assert should_continue(state) == Decision.RETURN_CURRENT

    def test_score_above_minimum_improving(self):
        """Score >= 7.0 and improving → CONTINUE."""
        state = LoopState(start_time=time.monotonic())
        state.record_build(MagicMock(), 6.0)
        state.record_build(MagicMock(), 7.5)
        assert should_continue(state) == Decision.CONTINUE

    def test_score_above_minimum_plateaued(self):
        """Score >= 7.0 but not improving → RETURN_CURRENT."""
        state = LoopState(start_time=time.monotonic())
        state.record_build(MagicMock(), 7.5)
        state.record_build(MagicMock(), 7.3)
        assert should_continue(state) == Decision.RETURN_CURRENT

    def test_score_below_minimum(self):
        """Score < 7.0 → CONTINUE."""
        state = LoopState(start_time=time.monotonic())
        state.record_build(MagicMock(), 5.0)
        assert should_continue(state) == Decision.CONTINUE

    def test_max_iterations_exceeded(self):
        """Max iterations reached → RETURN_BEST."""
        state = LoopState(start_time=time.monotonic(), max_iterations=2)
        state.record_build(MagicMock(), 5.0)
        state.record_build(MagicMock(), 6.0)
        assert should_continue(state) == Decision.RETURN_BEST

    def test_time_exceeded(self):
        """Time exceeded → RETURN_BEST."""
        state = LoopState(start_time=time.monotonic() - 200, max_time=120)
        state.record_build(MagicMock(), 8.0)
        assert should_continue(state) == Decision.RETURN_BEST

    def test_llm_failed(self):
        """LLM failure → RETURN_KNAPSACK_ONLY."""
        state = LoopState(start_time=time.monotonic())
        state.llm_failed = True
        assert should_continue(state) == Decision.RETURN_KNAPSACK_ONLY


# ──────────────────────────────────────────────
# Loop State Tests
# ──────────────────────────────────────────────


class TestLoopState:
    def test_record_build_tracks_best(self):
        """Best build should be the highest-scored one."""
        state = LoopState()
        state.record_build(MagicMock(), 7.0)
        state.record_build(MagicMock(), 9.0)
        state.record_build(MagicMock(), 8.0)
        assert state.best_score == 9.0
        assert state.iteration_count == 3

    def test_score_improving_property(self):
        """score_improving should compare last two scores."""
        state = LoopState()
        state.record_build(MagicMock(), 6.0)
        assert state.score_improving is True  # Only 1 score
        state.record_build(MagicMock(), 7.0)
        assert state.score_improving is True
        state.record_build(MagicMock(), 6.5)
        assert state.score_improving is False

    def test_oscillation_detection(self):
        """Should lock categories when A→B→A swap detected."""
        state = LoopState()

        # Iteration 1: swap GPU from RTX 4060 to RX 7700 XT
        s1 = [Suggestion(
            action="swap", component_category="gpu",
            current_component="RTX 4060",
            suggested_alternatives=["RX 7700 XT"],
            reason="better value", priority="high",
        )]
        locked1 = state.detect_oscillation(s1)
        assert len(locked1) == 0  # No oscillation yet

        # Iteration 2: swap GPU back from RX 7700 XT to RTX 4060
        s2 = [Suggestion(
            action="swap", component_category="gpu",
            current_component="RX 7700 XT",
            suggested_alternatives=["RTX 4060"],  # Was previous "current"
            reason="actually RTX is fine", priority="high",
        )]
        locked2 = state.detect_oscillation(s2)
        assert "gpu" in locked2  # Oscillation detected!
        assert "gpu" in state.locked_components

    def test_low_budget_threshold(self):
        state = LoopState(budget_max=20000)
        assert state.is_low_budget is True
        state2 = LoopState(budget_max=80000)
        assert state2.is_low_budget is False

    def test_high_budget_threshold(self):
        state = LoopState(budget_max=600000)
        assert state.is_high_budget is True


# ──────────────────────────────────────────────
# Full Loop Integration Tests (Mocked LLM)
# ──────────────────────────────────────────────


class TestLoopIntegration:
    @pytest.mark.asyncio
    async def test_high_score_converges_quickly(self):
        """High score on first iteration → exits after continue_optimizing."""
        client = GeminiClient(api_key="test")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = json.dumps({
                "scores": {"performance_match": 3, "value_score": 3,
                           "build_balance": 2, "future_proofing": 1,
                           "community_trust": 1},
                "final_score": 9.0,
                "reasoning": "Great build",
                "suggestions": [],
                "red_flags": [],
                "approved": True,
            })

            response = await run_build_loop(
                _make_request(),
                llm_client=client,
                max_iterations=5,
                max_time=120,
                optimization_window=0,  # Force exit immediately after target
            )

            assert response.quality.score == 9.0
            assert response.quality.iterations_used == 1
            assert response.build.total_price > 0

    @pytest.mark.asyncio
    async def test_knapsack_only_when_no_llm(self):
        """No LLM client → knapsack-only build."""
        response = await run_build_loop(_make_request(), llm_client=None)
        assert response.quality.score == 0.0
        assert response.metadata.llm_model == "knapsack-only"
        assert response.build.total_price > 0
        assert len(response.build.components) > 0

    @pytest.mark.asyncio
    async def test_knapsack_only_on_low_budget(self):
        """Budget below threshold → skip LLM, knapsack-only."""
        request = BuildRequest(
            budget_min=15000, budget_max=20000,
            purpose="office", components=SAMPLE_COMPONENTS,
        )
        client = GeminiClient(api_key="test")
        response = await run_build_loop(request, llm_client=client)
        assert response.quality.score == 0.0

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        """LLM failure → graceful degradation to knapsack-only."""
        client = GeminiClient(api_key="test")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMError("API down")
            response = await run_build_loop(
                _make_request(), llm_client=client,
            )
            # Should return a build even though LLM failed
            assert response.build.total_price > 0
            assert len(response.build.components) > 0

    @pytest.mark.asyncio
    async def test_max_iterations_returns_best(self):
        """Should return best build when max iterations reached."""
        client = GeminiClient(api_key="test")

        # Simulate improving scores over iterations
        call_count = 0

        async def fake_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            score = 5.0 + call_count * 0.5  # 5.5, 6.0, 6.5
            return json.dumps({
                "scores": {"performance_match": 2, "value_score": 1,
                           "build_balance": 1, "future_proofing": 1,
                           "community_trust": 1},
                "final_score": score,
                "reasoning": f"Score {score}",
                "suggestions": [
                    {"action": "swap", "component_category": "gpu",
                     "current_component": "RTX 4060",
                     "suggested_alternatives": [f"GPU-v{call_count + 1}"],
                     "reason": "better", "priority": "high"}
                ],
                "red_flags": [],
                "approved": False,
            })

        with patch.object(client, "_call_llm", side_effect=fake_llm):
            response = await run_build_loop(
                _make_request(), llm_client=client,
                max_iterations=3,
            )
            assert response.quality.iterations_used == 3
            assert response.build.total_price > 0

    @pytest.mark.asyncio
    async def test_response_structure(self):
        """Response should have all required fields."""
        response = await run_build_loop(_make_request(), llm_client=None)
        assert response.build is not None
        assert response.quality is not None
        assert response.explanation is not None
        assert response.metadata is not None
        assert response.metadata.engine_version == "0.1.0"
