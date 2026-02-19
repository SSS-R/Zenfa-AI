"""Tests for the LLM evaluator service.

All LLM calls are mocked — no real API keys needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from zenfa_ai.evaluator.llm_client import (
    BaseLLMClient,
    FallbackLLMClient,
    GeminiClient,
    LLMError,
    LLMTimeoutError,
    OpenAIClient,
    RateLimitError,
    create_llm_client,
)
from zenfa_ai.evaluator.prompts import (
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_user_prompt,
)
from zenfa_ai.evaluator.schemas import (
    EvaluationResponse,
    ExplanationResponse,
    ScoreBreakdown,
    Suggestion,
)
from zenfa_ai.models.build import CandidateBuild, SelectedComponent


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

VALID_EVALUATION_JSON = json.dumps({
    "scores": {
        "performance_match": 3,
        "value_score": 2,
        "build_balance": 2,
        "future_proofing": 1,
        "community_trust": 1,
    },
    "final_score": 9.0,
    "reasoning": "Solid gaming build with excellent GPU allocation.",
    "suggestions": [],
    "red_flags": [],
    "approved": True,
})

LOW_SCORE_EVALUATION_JSON = json.dumps({
    "scores": {
        "performance_match": 2,
        "value_score": 1,
        "build_balance": 1,
        "future_proofing": 1,
        "community_trust": 1,
    },
    "final_score": 6.0,
    "reasoning": "GPU is underpowered for gaming at this budget.",
    "suggestions": [
        {
            "action": "swap",
            "component_category": "gpu",
            "current_component": "RTX 4060",
            "suggested_alternatives": ["RX 7700 XT", "RX 7800 XT"],
            "reason": "Better gaming performance per taka",
            "priority": "high",
        }
    ],
    "red_flags": [],
    "approved": False,
})

VALID_EXPLANATION_JSON = json.dumps({
    "summary": "A well-balanced gaming build that maximizes GPU performance.",
    "per_component": {
        "cpu": "Ryzen 5 7600 offers excellent single-thread for gaming.",
        "gpu": "RX 7700 XT is the sweet spot for 1080p/1440p gaming.",
    },
    "trade_offs": "DDR5 RAM instead of DDR4 to save budget for the GPU.",
    "upgrade_path": "Add a second 16GB RAM stick for 32GB total.",
})


def _make_build() -> CandidateBuild:
    """Create a minimal CandidateBuild for testing."""
    return CandidateBuild(
        components=[
            SelectedComponent(
                id=1, name="AMD Ryzen 5 7600", component_type="cpu",
                price_bdt=22500, vendor_name="StarTech",
                specs={"socket": "AM5", "tdp": 65},
            ),
            SelectedComponent(
                id=6, name="MSI RTX 4060", component_type="gpu",
                price_bdt=37000, vendor_name="StarTech",
                specs={"vram_gb": 8},
            ),
        ],
        total_price=59500,
        remaining_budget=20500,
        compatibility_verified=True,
    )


# ──────────────────────────────────────────────
# Schema Tests
# ──────────────────────────────────────────────


class TestSchemas:
    def test_evaluation_response_parsing(self):
        """Valid JSON should parse into EvaluationResponse."""
        response = EvaluationResponse.model_validate_json(VALID_EVALUATION_JSON)
        assert response.final_score == 9.0
        assert response.approved is True
        assert response.scores.total == 9

    def test_low_score_with_suggestions(self):
        """Low-scoring response should have suggestions."""
        response = EvaluationResponse.model_validate_json(LOW_SCORE_EVALUATION_JSON)
        assert response.final_score == 6.0
        assert response.approved is False
        assert len(response.suggestions) == 1
        assert response.suggestions[0].action == "swap"
        assert response.suggestions[0].component_category == "gpu"

    def test_score_breakdown_totals(self):
        """ScoreBreakdown.total should sum all category scores."""
        breakdown = ScoreBreakdown(
            performance_match=3, value_score=3,
            build_balance=2, future_proofing=1, community_trust=1,
        )
        assert breakdown.total == 10

    def test_explanation_response_parsing(self):
        """Valid explanation JSON should parse correctly."""
        response = ExplanationResponse.model_validate_json(VALID_EXPLANATION_JSON)
        assert "gaming" in response.summary.lower()
        assert "cpu" in response.per_component

    def test_score_validation_bounds(self):
        """Score fields should reject out-of-range values."""
        with pytest.raises(Exception):
            ScoreBreakdown(
                performance_match=5,  # max is 3
                value_score=3,
                build_balance=2,
                future_proofing=1,
                community_trust=1,
            )


# ──────────────────────────────────────────────
# Prompt Tests
# ──────────────────────────────────────────────


class TestPrompts:
    def test_first_iteration_prompt(self):
        """First iteration prompt should include build info, no suggestion results."""
        build = _make_build()
        prompt = build_user_prompt(
            build=build, purpose="gaming",
            budget_min=80000, budget_max=80000,
            iteration=1,
        )
        assert "Iteration 1" in prompt
        assert "gaming" in prompt
        assert "80,000৳" in prompt
        assert "Ryzen 5 7600" in prompt
        assert "Previous Suggestion" not in prompt

    def test_subsequent_iteration_prompt(self):
        """Subsequent iterations should include suggestion results."""
        from zenfa_ai.engine.knapsack import SuggestionResult, SuggestionStatus

        build = _make_build()
        results = [
            SuggestionResult(
                suggestion="swap gpu to RX 7700 XT",
                status=SuggestionStatus.APPLIED,
                note="Available at StarTech",
            ),
        ]
        prompt = build_user_prompt(
            build=build, purpose="gaming",
            budget_min=80000, budget_max=80000,
            iteration=2,
            suggestion_results=results,
        )
        assert "Iteration 2" in prompt
        assert "Previous Suggestion" in prompt
        assert "APPLIED" in prompt

    def test_budget_range_display(self):
        """Budget range should show min–max when different."""
        build = _make_build()
        prompt = build_user_prompt(
            build=build, purpose="gaming",
            budget_min=60000, budget_max=100000,
            iteration=1,
        )
        assert "60,000৳" in prompt
        assert "100,000৳" in prompt

    def test_explanation_prompt(self):
        """Explanation prompt should include final score and components."""
        build = _make_build()
        prompt = build_explanation_prompt(
            build=build, purpose="gaming",
            budget_min=80000, budget_max=80000,
            final_score=9.0,
        )
        assert "9.0/10" in prompt
        assert "Ryzen 5 7600" in prompt

    def test_system_prompt_contains_rubric(self):
        """System prompt should contain the scoring rubric."""
        assert "Performance Match" in SYSTEM_PROMPT
        assert "Value Score" in SYSTEM_PROMPT
        assert "Build Balance" in SYSTEM_PROMPT
        assert "Future-Proofing" in SYSTEM_PROMPT
        assert "Community Trust" in SYSTEM_PROMPT

    def test_rag_context_injection(self):
        """Future RAG context should appear in prompt when provided."""
        build = _make_build()
        prompt = build_user_prompt(
            build=build, purpose="gaming",
            budget_min=80000, budget_max=80000,
            iteration=1,
            retrieval_context="Reddit says RX 7700 XT is amazing for 1080p",
        )
        assert "Reddit says RX 7700 XT" in prompt
        assert "Recent Community Data" in prompt


# ──────────────────────────────────────────────
# LLM Client Tests (Mocked)
# ──────────────────────────────────────────────


class TestLLMClientParsing:
    """Test JSON parsing and code fence stripping."""

    @pytest.mark.asyncio
    async def test_parse_clean_json(self):
        """Clean JSON response should parse correctly."""
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = VALID_EVALUATION_JSON
            result = await client.evaluate("system", "user")
            assert result.final_score == 9.0
            assert result.approved is True

    @pytest.mark.asyncio
    async def test_parse_json_with_code_fence(self):
        """JSON wrapped in ```json ... ``` should be parsed correctly."""
        fenced = f"```json\n{VALID_EVALUATION_JSON}\n```"
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = fenced
            result = await client.evaluate("system", "user")
            assert result.final_score == 9.0

    @pytest.mark.asyncio
    async def test_parse_invalid_json_retries(self):
        """Invalid JSON should trigger a retry."""
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.side_effect = [
                "not valid json {{{",   # First call fails
                VALID_EVALUATION_JSON,  # Retry succeeds
            ]
            result = await client.evaluate("system", "user")
            assert result.final_score == 9.0
            assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_all_retries_exhausted(self):
        """All retries failing should raise LLMError."""
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "not valid json {{{"
            with pytest.raises(LLMError, match="Failed after"):
                await client.evaluate("system", "user")

    @pytest.mark.asyncio
    async def test_explanation_parsing(self):
        """Explanation generation should parse ExplanationResponse."""
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = VALID_EXPLANATION_JSON
            result = await client.generate_explanation("system", "user")
            assert "gaming" in result.summary.lower()


class TestFallbackClient:
    """Test the fallback behavior (Gemini → OpenAI)."""

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self):
        """When primary works, fallback is not used."""
        primary = GeminiClient(api_key="gemini-key")
        fallback = OpenAIClient(api_key="openai-key")
        client = FallbackLLMClient(primary=primary, fallback=fallback)

        with patch.object(primary, "_call_llm", new_callable=AsyncMock) as p_mock:
            p_mock.return_value = VALID_EVALUATION_JSON
            result = await client.evaluate("system", "user")
            assert result.final_score == 9.0
            assert client.used_fallback is False

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self):
        """When primary fails, fallback should be used."""
        primary = GeminiClient(api_key="gemini-key")
        fallback = OpenAIClient(api_key="openai-key")
        client = FallbackLLMClient(primary=primary, fallback=fallback)

        with patch.object(primary, "_call_llm", new_callable=AsyncMock) as p_mock, \
             patch.object(fallback, "_call_llm", new_callable=AsyncMock) as f_mock:
            p_mock.side_effect = LLMError("Gemini down")
            f_mock.return_value = VALID_EVALUATION_JSON
            result = await client.evaluate("system", "user")
            assert result.final_score == 9.0
            assert client.used_fallback is True

    @pytest.mark.asyncio
    async def test_both_fail_raises(self):
        """When both primary and fallback fail, should raise LLMError."""
        primary = GeminiClient(api_key="gemini-key")
        fallback = OpenAIClient(api_key="openai-key")
        client = FallbackLLMClient(primary=primary, fallback=fallback)

        with patch.object(primary, "_call_llm", new_callable=AsyncMock) as p_mock, \
             patch.object(fallback, "_call_llm", new_callable=AsyncMock) as f_mock:
            p_mock.side_effect = LLMError("Gemini down")
            f_mock.side_effect = LLMError("OpenAI down")
            with pytest.raises(LLMError, match="Both LLMs failed"):
                await client.evaluate("system", "user")

    @pytest.mark.asyncio
    async def test_timeout_not_retried(self):
        """Timeout errors should not be retried (immediate raise)."""
        client = GeminiClient(api_key="test-key")
        with patch.object(client, "_call_llm", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMTimeoutError("timed out")
            with pytest.raises(LLMTimeoutError):
                await client.evaluate("system", "user")
            assert mock.call_count == 1  # No retry


class TestClientFactory:
    def test_both_keys_returns_fallback(self):
        """Both keys → FallbackLLMClient."""
        client = create_llm_client(
            gemini_api_key="gk", openai_api_key="ok"
        )
        assert isinstance(client, FallbackLLMClient)

    def test_gemini_only(self):
        """Only Gemini key → GeminiClient."""
        client = create_llm_client(gemini_api_key="gk")
        assert isinstance(client, GeminiClient)

    def test_openai_only(self):
        """Only OpenAI key → OpenAIClient."""
        client = create_llm_client(openai_api_key="ok")
        assert isinstance(client, OpenAIClient)

    def test_no_keys_returns_none(self):
        """No keys → None (knapsack-only mode)."""
        client = create_llm_client()
        assert client is None

    def test_provider_name(self):
        """Provider name should reflect which client is used."""
        gemini = GeminiClient(api_key="gk")
        assert gemini.provider_name == "GeminiClient"
        openai = OpenAIClient(api_key="ok")
        assert openai.provider_name == "OpenAIClient"
