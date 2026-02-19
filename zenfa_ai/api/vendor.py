"""Vendor API gateway for B2B API sales.

Routes under /v1/* — requires X-API-Key header.
This is the API that vendors pay for.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends

from zenfa_ai.api.auth import check_rate_limit
from zenfa_ai.engine.compatibility import check_compatibility
from zenfa_ai.evaluator.llm_client import BaseLLMClient
from zenfa_ai.models.build import BuildRequest, BuildResponse, CandidateBuild
from zenfa_ai.orchestrator.loop import run_build_loop

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1",
    tags=["Vendor API — B2B"],
    dependencies=[Depends(check_rate_limit)],  # Auth + rate limit on all routes
)

# Set by app lifespan — shared LLM client
_llm_client: Optional[BaseLLMClient] = None


def set_llm_client(client: Optional[BaseLLMClient]) -> None:
    """Called during app startup to inject the LLM client."""
    global _llm_client
    _llm_client = client


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────


@router.get("/health")
async def health():
    """Health check for vendor monitoring dashboards."""
    return {
        "status": "healthy",
        "gateway": "vendor",
        "api_version": "v1",
    }


@router.post("/build", response_model=BuildResponse)
async def generate_build(request: BuildRequest):
    """Generate an optimized PC build via the Zenfa AI engine.

    Requires a valid X-API-Key header. Rate limited per key.

    The engine runs a Knapsack ⇄ LLM negotiation loop to find
    the best build within the given budget and purpose.
    """
    logger.info(
        "Vendor build request: %s, budget %d–%d৳",
        request.purpose, request.budget_min, request.budget_max,
    )

    response = await run_build_loop(
        request=request,
        llm_client=_llm_client,
    )

    logger.info(
        "Vendor build complete: score=%.1f, %d iterations, %d৳",
        response.quality.score,
        response.quality.iterations_used,
        response.build.total_price,
    )

    return response


@router.post("/compatibility/check")
async def check_build_compatibility(build: CandidateBuild):
    """Validate component compatibility.

    Pass a candidate build to check all 9 compatibility rules.
    Useful for vendors building their own UI on top of Zenfa.
    """
    result = check_compatibility(build)
    return {
        "compatible": result.passed,
        "violations": [
            {"rule": v.rule, "message": v.message}
            for v in result.violations
        ],
    }
