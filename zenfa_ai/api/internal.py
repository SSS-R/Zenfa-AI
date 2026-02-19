"""Internal API gateway for PC Lagbe? website integration.

Routes under /internal/* — no API key required.
This is meant to be called by the PC Lagbe? backend
(service-to-service, behind a firewall or VPN).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter

from zenfa_ai.cache.redis_cache import BuildCache, request_to_cache_key
from zenfa_ai.engine.compatibility import check_compatibility
from zenfa_ai.evaluator.llm_client import BaseLLMClient
from zenfa_ai.models.build import BuildRequest, BuildResponse, CandidateBuild
from zenfa_ai.orchestrator.loop import run_build_loop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["Internal — PC Lagbe?"])

# Set by app lifespan — shared resources
_llm_client: Optional[BaseLLMClient] = None
_cache: Optional[BuildCache] = None


def set_llm_client(client: Optional[BaseLLMClient]) -> None:
    """Called during app startup to inject the LLM client."""
    global _llm_client
    _llm_client = client


def set_cache(cache: Optional[BuildCache]) -> None:
    """Called during app startup to inject the cache."""
    global _cache
    _cache = cache


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────


@router.get("/health")
async def health():
    """Health check for PC Lagbe? backend monitoring."""
    return {
        "status": "healthy",
        "gateway": "internal",
        "llm_available": _llm_client is not None,
        "cache_available": _cache is not None and _cache.available,
    }


@router.post("/build", response_model=BuildResponse)
async def generate_build(request: BuildRequest):
    """Generate an optimized PC build for PC Lagbe? users.

    This runs the full Knapsack ⇄ LLM negotiation loop.
    No API key required — trusted internal call.
    Results are cached for repeat requests.
    """
    logger.info(
        "Internal build request: %s, budget %d–%d৳",
        request.purpose, request.budget_min, request.budget_max,
    )

    # Check cache first
    cache_key = request_to_cache_key(request)
    if _cache:
        cached = await _cache.get(cache_key)
        if cached:
            logger.info("Cache HIT for internal build")
            resp = BuildResponse.model_validate_json(cached)
            # Mark as cached in metadata
            resp.metadata.cached = True
            return resp

    response = await run_build_loop(
        request=request,
        llm_client=_llm_client,
    )

    # Store in cache
    if _cache:
        await _cache.set(cache_key, response.model_dump_json())

    logger.info(
        "Internal build complete: score=%.1f, %d iterations, %d৳",
        response.quality.score,
        response.quality.iterations_used,
        response.build.total_price,
    )

    return response


@router.post("/compatibility/check")
async def check_build_compatibility(build: CandidateBuild):
    """Validate component compatibility for PC Lagbe? builder UI.

    Use this to check compatibility in real-time as users
    select components in the interactive builder.
    """
    result = check_compatibility(build)
    return {
        "compatible": result.passed,
        "violations": [
            {"rule": v.rule, "message": v.message}
            for v in result.violations
        ],
    }
