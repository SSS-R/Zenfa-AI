"""Zenfa AI Engine — FastAPI application.

Mounts two API gateways:
- /internal/* — PC Lagbe? website (unauthenticated, service-to-service)
- /v1/*      — Vendor B2B API (X-API-Key authenticated, rate limited)

Both gateways share the same core engine.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zenfa_ai.api.internal import router as internal_router
from zenfa_ai.api.internal import set_llm_client as set_internal_llm
from zenfa_ai.api.internal import set_cache as set_internal_cache
from zenfa_ai.api.vendor import router as vendor_router
from zenfa_ai.api.vendor import set_llm_client as set_vendor_llm
from zenfa_ai.api.vendor import set_cache as set_vendor_cache
from zenfa_ai.cache.redis_cache import BuildCache
from zenfa_ai.evaluator.llm_client import create_llm_client

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan — startup / shutdown
# ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, cleanup on shutdown."""
    # Create LLM client (shared by both gateways)
    llm_client = create_llm_client(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        timeout=int(os.getenv("LLM_TIMEOUT", "15")),
    )

    # Initialize Redis cache
    cache = BuildCache()
    cache_connected = await cache.connect()

    # Inject into both gateways
    set_internal_llm(llm_client)
    set_vendor_llm(llm_client)
    set_internal_cache(cache if cache_connected else None)
    set_vendor_cache(cache if cache_connected else None)

    if llm_client:
        logger.info(
            "LLM client ready: %s (fallback: %s)",
            llm_client.provider_name,
            hasattr(llm_client, "fallback"),
        )
    else:
        logger.warning("No LLM keys — running in knapsack-only mode")

    if cache_connected:
        logger.info("Redis cache connected")
    else:
        logger.warning("Redis unavailable — caching disabled")

    yield

    # Cleanup
    await cache.disconnect()
    logger.info("Shutting down Zenfa AI Engine")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────


def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI app."""
    app = FastAPI(
        title="Zenfa AI Engine",
        description=(
            "AI-powered PC build optimization engine for the Bangladesh market.\n\n"
            "## Gateways\n\n"
            "- **Internal** (`/internal/*`): For PC Lagbe? website — no auth required\n"
            "- **Vendor** (`/v1/*`): For B2B API customers — requires `X-API-Key`\n"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow PC Lagbe? frontend and vendor domains
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount gateways
    app.include_router(internal_router)
    app.include_router(vendor_router)

    # Root health check
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "engine": "Zenfa AI",
            "version": "0.1.0",
            "gateways": {
                "internal": "/internal (PC Lagbe?)",
                "vendor": "/v1 (B2B API — requires X-API-Key)",
            },
        }

    return app


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "zenfa_ai.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
