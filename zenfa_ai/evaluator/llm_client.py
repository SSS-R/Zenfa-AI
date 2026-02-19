"""LLM client with provider abstraction.

Supports Gemini (primary) and OpenAI (fallback) with:
- Structured JSON output
- Retry logic (1 retry on invalid JSON, 1 retry on rate limit)
- 15-second timeout per call
- Automatic fallback: Gemini fails → OpenAI

Designed for future RAG extension — the evaluate() method accepts
a retrieval_context parameter that will be passed through to prompts.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from zenfa_ai.evaluator.schemas import EvaluationResponse, ExplanationResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_TIMEOUT = 15  # seconds
MAX_RETRIES = 1
RATE_LIMIT_BACKOFF = 2  # seconds


# ──────────────────────────────────────────────
# Abstract Base Client
# ──────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract LLM client interface.

    Subclass this to add new providers. The engine only depends on this
    interface, so swapping providers requires no logic changes.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make the raw LLM API call. Returns the response text."""
        ...

    async def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> EvaluationResponse:
        """Call the LLM and parse the response into an EvaluationResponse."""
        return await self._call_with_retry(
            system_prompt, user_prompt, EvaluationResponse
        )

    async def generate_explanation(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ExplanationResponse:
        """Call the LLM for explanation generation."""
        return await self._call_with_retry(
            system_prompt, user_prompt, ExplanationResponse
        )

    async def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_type: Type[T],
    ) -> T:
        """Call LLM with retry logic for JSON parsing and rate limits."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = await self._call_llm(system_prompt, user_prompt)
                return self._parse_response(raw, response_type)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    "LLM response parse failed (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES + 1, e,
                )
                last_error = e
                # Retry with no backoff for parse errors

            except RateLimitError as e:
                logger.warning(
                    "Rate limited (attempt %d/%d), backing off %ds",
                    attempt + 1, MAX_RETRIES + 1, RATE_LIMIT_BACKOFF,
                )
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RATE_LIMIT_BACKOFF)

            except LLMTimeoutError as e:
                logger.error("LLM timed out after %ds", self.timeout)
                raise

            except LLMError as e:
                logger.error("LLM call failed: %s", e)
                raise

        raise LLMError(f"Failed after {MAX_RETRIES + 1} attempts: {last_error}")

    def _parse_response(self, raw: str, response_type: Type[T]) -> T:
        """Extract JSON from the LLM response and validate against schema."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` or ``` ... ```
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)
        return response_type.model_validate(data)

    @property
    def provider_name(self) -> str:
        """Human-readable provider name for metadata."""
        return self.__class__.__name__


# ──────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────


class LLMError(Exception):
    """Base exception for LLM client errors."""


class RateLimitError(LLMError):
    """Raised when the LLM API returns a rate limit error."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM API call times out."""


# ──────────────────────────────────────────────
# Gemini Client
# ──────────────────────────────────────────────


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.3,
                        "max_output_tokens": 2048,
                    },
                )
            except ImportError:
                raise LLMError(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API with structured JSON output."""
        import asyncio

        client = self._get_client()

        try:
            # Gemini uses generate_content — run in thread for async
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.generate_content(
                        [
                            {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]},
                        ]
                    ),
                ),
                timeout=self.timeout,
            )
            return response.text

        except asyncio.TimeoutError:
            raise LLMTimeoutError(f"Gemini timed out after {self.timeout}s")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                raise RateLimitError(f"Gemini rate limited: {e}")
            raise LLMError(f"Gemini API error: {e}")


# ──────────────────────────────────────────────
# OpenAI Client
# ──────────────────────────────────────────────


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible LLM client (GPT-4o-mini, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except ImportError:
                raise LLMError(
                    "openai not installed. "
                    "Install with: pip install openai"
                )
        return self._client

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API with JSON mode."""
        import asyncio

        client = self._get_client()

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=2048,
                ),
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""

        except asyncio.TimeoutError:
            raise LLMTimeoutError(f"OpenAI timed out after {self.timeout}s")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                raise RateLimitError(f"OpenAI rate limited: {e}")
            raise LLMError(f"OpenAI API error: {e}")


# ──────────────────────────────────────────────
# Factory + Fallback Client
# ──────────────────────────────────────────────


class FallbackLLMClient(BaseLLMClient):
    """Wraps a primary + fallback client. Falls back on any LLMError."""

    def __init__(
        self,
        primary: BaseLLMClient,
        fallback: BaseLLMClient,
    ) -> None:
        super().__init__(
            model=primary.model,
            api_key=primary.api_key,
            timeout=primary.timeout,
        )
        self.primary = primary
        self.fallback = fallback
        self._used_fallback = False

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Try primary, fall back to secondary on failure."""
        try:
            result = await self.primary._call_llm(system_prompt, user_prompt)
            self._used_fallback = False
            return result
        except LLMError as primary_error:
            logger.warning(
                "Primary LLM (%s) failed: %s. Falling back to %s.",
                self.primary.provider_name,
                primary_error,
                self.fallback.provider_name,
            )
            try:
                result = await self.fallback._call_llm(system_prompt, user_prompt)
                self._used_fallback = True
                return result
            except LLMError as fallback_error:
                raise LLMError(
                    f"Both LLMs failed. "
                    f"Primary ({self.primary.provider_name}): {primary_error}. "
                    f"Fallback ({self.fallback.provider_name}): {fallback_error}"
                )

    @property
    def used_fallback(self) -> bool:
        """Whether the last call used the fallback provider."""
        return self._used_fallback

    @property
    def provider_name(self) -> str:
        if self._used_fallback:
            return f"{self.fallback.provider_name} (fallback)"
        return self.primary.provider_name


def create_llm_client(
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.0-flash",
    openai_model: str = "gpt-4o-mini",
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[BaseLLMClient]:
    """Factory function to create the best available LLM client.

    Priority:
    1. Gemini + OpenAI → FallbackLLMClient
    2. Gemini only
    3. OpenAI only
    4. None → returns None (knapsack-only mode)
    """
    primary = None
    fallback = None

    if gemini_api_key:
        primary = GeminiClient(
            model=gemini_model, api_key=gemini_api_key, timeout=timeout
        )

    if openai_api_key:
        fallback = OpenAIClient(
            model=openai_model, api_key=openai_api_key, timeout=timeout
        )

    if primary and fallback:
        return FallbackLLMClient(primary=primary, fallback=fallback)
    elif primary:
        return primary
    elif fallback:
        return fallback
    else:
        logger.warning(
            "No LLM API keys configured. Engine will run in knapsack-only mode."
        )
        return None
