"""
Abstract LLM Client Interface
==============================

``LLMClient`` is the **only** interface the rest of the application should
program against.  It deliberately hides all provider-specific details (URLs,
auth, payload shapes) behind a stable, semantic API.

The architecture enforces the Dependency Inversion Principle:

    High-level modules (AIService, pipeline, agents)
        depend on → LLMClient (abstraction)

    Low-level modules (OllamaClient, future OpenAIClient)
        implement  → LLMClient (abstraction)

This means swapping Ollama for GPT-4 or Claude means:
  1. Write a new class that inherits LLMClient.
  2. Change a single env var (LLM_PROVIDER).
  3. Zero changes to AIService or any business logic.

Method Contract
---------------
All methods are ``async`` because:
  - LLM calls are I/O-bound (HTTP).
  - We run inside FastAPI which is an async framework.
  - Blocking the event loop would degrade ALL concurrent requests.

Every concrete implementation MUST honour these contracts:
  - Raise ``LLMConnectionError`` when the backend is unreachable.
  - Raise ``LLMTimeoutError`` when a request exceeds the configured timeout.
  - Raise ``LLMJSONError`` from ``extract_json`` when valid JSON cannot be parsed.
  - Raise ``LLMResponseError`` for unexpected HTTP errors.
  - Never raise raw ``httpx`` or ``requests`` exceptions to callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.llm.models import (
    ChatMessage,
    ChatResponse,
    ExtractionResponse,
    GenerateResponse,
    HealthResponse,
)


class LLMClient(ABC):
    """
    Abstract base class defining the contract for all LLM provider clients.

    All public methods are async.  Implementations must not block the event loop.

    Usage example (production code should use ``get_llm()`` dependency instead):

        client = OllamaClient(settings)
        response = await client.generate("Summarise this policy.")
        print(response.text)
    """

    # ------------------------------------------------------------------
    # Infrastructure / Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def health(self, *, request_id: str | None = None) -> HealthResponse:
        """
        Check whether the LLM backend is reachable and responsive.

        Returns a ``HealthResponse`` regardless of backend state; it is
        the caller's responsibility to inspect ``HealthResponse.status``.

        Raises
        ------
        LLMConnectionError
            If the backend cannot be reached at all (e.g. server is down).
        """

    # ------------------------------------------------------------------
    # Core Generation
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        """
        Send a single prompt and return the model's completion.

        This is a stateless, single-turn call — no conversation history.
        Use ``chat`` for multi-turn conversations.

        Parameters
        ----------
        prompt :
            The full prompt text.  System instructions should be embedded
            directly in the prompt or passed via ``chat`` messages.
        temperature :
            Overrides ``OLLAMA_TEMPERATURE`` for this call only.
            ``None`` uses the client's configured default.
        seed :
            For deterministic outputs (testing, reproducible extractions).
        max_tokens :
            Hard cap on generated tokens.  ``None`` uses configured default.

        Returns
        -------
        GenerateResponse
            Contains ``text``, token counts, latency, and request_id.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        """
        Send a conversation history and return the assistant's next turn.

        Supports multi-turn conversations, system messages, and tool-call
        scaffolding.

        Parameters
        ----------
        messages :
            Ordered list of ``ChatMessage`` objects.  The last message
            should typically have role ``MessageRole.USER``.
        temperature :
            Per-call temperature override.
        seed :
            For deterministic outputs.
        max_tokens :
            Token limit for the assistant reply.

        Returns
        -------
        ChatResponse
            Contains the assistant ``ChatMessage``, token counts, and latency.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """

    # ------------------------------------------------------------------
    # Structured Extraction
    # ------------------------------------------------------------------

    @abstractmethod
    async def extract_json(
        self,
        prompt: str,
        *,
        schema_hint: dict[str, Any] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        request_id: str | None = None,
    ) -> ExtractionResponse:
        """
        Ask the model to produce a JSON object and parse it into a dict.

        The implementation MUST:
          1. Append a JSON instruction to the prompt if not already present.
          2. Strip markdown fences from the response.
          3. Attempt heuristic repair of common JSON errors.
          4. Validate the result with ``json.loads``.
          5. Raise ``LLMJSONError`` if still invalid after repair.

        Parameters
        ----------
        prompt :
            Task description.  The implementation will append JSON output
            instructions automatically.
        schema_hint :
            Optional dict describing the expected output shape.  When
            provided, it is embedded in the prompt to guide the model.
        temperature :
            Defaults to 0.0 for extraction (deterministic preferred).
        seed :
            Seed for reproducibility.

        Returns
        -------
        ExtractionResponse
            Contains ``data`` (parsed dict) and ``raw_output`` (original text).

        Raises
        ------
        LLMJSONError
            When the model output cannot be parsed as JSON after all repair
            attempts.  NOT retried automatically.
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """

    # ------------------------------------------------------------------
    # Semantic Utilities
    # ------------------------------------------------------------------

    @abstractmethod
    async def verify(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        """
        Submit a verification / fact-checking prompt and return the result.

        Verification prompts are typically structured as:

            "Given the context below, does the following statement hold?
             Answer with 'supported', 'refuted', or 'unknown' and explain."

        The implementation may apply specific temperature defaults suitable
        for conservative, deterministic verification.

        Parameters
        ----------
        prompt :
            The fully-formed verification prompt (context + claim + instruction).
        temperature :
            Defaults to low value (0.1) for conservative verification.

        Returns
        -------
        GenerateResponse
            The model's verification verdict as text.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """

    @abstractmethod
    async def summarize(
        self,
        text: str,
        *,
        max_words: int = 150,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        """
        Produce a concise summary of the given text.

        Parameters
        ----------
        text :
            The document or passage to summarise.  Very long texts should
            be pre-chunked by the caller.
        max_words :
            Approximate word-count target for the summary.
        temperature :
            Defaults to moderate value (0.3) for coherent, creative summaries.

        Returns
        -------
        GenerateResponse
            ``text`` field contains the summary.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """
