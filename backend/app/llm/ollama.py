"""
Ollama HTTP Client
==================

Concrete implementation of ``LLMClient`` that communicates with a locally-running
Ollama server via its REST API (no subprocesses, no shell commands).

API Endpoints Used
------------------
  POST /api/generate  — single-prompt completion
  POST /api/chat      — multi-turn conversation
  GET  /api/tags      — model list (used by health check)

Configuration
-------------
All tuneable parameters are read from environment variables via ``LLMSettings``:

  OLLAMA_URL          http://localhost:11434
  OLLAMA_MODEL        qwen2.5:7b
  OLLAMA_TIMEOUT      120 (seconds)
  OLLAMA_TEMPERATURE  0.7
  OLLAMA_TOP_P        0.9
  OLLAMA_NUM_PREDICT  -1 (unlimited)

Retry Policy
------------
Uses exponential backoff via ``tenacity`` for:
  - ``LLMConnectionError``  (server unreachable)
  - ``LLMTimeoutError``     (request exceeded timeout)

Does NOT retry:
  - ``LLMJSONError``        (malformed model output)
  - ``LLMResponseError``    (HTTP 4xx/5xx — indicates misconfiguration)

Structured Logging
------------------
Every request logs:
  - request_id (UUID4)
  - endpoint
  - model
  - prompt_size (chars)
  - response_size (chars)
  - latency_ms
  - token counts
  - error details

Design Notes
------------
- ``httpx.AsyncClient`` is used for non-blocking I/O.
- A single ``httpx.AsyncClient`` is created per ``OllamaClient`` instance and
  should be held for the lifetime of the application (created at startup,
  closed at shutdown).  This avoids TCP connection overhead on every request.
- ``stream=False`` is always set.  Streaming is left for a future milestone.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.llm.base import LLMClient
from app.llm.exceptions import (
    LLMConnectionError,
    LLMJSONError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ExtractionResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MessageRole,
)
from app.llm.parser import parse_llm_json
from app.llm.prompts.extraction import build_json_extraction_prompt
from app.llm.prompts.generation import build_summarize_prompt
from app.llm.prompts.verification import build_verification_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (loaded from environment via Pydantic Settings)
# ---------------------------------------------------------------------------


class LLMSettings:
    """
    Holds all configuration for the Ollama client.

    Reads from environment variables at construction time.  Providing explicit
    values in the constructor overrides env vars — useful in tests.

    Environment Variables
    ---------------------
    OLLAMA_URL          : Base URL of the Ollama server.
    OLLAMA_MODEL        : Default model name (e.g. 'qwen2.5:7b').
    OLLAMA_TIMEOUT      : Request timeout in seconds (float).
    OLLAMA_TEMPERATURE  : Default sampling temperature.
    OLLAMA_TOP_P        : Default nucleus sampling probability.
    OLLAMA_NUM_PREDICT  : Default max tokens to generate (-1 = unlimited).
    OLLAMA_MAX_RETRIES  : Maximum retry attempts for transient errors.
    OLLAMA_RETRY_MIN_WAIT : Minimum seconds between retries (exponential backoff).
    OLLAMA_RETRY_MAX_WAIT : Maximum seconds between retries.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
        max_retries: int | None = None,
        retry_min_wait: float | None = None,
        retry_max_wait: float | None = None,
    ) -> None:
        import os

        self.url: str = url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model: str = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout: float = timeout or float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.temperature: float = temperature or float(
            os.getenv("OLLAMA_TEMPERATURE", "0.7")
        )
        self.top_p: float = top_p or float(os.getenv("OLLAMA_TOP_P", "0.9"))
        self.num_predict: int = num_predict or int(
            os.getenv("OLLAMA_NUM_PREDICT", "-1")
        )
        self.max_retries: int = max_retries or int(
            os.getenv("OLLAMA_MAX_RETRIES", "3")
        )
        self.retry_min_wait: float = retry_min_wait or float(
            os.getenv("OLLAMA_RETRY_MIN_WAIT", "1.0")
        )
        self.retry_max_wait: float = retry_max_wait or float(
            os.getenv("OLLAMA_RETRY_MAX_WAIT", "30.0")
        )

    def __repr__(self) -> str:
        return (
            f"LLMSettings(url={self.url!r}, model={self.model!r}, "
            f"timeout={self.timeout}s, temperature={self.temperature}, "
            f"top_p={self.top_p}, num_predict={self.num_predict})"
        )


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


class OllamaClient(LLMClient):
    """
    Async HTTP client for the Ollama REST API.

    Implements the full ``LLMClient`` interface.

    Lifecycle
    ---------
    Create once at application startup and hold for the application lifetime:

        client = OllamaClient(settings)
        await client.aclose()  # call at shutdown

    Parameters
    ----------
    settings :
        ``LLMSettings`` instance.  When ``None``, a default instance is
        created from environment variables.

    Example
    -------
    ::

        settings = LLMSettings()
        client = OllamaClient(settings)
        response = await client.generate("What is AI?")
        print(response.text)
        await client.aclose()
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        self._http = httpx.AsyncClient(
            base_url=self._settings.url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._settings.timeout,
                write=30.0,
                pool=5.0,
            ),
            headers={"Content-Type": "application/json"},
        )
        logger.info("OllamaClient initialised: %s", self._settings)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """
        Close the underlying HTTP connection pool.

        Must be called at application shutdown to release file descriptors.
        """
        await self._http.aclose()
        logger.info("OllamaClient HTTP client closed.")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self, *, request_id: str | None = None) -> HealthResponse:
        """
        Ping Ollama's ``GET /api/tags`` endpoint to verify the server is running
        and retrieve the list of available models.

        Returns a ``HealthResponse`` with status 'healthy' or 'unhealthy'.
        """
        request_id = request_id or str(uuid.uuid4())
        start = time.monotonic()

        try:
            response = await self._http.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            latency_ms = self._elapsed_ms(start)

            models = [m.get("name", "") for m in data.get("models", [])]
            model_found = self._settings.model in models

            logger.info(
                "Health check OK | request_id=%s | latency_ms=%d | "
                "available_models=%s | target_model=%s | found=%s",
                request_id,
                latency_ms,
                models,
                self._settings.model,
                model_found,
            )

            return HealthResponse(
                status="healthy",
                model=self._settings.model,
                backend="ollama",
                latency_ms=latency_ms,
                details={
                    "available_models": models,
                    "target_model_found": model_found,
                    "url": self._settings.url,
                },
            )

        except httpx.ConnectError as exc:
            latency_ms = self._elapsed_ms(start)
            logger.error(
                "Health check FAILED — cannot connect | request_id=%s | error=%s",
                request_id,
                exc,
            )
            return HealthResponse(
                status="unhealthy",
                model=self._settings.model,
                backend="ollama",
                latency_ms=latency_ms,
                details={"error": str(exc), "url": self._settings.url},
            )
        except Exception as exc:
            latency_ms = self._elapsed_ms(start)
            logger.error(
                "Health check FAILED — unexpected error | request_id=%s | error=%s",
                request_id,
                exc,
            )
            return HealthResponse(
                status="unhealthy",
                model=self._settings.model,
                backend="ollama",
                latency_ms=latency_ms,
                details={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

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
        Call ``POST /api/generate`` with retry on transient errors.
        """
        request_id = request_id or str(uuid.uuid4())
        request = GenerateRequest(
            prompt=prompt,
            model=self._settings.model,
            temperature=temperature if temperature is not None else self._settings.temperature,
            top_p=self._settings.top_p,
            seed=seed,
            max_tokens=max_tokens if max_tokens is not None else (None if self._settings.num_predict == -1 else self._settings.num_predict),
            stream=False,
        )

        logger.info(
            "generate() | request_id=%s | model=%s | prompt_chars=%d | temperature=%s",
            request_id,
            request.model,
            len(prompt),
            request.temperature,
        )

        start = time.monotonic()
        raw_response = await self._post_with_retry(
            endpoint="/api/generate",
            payload=self._build_generate_payload(request),
            request_id=request_id,
        )
        latency_ms = self._elapsed_ms(start)

        text = raw_response.get("response", "")
        prompt_tokens = raw_response.get("prompt_eval_count", 0) or 0
        completion_tokens = raw_response.get("eval_count", 0) or 0

        logger.info(
            "generate() OK | request_id=%s | latency_ms=%d | "
            "prompt_tokens=%d | completion_tokens=%d | response_chars=%d",
            request_id,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            len(text),
        )

        return GenerateResponse(
            text=text,
            model=self._settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

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
        Call ``POST /api/chat`` with retry on transient errors.
        """
        request_id = request_id or str(uuid.uuid4())
        request = ChatRequest(
            messages=messages,
            model=self._settings.model,
            temperature=temperature if temperature is not None else self._settings.temperature,
            top_p=self._settings.top_p,
            seed=seed,
            max_tokens=max_tokens if max_tokens is not None else (None if self._settings.num_predict == -1 else self._settings.num_predict),
            stream=False,
        )

        total_prompt_chars = sum(len(m.content) for m in messages)
        logger.info(
            "chat() | request_id=%s | model=%s | n_messages=%d | total_chars=%d",
            request_id,
            request.model,
            len(messages),
            total_prompt_chars,
        )

        start = time.monotonic()
        raw_response = await self._post_with_retry(
            endpoint="/api/chat",
            payload=self._build_chat_payload(request),
            request_id=request_id,
        )
        latency_ms = self._elapsed_ms(start)

        msg_data = raw_response.get("message", {})
        assistant_message = ChatMessage(
            role=MessageRole(msg_data.get("role", "assistant")),
            content=msg_data.get("content", ""),
        )
        prompt_tokens = raw_response.get("prompt_eval_count", 0) or 0
        completion_tokens = raw_response.get("eval_count", 0) or 0

        logger.info(
            "chat() OK | request_id=%s | latency_ms=%d | "
            "prompt_tokens=%d | completion_tokens=%d | response_chars=%d",
            request_id,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            len(assistant_message.content),
        )

        return ChatResponse(
            message=assistant_message,
            model=self._settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Extract JSON
    # ------------------------------------------------------------------

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
        Ask the model to produce JSON output and parse the result.

        Uses a low default temperature (0.0) for deterministic extraction.
        Applies the full JSON repair pipeline from ``parser.py``.

        Raises
        ------
        LLMJSONError
            When the output cannot be parsed as JSON after all repair attempts.
            This exception is NOT retried.
        """
        request_id = request_id or str(uuid.uuid4())
        # Default to deterministic temperature for extraction
        effective_temp = temperature if temperature is not None else 0.0

        # Build an extraction-specific prompt that instructs JSON output
        extraction_prompt = build_json_extraction_prompt(
            base_prompt=prompt,
            schema_hint=schema_hint,
        )

        logger.info(
            "extract_json() | request_id=%s | model=%s | prompt_chars=%d | temperature=%s",
            request_id,
            self._settings.model,
            len(extraction_prompt),
            effective_temp,
        )

        start = time.monotonic()
        # Call generate — JSON errors should NOT be retried so we do not wrap
        # the parse step in the retry logic.
        raw_response = await self._post_with_retry(
            endpoint="/api/generate",
            payload=self._build_generate_payload(
                GenerateRequest(
                    prompt=extraction_prompt,
                    model=self._settings.model,
                    temperature=effective_temp,
                    top_p=self._settings.top_p,
                    seed=seed,
                    max_tokens=None if self._settings.num_predict == -1 else self._settings.num_predict,
                    stream=False,
                )
            ),
            request_id=request_id,
        )
        latency_ms = self._elapsed_ms(start)

        raw_text = raw_response.get("response", "")

        logger.debug(
            "extract_json() raw output | request_id=%s | raw=%.300s",
            request_id,
            raw_text,
        )

        # LLMJSONError is raised here — intentionally NOT inside the retry loop
        parsed = parse_llm_json(raw_text, request_id=request_id)

        logger.info(
            "extract_json() OK | request_id=%s | latency_ms=%d | keys=%s",
            request_id,
            latency_ms,
            list(parsed.keys()),
        )

        return ExtractionResponse(
            data=parsed,
            raw_output=raw_text,
            model=self._settings.model,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    async def verify(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        """
        Submit a verification prompt with a conservative default temperature (0.1).
        """
        effective_temp = temperature if temperature is not None else 0.1
        full_prompt = build_verification_prompt(claim_and_context=prompt)
        return await self.generate(full_prompt, temperature=effective_temp, request_id=request_id)

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    async def summarize(
        self,
        text: str,
        *,
        max_words: int = 150,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        """
        Summarise ``text`` in approximately ``max_words`` words.

        Uses a moderate default temperature (0.3) for coherent summaries.
        """
        effective_temp = temperature if temperature is not None else 0.3
        full_prompt = build_summarize_prompt(text=text, max_words=max_words)
        return await self.generate(full_prompt, temperature=effective_temp, request_id=request_id)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """
        POST to ``endpoint`` with exponential backoff retry on transient errors.

        Retries on:   LLMConnectionError, LLMTimeoutError
        No retry on:  LLMJSONError, LLMResponseError

        Parameters
        ----------
        endpoint :
            Relative path (e.g. '/api/generate').
        payload :
            JSON-serialisable request body.
        request_id :
            Correlation ID passed into exceptions.

        Returns
        -------
        dict[str, Any]
            The parsed JSON response body.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """
        settings = self._settings
        attempt_number = 0

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((LLMConnectionError, LLMTimeoutError)),
                stop=stop_after_attempt(settings.max_retries),
                wait=wait_exponential(
                    multiplier=1,
                    min=settings.retry_min_wait,
                    max=settings.retry_max_wait,
                ),
                reraise=True,
            ):
                with attempt:
                    attempt_number += 1
                    if attempt_number > 1:
                        logger.warning(
                            "Retry attempt %d/%d | request_id=%s | endpoint=%s",
                            attempt_number,
                            settings.max_retries,
                            request_id,
                            endpoint,
                        )
                    return await self._http_post(
                        endpoint=endpoint,
                        payload=payload,
                        request_id=request_id,
                    )
        except RetryError as exc:
            # tenacity wraps the last exception; unwrap it
            raise exc.last_attempt.exception() from exc  # type: ignore[union-attr]

        # Unreachable but satisfies type checker
        raise LLMConnectionError(  # pragma: no cover
            "Retry loop exited without returning or raising.",
            request_id=request_id,
        )

    async def _http_post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """
        Execute a single HTTP POST without retry logic.

        Maps httpx exceptions to the LLM exception hierarchy.
        """
        try:
            response = await self._http.post(endpoint, json=payload)
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._settings.url}: {exc}",
                request_id=request_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Request to {endpoint} timed out after {self._settings.timeout}s: {exc}",
                timeout_seconds=self._settings.timeout,
                request_id=request_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"HTTP error communicating with Ollama: {exc}",
                request_id=request_id,
            ) from exc

        # Map HTTP error status codes
        if response.status_code >= 400:
            raise LLMResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                request_id=request_id,
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Ollama response is not valid JSON: {exc}",
                request_id=request_id,
            ) from exc

    @staticmethod
    def _build_generate_payload(request: GenerateRequest) -> dict[str, Any]:
        """Serialise a ``GenerateRequest`` into an Ollama ``/api/generate`` payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        else:
            options["num_predict"] = -1
        if options:
            payload["options"] = options
        return payload

    @staticmethod
    def _build_chat_payload(request: ChatRequest) -> dict[str, Any]:
        """Serialise a ``ChatRequest`` into an Ollama ``/api/chat`` payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            "stream": False,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        else:
            options["num_predict"] = -1
        if options:
            payload["options"] = options
        return payload

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        """Compute milliseconds elapsed since ``start`` (from time.monotonic())."""
        return int((time.monotonic() - start) * 1000)
