"""
LLM Exception Hierarchy
=======================

All exceptions raised by the LLM subsystem are subclasses of ``LLMError``.
This lets callers catch at any level of specificity:

    try:
        result = await client.generate(prompt)
    except LLMTimeoutError:
        # handle timeout specifically
        ...
    except LLMError:
        # catch-all for any LLM problem
        ...

Retry Policy (enforced by OllamaClient):
-----------------------------------------
RETRYABLE  → LLMConnectionError, LLMTimeoutError
NOT RETRIED → LLMJSONError, LLMResponseError, LLMError (base)
"""

from __future__ import annotations


class LLMError(Exception):
    """
    Base exception for all LLM-layer failures.

    Attributes
    ----------
    message : str
        Human-readable error description.
    request_id : str | None
        Correlation ID of the request that caused this error, if available.
    """

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"request_id={self.request_id!r})"
        )


class LLMConnectionError(LLMError):
    """
    Raised when the LLM backend cannot be reached.

    Causes
    ------
    - Ollama server not running
    - Wrong OLLAMA_URL in .env
    - Network partition
    - DNS resolution failure

    This error IS retryable with exponential backoff.
    """


class LLMTimeoutError(LLMError):
    """
    Raised when an LLM request exceeds the configured timeout.

    Causes
    ------
    - Model too slow for the given prompt size
    - OLLAMA_TIMEOUT too aggressive
    - GPU resource exhaustion

    This error IS retryable (the model may recover on next attempt).

    Attributes
    ----------
    timeout_seconds : float | None
        The timeout value (in seconds) that was exceeded.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"request_id={self.request_id!r})"
        )


class LLMJSONError(LLMError):
    """
    Raised when the model returns output that cannot be parsed as valid JSON,
    even after repair attempts.

    Causes
    ------
    - Model did not follow the JSON instruction
    - Model returned pure prose instead of structured output
    - Malformed JSON that repair heuristics cannot fix

    This error is NOT retried (retrying with the same prompt will likely
    reproduce the same malformed output; the caller should modify the prompt).

    Attributes
    ----------
    raw_output : str | None
        The raw text the model returned before parse attempts.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.raw_output = raw_output

    def __repr__(self) -> str:
        preview = (self.raw_output or "")[:120]
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"raw_output_preview={preview!r}, "
            f"request_id={self.request_id!r})"
        )


class LLMResponseError(LLMError):
    """
    Raised when the model returns an HTTP error or an unexpected API response.

    Causes
    ------
    - Ollama returns 4xx / 5xx status codes
    - Response body is missing expected fields
    - Model name not found in Ollama

    This error is NOT retried automatically because it usually indicates
    a configuration problem rather than a transient failure.

    Attributes
    ----------
    status_code : int | None
        HTTP status code from the upstream API, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.status_code = status_code

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code!r}, "
            f"request_id={self.request_id!r})"
        )
