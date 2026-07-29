"""
PolicyIntel AI - LLM Layer

Public re-exports so callers can do:

    from app.llm import LLMClient, OllamaClient, AIService
    from app.llm import get_llm, get_ai_service
    from app.llm import LLMError, LLMConnectionError, LLMTimeoutError

This module is the ONLY entry-point into the LLM subsystem.
Nothing outside this package should import from sub-modules directly.
"""

from app.llm.base import LLMClient
from app.llm.ollama import OllamaClient
from app.llm.models import (
    GenerateRequest,
    GenerateResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ExtractionResponse,
    MessageRole,
)
from app.llm.exceptions import (
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMJSONError,
    LLMResponseError,
)
from app.llm.dependency import get_llm, get_ai_service

__all__ = [
    # Interfaces
    "LLMClient",
    # Implementations
    "OllamaClient",
    # Pydantic Models
    "GenerateRequest",
    "GenerateResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "ExtractionResponse",
    "MessageRole",
    # Exceptions
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMJSONError",
    "LLMResponseError",
    # FastAPI DI
    "get_llm",
    "get_ai_service",
]
