"""
Pydantic V2 Data Models for the LLM Layer
==========================================

These models are the single source of truth for all data shapes that flow
through the LLM subsystem.  They are shared between:

  - OllamaClient  (serialise requests, deserialise responses)
  - AIService     (method signatures)
  - FastAPI routes (future – Week 2+)
  - Unit tests    (fixture construction)

Design Decisions
----------------
- All models inherit from ``BaseModel`` with ``model_config = ConfigDict(frozen=True)``
  so instances are immutable after construction (safe to cache, share across threads).
- ``ChatMessage`` uses a ``MessageRole`` enum to prevent typos like "System" vs "system".
- ``ExtractionResponse`` carries both the parsed dict AND the raw string so callers
  can do their own post-processing without a second LLM call.
- Latency is expressed in milliseconds (int) for easy storage in monitoring systems.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    """
    Role of a participant in a chat conversation.

    Values must match what Ollama (and OpenAI-compatible APIs) expect.
    Using an enum instead of a bare str prevents silent mismatches between
    provider strings (e.g. "System" vs "system").
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """
    Parameters for a single text-completion (non-chat) request.

    Fields map 1-to-1 to the Ollama ``/api/generate`` payload so this model
    can be serialised directly.  Optional fields use the Ollama defaults when
    omitted.

    See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="The prompt text sent to the model.")
    model: str = Field(..., description="Ollama model identifier, e.g. 'qwen2.5:7b'.")
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher → more creative. None = model default.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass. None = model default.",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for deterministic generation. None = non-deterministic.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate. None = model default (unlimited).",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response token-by-token. Always False in Week 1.",
    )


class ChatMessage(BaseModel):
    """
    A single message in a multi-turn conversation.

    Compatible with OpenAI chat format so switching providers requires
    no data model changes.
    """

    model_config = ConfigDict(frozen=True)

    role: MessageRole = Field(..., description="Who produced this message.")
    content: str = Field(..., description="Text content of the message.")


class ChatRequest(BaseModel):
    """
    Parameters for a multi-turn chat completion request.

    Maps to the Ollama ``/api/chat`` payload.
    """

    model_config = ConfigDict(frozen=True)

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Ordered conversation history, ending with the latest user message.",
    )
    model: str = Field(..., description="Ollama model identifier.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = Field(default=None)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class GenerateResponse(BaseModel):
    """
    Normalised response from a generate call.

    Decoupled from Ollama's raw JSON so switching providers does not
    require touching business logic.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="The model's generated text.")
    model: str = Field(..., description="The model that produced this response.")
    prompt_tokens: int = Field(
        default=0, ge=0, description="Number of tokens in the prompt (from Ollama eval_count)."
    )
    completion_tokens: int = Field(
        default=0, ge=0, description="Number of tokens generated (from Ollama eval_count)."
    )
    latency_ms: int = Field(
        default=0, ge=0, description="Total wall-clock latency in milliseconds."
    )
    request_id: str = Field(..., description="Correlation ID for distributed tracing.")


class ChatResponse(BaseModel):
    """
    Normalised response from a chat call.
    """

    model_config = ConfigDict(frozen=True)

    message: ChatMessage = Field(..., description="The assistant's reply.")
    model: str = Field(..., description="The model that produced this response.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    request_id: str = Field(..., description="Correlation ID for distributed tracing.")


class HealthResponse(BaseModel):
    """
    Result of an LLM backend health check.

    Used by the ``/health`` FastAPI endpoint and by AIService.health().
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="'healthy' or 'unhealthy'.")
    model: str = Field(..., description="Model name reported by the backend.")
    backend: str = Field(..., description="Backend identifier, e.g. 'ollama'.")
    latency_ms: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional backend-specific metadata.",
    )


class ExtractionResponse(BaseModel):
    """
    Result of a structured JSON extraction call.

    Carries both the parsed Python dict AND the raw model string so callers
    can perform secondary validation without another round-trip to the LLM.
    """

    model_config = ConfigDict(frozen=True)

    data: dict[str, Any] = Field(
        ..., description="The parsed, validated JSON object from the model."
    )
    raw_output: str = Field(
        ..., description="The original text returned by the model before parsing."
    )
    model: str = Field(..., description="Model that performed the extraction.")
    latency_ms: int = Field(default=0, ge=0)
    request_id: str = Field(..., description="Correlation ID.")
