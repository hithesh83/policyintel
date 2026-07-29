"""
FastAPI Dependency Injection for the LLM Layer
===============================================

Provides ``get_llm()`` and ``get_ai_service()`` as FastAPI ``Depends``-compatible
async generators.

A single ``OllamaClient`` (and a single ``AIService``) is created at application
startup and shared across all requests.  This is intentional:

  - ``httpx.AsyncClient`` maintains a connection pool — creating one per request
    would destroy this optimisation.
  - ``AIService`` is stateless — it is safe to share across concurrent requests.
  - The singleton pattern is enforced through ``functools.lru_cache`` on the
    factory, combined with FastAPI's lifespan context for cleanup.

Lifecycle
---------
The application must call ``initialise_llm_client()`` at startup (inside the
FastAPI ``lifespan`` context manager) and ``close_llm_client()`` at shutdown.
This is already wired in the application's ``lifespan`` function.

Usage in route handlers
-----------------------
::

    from fastapi import Depends
    from app.llm.dependency import get_ai_service
    from app.services.ai_service import AIService

    @router.post("/extract")
    async def extract_endpoint(
        payload: ExtractionPayload,
        ai: AIService = Depends(get_ai_service),
    ):
        result = await ai.extract_json(payload.text)
        return result.data

Usage outside FastAPI (e.g., pipeline scripts, tests)
------------------------------------------------------
::

    from app.llm.dependency import get_ai_service_instance
    from app.llm.ollama import LLMSettings

    settings = LLMSettings(temperature=0.0)
    ai = get_ai_service_instance(settings=settings)
    result = await ai.summarize("Long policy text...")
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from app.llm.base import LLMClient
from app.llm.ollama import LLMSettings, OllamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_llm_client: OllamaClient | None = None
_ai_service = None  # AIService | None — imported lazily to avoid circular import


# ---------------------------------------------------------------------------
# Lifecycle Management
# ---------------------------------------------------------------------------


async def initialise_llm_client(settings: LLMSettings | None = None) -> None:
    """
    Create the global ``OllamaClient`` singleton.

    Call this once from the FastAPI ``lifespan`` startup section.

    Parameters
    ----------
    settings :
        ``LLMSettings`` instance.  Defaults to env-var-driven configuration.

    Raises
    ------
    RuntimeError
        If called more than once without calling ``close_llm_client`` first.
    """
    global _llm_client, _ai_service
    from app.services.ai_service import AIService

    if _llm_client is not None:
        logger.warning(
            "initialise_llm_client() called but client already exists. "
            "Call close_llm_client() first if you intend to reinitialise."
        )
        return

    resolved_settings = settings or LLMSettings()
    _llm_client = OllamaClient(resolved_settings)
    _ai_service = AIService(client=_llm_client)
    logger.info("LLM client and AIService initialised: %s", resolved_settings)


async def close_llm_client() -> None:
    """
    Close the global ``OllamaClient`` and release its HTTP connection pool.

    Call this from the FastAPI ``lifespan`` shutdown section.
    Safe to call even if ``initialise_llm_client`` was never called.
    """
    global _llm_client, _ai_service

    if _llm_client is not None:
        await _llm_client.aclose()
        _llm_client = None
        _ai_service = None
        logger.info("LLM client closed and resources released.")
    else:
        logger.debug("close_llm_client() called but no client was initialised.")


# ---------------------------------------------------------------------------
# FastAPI Dependency Functions
# ---------------------------------------------------------------------------


async def get_llm() -> AsyncGenerator[LLMClient, None]:
    """
    FastAPI dependency that yields the global ``LLMClient``.

    Yields the same singleton on every call — no per-request allocation.
    Use ``Depends(get_llm)`` in route handlers that need raw LLM access.

    Usage
    -----
    ::

        @router.get("/health")
        async def llm_health(client: LLMClient = Depends(get_llm)):
            return await client.health()

    Raises
    ------
    RuntimeError
        If ``initialise_llm_client()`` has not been called before the first request.
    """
    if _llm_client is None:
        raise RuntimeError(
            "LLM client has not been initialised. "
            "Call initialise_llm_client() in the FastAPI lifespan startup."
        )
    yield _llm_client


async def get_ai_service():
    """
    FastAPI dependency that yields the global ``AIService`` instance.

    This is the PREFERRED dependency for all application code.
    Route handlers should depend on ``AIService``, not on ``LLMClient``
    directly, because ``AIService`` provides higher-level operations.

    Usage
    -----
    ::

        from app.services.ai_service import AIService

        @router.post("/query")
        async def answer_query(
            query: str,
            ai: AIService = Depends(get_ai_service),
        ):
            return await ai.answer_policy_question(query=query, context=[...])

    Yields
    ------
    AIService
        The application-wide singleton.

    Raises
    ------
    RuntimeError
        If ``initialise_llm_client()`` has not been called.
    """
    if _ai_service is None:
        raise RuntimeError(
            "AIService has not been initialised. "
            "Call initialise_llm_client() in the FastAPI lifespan startup."
        )
    yield _ai_service


# ---------------------------------------------------------------------------
# Direct Instance Access (for non-FastAPI use)
# ---------------------------------------------------------------------------


def get_llm_instance(settings: LLMSettings | None = None) -> OllamaClient:
    """
    Return a **new** (non-singleton) ``OllamaClient`` instance.

    Use this in scripts, CLI tools, or tests where you do not have a running
    FastAPI application.  The caller is responsible for calling ``.aclose()``
    when done.

    Parameters
    ----------
    settings :
        ``LLMSettings`` instance.  Defaults to env-var-driven configuration.

    Returns
    -------
    OllamaClient
        A fresh client instance (not the application singleton).

    Example
    -------
    ::

        import asyncio
        from app.llm.dependency import get_llm_instance

        async def main():
            client = get_llm_instance()
            try:
                health = await client.health()
                print(health)
            finally:
                await client.aclose()

        asyncio.run(main())
    """
    return OllamaClient(settings or LLMSettings())


def get_ai_service_instance(settings: LLMSettings | None = None):
    """
    Return a **new** (non-singleton) ``AIService`` instance.

    Use this in scripts, CLI tools, or tests.  The caller must call
    ``await ai_service.client.aclose()`` when done to release resources.

    Parameters
    ----------
    settings :
        ``LLMSettings`` instance.  Defaults to env-var-driven configuration.

    Returns
    -------
    AIService
        A fresh AIService wrapping a fresh OllamaClient.
    """
    from app.services.ai_service import AIService

    client = get_llm_instance(settings)
    return AIService(client=client)
