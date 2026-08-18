"""
PolicyIntel AI — FastAPI Application
======================================

Entry point for the backend service.

Startup sequence (lifespan)
---------------------------
1. Initialise PostgreSQL (create tables)
2. Initialise Neo4j (create constraints + indexes)
3. Initialise Qdrant (create collection + payload indexes)

Shutdown sequence
-----------------
1. Close all database connections
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.init_db import close_all, init_all
from app.llm.dependency import close_llm_client, initialise_llm_client

# Configure structured logging before anything else
configure_logging(
    level="DEBUG" if settings.debug else "INFO",
    json_logs=settings.environment == "production",
)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: database init on startup, graceful close on shutdown."""
    logger.info("PolicyIntel AI starting up...")
    await init_all()
    await initialise_llm_client()  # Must be called before any request uses get_ai_service()
    yield
    logger.info("PolicyIntel AI shutting down...")
    await close_llm_client()  # Releases httpx connection pool
    await close_all()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description=(
        "Enterprise Policy Intelligence Engine — "
        "Explainable AI & Knowledge Graph for Government Policy Documents."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — restrict in production; allow_origins=["*"] with allow_credentials=True
# is rejected by browsers (CORS spec §3.2.2). Use explicit origin list.
_CORS_ORIGINS = (
    ["*"]
    if settings.environment == "development"
    else ["http://localhost:3000"]  # Override via CORS_ORIGINS env var in production
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=settings.environment != "development",
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.api.v1.router import api_router  # noqa: E402

app.include_router(api_router, prefix=settings.api_v1_str)

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "service": settings.project_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs": "/docs",
    }
