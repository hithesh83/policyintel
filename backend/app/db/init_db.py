"""
Database Initialisation
=======================

Orchestrates startup initialisation for all three databases:

1. **PostgreSQL** — creates all ORM tables (``create_all``).
2. **Neo4j** — creates constraints and full-text indexes.
3. **Qdrant** — creates the vector collection and payload indexes.

All operations are idempotent — safe to call on every application startup.

Usage (called from FastAPI lifespan)::

    from app.db.init_db import init_all, close_all

    async def lifespan(app):
        await init_all()
        yield
        await close_all()
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.core.db.neo4j import neo4j_manager
from app.core.db.postgres import postgres_manager
from app.core.db.qdrant import qdrant_manager

logger = logging.getLogger(__name__)


async def init_postgres() -> None:
    """
    Connect to PostgreSQL and create all ORM tables.

    Uses SQLAlchemy ``create_all`` — safe when tables already exist.
    For production, Alembic should manage migrations instead.
    """
    logger.info("init_postgres() START")
    postgres_manager.connect()

    # Import all models so their metadata is registered before create_all
    from app.models.document import Base  # noqa: F401 (registers Document, DocumentUpload)
    from app.models.feedback import Feedback  # noqa: F401
    from app.models.query_log import QueryLog  # noqa: F401

    async with postgres_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("init_postgres() DONE — all tables created/verified")


async def init_neo4j() -> None:
    """
    Connect to Neo4j and create schema constraints and indexes.
    """
    logger.info("init_neo4j() START")
    neo4j_manager.connect()
    await neo4j_manager.init_schema()
    logger.info("init_neo4j() DONE")


async def init_qdrant() -> None:
    """
    Connect to Qdrant and ensure the policy_chunks collection exists.
    """
    logger.info("init_qdrant() START")
    qdrant_manager.connect()
    await qdrant_manager.ensure_collection()
    logger.info("init_qdrant() DONE")


async def init_all() -> None:
    """
    Run all database initialisations concurrently.

    Called once from the FastAPI lifespan at application startup.
    Failures are logged but do NOT crash the application — partial
    database availability is better than complete startup failure.
    """
    import asyncio

    logger.info("=" * 60)
    logger.info("PolicyIntel AI — Database Initialisation")
    logger.info("=" * 60)

    results = await asyncio.gather(
        _safe_init("PostgreSQL", init_postgres),
        _safe_init("Neo4j", init_neo4j),
        _safe_init("Qdrant", init_qdrant),
        return_exceptions=True,
    )

    for name, result in zip(["PostgreSQL", "Neo4j", "Qdrant"], results):
        if isinstance(result, Exception):
            logger.error("init_all: %s initialisation FAILED: %s", name, result)
        else:
            logger.info("init_all: %s ✓", name)

    logger.info("=" * 60)


async def close_all() -> None:
    """
    Gracefully close all database connections.

    Called from the FastAPI lifespan on application shutdown.
    """
    import asyncio

    logger.info("Closing all database connections...")
    await asyncio.gather(
        postgres_manager.close(),
        neo4j_manager.close(),
        qdrant_manager.close(),
        return_exceptions=True,
    )
    logger.info("All database connections closed.")


# ---------------------------------------------------------------------------
# Health checks (used by /health endpoint)
# ---------------------------------------------------------------------------


async def health_check_all() -> dict[str, bool]:
    """
    Run health checks on all three databases concurrently.

    Returns a mapping of database name → is_healthy.
    """
    import asyncio

    pg_ok, neo4j_ok, qdrant_ok = await asyncio.gather(
        postgres_manager.health_check(),
        neo4j_manager.health_check(),
        qdrant_manager.health_check(),
        return_exceptions=True,
    )

    return {
        "postgres": pg_ok is True,
        "neo4j": neo4j_ok is True,
        "qdrant": qdrant_ok is True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_init(name: str, fn) -> None:
    """Run an init function, catching and logging exceptions."""
    try:
        await fn()
    except Exception as exc:
        logger.error("%s init failed: %s", name, exc)
        raise
