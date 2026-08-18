"""
PostgreSQL Connection Manager
=============================

Provides an async SQLAlchemy engine, session factory, and health check.
Reads configuration from ``app.core.config.settings``.

Usage (FastAPI DI)::

    from app.core.db.postgres import get_db_session

    @router.post("/")
    async def endpoint(db: AsyncSession = Depends(get_db_session)):
        ...

Usage (manual)::

    from app.core.db.postgres import postgres_manager

    async with postgres_manager.session() as db:
        result = await db.execute(select(Document))
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class PostgresManager:
    """
    Lifecycle manager for the async SQLAlchemy connection pool.

    Create once at application startup and reuse throughout the process
    lifetime.  Call ``close()`` at shutdown.
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Create the async engine and session factory.

        Called once at startup — idempotent.
        """
        if self._engine is not None:
            return

        logger.info(
            "PostgresManager: creating engine | url=%s",
            _redact_url(settings.database_url),
        )
        self._engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        logger.info("PostgresManager: engine ready")

    async def close(self) -> None:
        """Dispose the connection pool gracefully."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("PostgresManager: engine disposed")

    # ------------------------------------------------------------------
    # Session factory
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Yield an ``AsyncSession`` as an async context manager.

        Commits on clean exit, rolls back on exception.
        """
        if self._session_factory is None:
            self.connect()

        assert self._session_factory is not None
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Return True if PostgreSQL is reachable and responsive.
        """
        try:
            async with self.session() as db:
                await db.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("PostgresManager.health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Engine accessor
    # ------------------------------------------------------------------

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self.connect()
        assert self._engine is not None
        return self._engine


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

postgres_manager = PostgresManager()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Commits on clean exit, rolls back on exception, always closes.
    """
    async with postgres_manager.session() as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_url(url: str) -> str:
    """Redact the password from a database URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url
