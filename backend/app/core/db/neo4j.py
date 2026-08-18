"""
Neo4j Connection Manager
========================

Provides an async Neo4j driver, schema initialisation (constraints + indexes),
and a health check.  Reads configuration from ``app.core.config.settings``.

Schema Init
-----------
``init_schema()`` is idempotent — safe to call at every startup.
Creates the following constraints and indexes (all ``IF NOT EXISTS``):

Constraints (UNIQUE on node_id for each node type)::

    :Scheme(node_id), :Ministry(node_id), :State(node_id),
    :Beneficiary(node_id), :EligibilityRule(node_id), :Benefit(node_id),
    :Clause(node_id), :Amendment(node_id)

Full-text index::

    scheme_name_ft on :Scheme(scheme_name)

Usage::

    from app.core.db.neo4j import neo4j_manager

    async with neo4j_manager.session() as session:
        await session.run("MATCH (s:Scheme) RETURN s LIMIT 1")
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from app.core.config import settings

logger = logging.getLogger(__name__)

# Node type labels that receive a UNIQUE constraint on node_id
_NODE_TYPES = [
    "Scheme",
    "Ministry",
    "State",
    "Beneficiary",
    "EligibilityRule",
    "Benefit",
    "Clause",
    "Amendment",
]


class Neo4jManager:
    """
    Lifecycle manager for the async Neo4j driver.

    Create once at application startup and reuse throughout the process
    lifetime.  Call ``close()`` at shutdown.
    """

    def __init__(self) -> None:
        self._driver: Any = None  # neo4j.AsyncDriver

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Create the async Neo4j driver.  Idempotent.
        """
        if self._driver is not None:
            return

        from neo4j import AsyncGraphDatabase  # local import — optional dep

        logger.info(
            "Neo4jManager: connecting | uri=%s | user=%s",
            settings.neo4j_uri,
            settings.neo4j_user,
        )
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
        logger.info("Neo4jManager: driver ready")

    async def close(self) -> None:
        """Close the driver and all its connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4jManager: driver closed")

    # ------------------------------------------------------------------
    # Session factory
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, None]:
        """
        Yield an async Neo4j session.
        """
        if self._driver is None:
            self.connect()
        async with self._driver.session() as neo4j_session:
            yield neo4j_session

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            if self._driver is None:
                self.connect()
            await self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning("Neo4jManager.health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """
        Create all constraints and indexes idempotently.

        Safe to call on every startup — uses ``IF NOT EXISTS``.
        """
        logger.info("Neo4jManager.init_schema() START")
        async with self.session() as neo4j_session:
            # Unique node_id constraints for each node type
            for label in _NODE_TYPES:
                constraint_name = f"{label.lower()}_node_id_unique"
                query = (
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.node_id IS UNIQUE"
                )
                try:
                    await neo4j_session.run(query)
                    logger.debug("Neo4j constraint ensured: %s", constraint_name)
                except Exception as exc:
                    logger.warning(
                        "Neo4j constraint creation skipped: %s | %s",
                        constraint_name,
                        exc,
                    )

            # Full-text index on scheme_name for search
            try:
                await neo4j_session.run(
                    "CREATE FULLTEXT INDEX scheme_name_ft IF NOT EXISTS "
                    "FOR (n:Scheme) ON EACH [n.scheme_name]"
                )
                logger.debug("Neo4j full-text index ensured: scheme_name_ft")
            except Exception as exc:
                logger.warning("Neo4j full-text index skipped: %s", exc)

            # Relationship index on ADMINISTERED_BY for fast lookup
            try:
                await neo4j_session.run(
                    "CREATE INDEX rel_admin_idx IF NOT EXISTS "
                    "FOR ()-[r:ADMINISTERED_BY]-() ON (r.source_document_id)"
                )
            except Exception as exc:
                logger.debug("Neo4j relationship index skipped: %s", exc)

        logger.info("Neo4jManager.init_schema() DONE")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

neo4j_manager = Neo4jManager()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_neo4j_session() -> AsyncGenerator[Any, None]:
    """FastAPI dependency that yields a Neo4j async session."""
    async with neo4j_manager.session() as session:
        yield session
