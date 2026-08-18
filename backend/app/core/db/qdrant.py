"""
Qdrant Connection Manager
=========================

Provides an ``AsyncQdrantClient``, collection initialisation (with payload
indexes), and a health check.  Reads configuration from
``app.core.config.settings``.

Collection Config
-----------------
- **Collection name**: from ``settings.qdrant_collection_name``
- **Vector size**: 1024 (matches ``indexer._EMBEDDING_DIM``)
- **Distance**: Cosine
- **Payload indexes** (for efficient metadata filtering):
  - ``document_id``            (keyword)
  - ``hierarchy_level``        (keyword)
  - ``scheme_name``            (keyword)
  - ``ministry``               (keyword)
  - ``policy_type``            (keyword)
  - ``has_eligibility_criteria`` (bool)
  - ``is_direct_benefit_transfer`` (bool)
  - ``page_number``            (integer)

Usage::

    from app.core.db.qdrant import qdrant_manager

    client = await qdrant_manager.get_client()
    await client.upsert(collection_name=..., points=[...])
"""

from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Must match indexer._EMBEDDING_DIM — changing requires a collection rebuild.
_VECTOR_SIZE = 1024

# Payload fields to create keyword indexes on (for metadata filtering)
_KEYWORD_INDEXES = [
    "document_id",
    "chunk_id",
    "hierarchy_level",
    "scheme_name",
    "ministry",
    "policy_type",
    "geographic_scope",
    "content_type",
    "topic",
]

_BOOL_INDEXES = [
    "has_eligibility_criteria",
    "has_procedure_steps",
    "is_direct_benefit_transfer",
    "has_amendment",
]

_INT_INDEXES = [
    "page_number",
    "word_count",
]


class QdrantManager:
    """
    Lifecycle manager for the async Qdrant client.

    Create once at startup; call ``close()`` at shutdown.
    """

    def __init__(self) -> None:
        self._client: object | None = None  # AsyncQdrantClient

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create the async Qdrant client.  Idempotent."""
        if self._client is not None:
            return

        from qdrant_client import AsyncQdrantClient  # local import

        logger.info(
            "QdrantManager: connecting | host=%s | port=%d",
            settings.qdrant_host,
            settings.qdrant_port,
        )
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("QdrantManager: client ready")

    async def close(self) -> None:
        """Close the Qdrant HTTP connections."""
        if self._client:
            try:
                await self._client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._client = None
            logger.info("QdrantManager: client closed")

    # ------------------------------------------------------------------
    # Client accessor
    # ------------------------------------------------------------------

    def get_client(self):  # type: ignore[return]
        """Return the ``AsyncQdrantClient`` singleton."""
        if self._client is None:
            self.connect()
        return self._client

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            client = self.get_client()
            await client.get_collections()  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            logger.warning("QdrantManager.health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Collection initialisation
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.

        Idempotent — safe to call on every startup.  Does NOT re-create an
        existing collection (would cause data loss).
        """
        from qdrant_client.http import models as qmodels

        collection_name = settings.qdrant_collection_name
        client = self.get_client()

        logger.info(
            "QdrantManager.ensure_collection() | collection=%s | vector_size=%d",
            collection_name,
            _VECTOR_SIZE,
        )

        try:
            existing = await client.get_collections()  # type: ignore[attr-defined]
            names = [c.name for c in existing.collections]
        except Exception as exc:
            logger.error("QdrantManager: could not list collections: %s", exc)
            raise

        if collection_name in names:
            logger.info(
                "QdrantManager: collection '%s' already exists — skipping creation",
                collection_name,
            )
        else:
            logger.info(
                "QdrantManager: creating collection '%s'", collection_name
            )
            await client.create_collection(  # type: ignore[attr-defined]
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=_VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                    on_disk=False,
                ),
            )
            logger.info("QdrantManager: collection created")

        # Ensure payload indexes (idempotent — Qdrant ignores duplicates)
        await self._ensure_indexes(collection_name)

    async def _ensure_indexes(self, collection_name: str) -> None:
        """Create payload indexes for metadata filtering."""
        from qdrant_client.http import models as qmodels

        client = self.get_client()

        for field in _KEYWORD_INDEXES:
            try:
                await client.create_payload_index(  # type: ignore[attr-defined]
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Already exists — ignore

        for field in _BOOL_INDEXES:
            try:
                await client.create_payload_index(  # type: ignore[attr-defined]
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.BOOL,
                )
            except Exception:
                pass

        for field in _INT_INDEXES:
            try:
                await client.create_payload_index(  # type: ignore[attr-defined]
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.INTEGER,
                )
            except Exception:
                pass

        logger.info(
            "QdrantManager: payload indexes ensured | collection=%s", collection_name
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

qdrant_manager = QdrantManager()
