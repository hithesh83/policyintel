"""
Qdrant Repository
=================

``QdrantRepository`` persists ``VectorDocument`` objects to Qdrant and
provides semantic search.

Upsert Strategy
---------------
``AsyncQdrantClient.upsert()`` — idempotent.  Points with the same
``vector_id`` (= ``chunk_id``) are overwritten on re-ingestion.

Batching
--------
Points are upserted in batches of 100 to avoid oversized HTTP requests.

Search
------
``search()`` accepts a raw float vector + optional Qdrant filter dict and
returns the top-k matching payload dicts.  The caller is responsible for
constructing the filter.

Usage::

    repo = QdrantRepository(qdrant_manager.get_client(), collection_name)
    await repo.upsert_documents(vector_docs)
    results = await repo.search(query_vector, limit=10)
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.pipeline import VectorDocument

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class QdrantRepository:
    """
    Repository for persisting ``VectorDocument`` objects to Qdrant.

    Parameters
    ----------
    client :
        An ``AsyncQdrantClient`` instance.
    collection_name :
        Target Qdrant collection.
    """

    def __init__(self, client: Any, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_documents(self, docs: list[VectorDocument]) -> None:
        """
        Upsert a list of ``VectorDocument`` objects into Qdrant.

        Documents are batched to avoid oversized payloads.

        Parameters
        ----------
        docs :
            Output of ``DocumentIndexer.index()``.
        """
        from qdrant_client.http import models as qmodels

        logger.info(
            "QdrantRepository.upsert_documents() | collection=%s | count=%d",
            self._collection,
            len(docs),
        )

        for i in range(0, len(docs), _BATCH_SIZE):
            batch = docs[i : i + _BATCH_SIZE]
            points = [
                qmodels.PointStruct(
                    id=doc.vector_id,
                    vector=doc.vector,
                    payload=doc.payload,
                )
                for doc in batch
            ]
            await self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )
            logger.debug(
                "QdrantRepository: upserted batch %d/%d | size=%d",
                i // _BATCH_SIZE + 1,
                (len(docs) - 1) // _BATCH_SIZE + 1,
                len(batch),
            )

        logger.info(
            "QdrantRepository.upsert_documents() DONE | collection=%s | total=%d",
            self._collection,
            len(docs),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a semantic similarity search.

        Parameters
        ----------
        vector :
            Query embedding (must match collection vector size).
        limit :
            Maximum number of results to return.
        score_threshold :
            Minimum cosine similarity score (0.0–1.0).
        filters :
            Optional Qdrant filter dict (see Qdrant filter syntax).

        Returns
        -------
        list[dict]
            Each dict contains ``score``, ``chunk_id``, and the full payload.
        """
        from qdrant_client.http import models as qmodels

        qdrant_filter = None
        if filters:
            qdrant_filter = qmodels.Filter(**filters)

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        return [
            {
                "score": r.score,
                "chunk_id": r.payload.get("chunk_id") if r.payload else r.id,
                "document_id": r.payload.get("document_id") if r.payload else None,
                "payload": r.payload or {},
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all vectors belonging to a specific document.

        Uses a payload filter on ``document_id``.
        """
        from qdrant_client.http import models as qmodels

        logger.info(
            "QdrantRepository.delete_by_document_id() | document_id=%s", document_id
        )
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info(
            "QdrantRepository.delete_by_document_id() DONE | document_id=%s", document_id
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the collection is accessible."""
        try:
            info = await self._client.get_collection(self._collection)
            return info is not None
        except Exception as exc:
            logger.warning("QdrantRepository.health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    async def count(self, document_id: str | None = None) -> int:
        """
        Return the number of vectors in the collection.

        If ``document_id`` is provided, return count for that document only.
        """
        from qdrant_client.http import models as qmodels

        if document_id:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
        else:
            result = await self._client.count(
                collection_name=self._collection, exact=True
            )
        return result.count
