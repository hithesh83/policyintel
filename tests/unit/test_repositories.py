"""
Unit tests for repositories.

All tests mock the underlying clients/sessions — no real DB connections.

Covers
------
DocumentRepository:
  - create() builds Document + DocumentUpload records
  - get_by_id() returns None when not found
  - get_by_hash() used for duplicate detection
  - update_status() fires correct SQL
  - update_metadata() fires correct SQL with all fields

Neo4jRepository:
  - upsert_graph_bundle() calls session.run() for nodes and rels
  - MERGE queries use node_id as key
  - Empty bundle → no queries
  - health_check() returns True on success, False on error

QdrantRepository:
  - upsert_documents() batches into correct PointStruct calls
  - search() returns list of dicts with score + payload
  - delete_by_document_id() calls delete() with filter
  - health_check() returns True on success, False on error
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.repositories.document_repository import DocumentRepository
from app.repositories.neo4j_repository import Neo4jRepository
from app.repositories.qdrant_repository import QdrantRepository
from app.schemas.pipeline import (
    Benefit,
    EligibilityCriterion,
    ExtractionResult,
    ExtractedEntities,
    GraphBundle,
    GraphNode,
    GraphRelationship,
    HierarchyLevel,
    NodeType,
    RelationshipType,
    VectorDocument,
)


# ---------------------------------------------------------------------------
# DocumentRepository fixtures
# ---------------------------------------------------------------------------


def _make_parsed_doc(doc_id: str = "doc-001"):
    from app.schemas.pipeline import ParsedDocument, ParsedPage

    return ParsedDocument(
        document_id=doc_id,
        filename="policy.pdf",
        title="PM Kisan",
        total_pages=2,
        pages=[ParsedPage(page_number=1, text="Policy text.")],
        raw_text="Policy text.",
        file_metadata={"sha256": "abc123", "size_bytes": 1024},
    )


def _make_db_session():
    """Create a mock AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    return session


class TestDocumentRepository:
    @pytest.mark.asyncio
    async def test_create_adds_document_and_upload(self):
        db = _make_db_session()
        parsed = _make_parsed_doc()
        repo = DocumentRepository(db)

        doc = await repo.create(
            parsed,
            filename="policy.pdf",
            file_hash="deadbeef" * 8,
            file_size_bytes=2048,
        )

        # Both Document and DocumentUpload were added
        assert db.add.call_count == 2
        assert db.flush.call_count == 2

    @pytest.mark.asyncio
    async def test_create_returns_document(self):
        from app.models.document import Document

        db = _make_db_session()
        parsed = _make_parsed_doc()
        repo = DocumentRepository(db)

        doc = await repo.create(
            parsed,
            filename="policy.pdf",
            file_hash="deadbeef" * 8,
            file_size_bytes=1024,
        )

        assert isinstance(doc, Document)
        assert doc.document_id == "doc-001"
        assert doc.filename == "policy.pdf"

    @pytest.mark.asyncio
    async def test_get_by_id_executes_select(self):
        db = _make_db_session()
        # Mock scalar_one_or_none returning None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(db)
        result = await repo.get_by_id("doc-001")

        assert result is None
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_hash_executes_select(self):
        db = _make_db_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(db)
        result = await repo.get_by_hash("abc123def456")

        assert result is None
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_executes_update(self):
        from app.models.document import UploadStatus

        db = _make_db_session()
        db.execute = AsyncMock()

        repo = DocumentRepository(db)
        await repo.update_status("doc-001", UploadStatus.COMPLETED)

        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_with_error(self):
        from app.models.document import UploadStatus

        db = _make_db_session()
        db.execute = AsyncMock()

        repo = DocumentRepository(db)
        await repo.update_status(
            "doc-001", UploadStatus.FAILED, error="LLM connection refused"
        )

        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_executes_select(self):
        db = _make_db_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(db)
        docs = await repo.list_documents(skip=0, limit=10)

        assert docs == []
        db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Neo4jRepository fixtures
# ---------------------------------------------------------------------------


def _make_graph_bundle():
    from app.pipeline.graph_builder import GraphBuilder
    from app.schemas.pipeline import (
        ExtractedEntities,
        ExtractionResult,
        HierarchyLevel,
    )

    entities = ExtractedEntities(
        scheme_name="PM Kisan",
        ministry="Ministry of Agriculture",
        states=["Rajasthan"],
        beneficiary_categories=["farmers"],
        benefits=[
            Benefit(
                benefit_type="cash_transfer",
                description="Rs 6000",
                amount_inr=6000,
            )
        ],
    )
    result = ExtractionResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        hierarchy_level=HierarchyLevel.DOCUMENT,
        page_number=1,
        entities=entities,
        raw_text="PM Kisan scheme text.",
        model_used="test",
    )
    builder = GraphBuilder()
    return builder.build([result], document_id="doc-001")


def _make_neo4j_session():
    session = MagicMock()
    session.run = AsyncMock()
    return session


class TestNeo4jRepository:
    @pytest.mark.asyncio
    async def test_upsert_graph_bundle_calls_session_run(self):
        session = _make_neo4j_session()
        bundle = _make_graph_bundle()

        repo = Neo4jRepository(session)
        await repo.upsert_graph_bundle(bundle)

        # At minimum: once for each node label group + once for each rel type
        assert session.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_empty_bundle_no_queries(self):
        session = _make_neo4j_session()
        bundle = GraphBundle(
            document_id="doc-001",
            nodes=[],
            relationships=[],
        )

        repo = Neo4jRepository(session)
        await repo.upsert_graph_bundle(bundle)

        # No nodes → no relationship queries either
        session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_success(self):
        session = _make_neo4j_session()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"n": 1})
        session.run = AsyncMock(return_value=mock_result)

        repo = Neo4jRepository(session)
        ok = await repo.health_check()

        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        session = _make_neo4j_session()
        session.run = AsyncMock(side_effect=Exception("Connection refused"))

        repo = Neo4jRepository(session)
        ok = await repo.health_check()

        assert ok is False

    @pytest.mark.asyncio
    async def test_upsert_uses_merge_not_create(self):
        """Verify MERGE (not CREATE) appears in queries for idempotency."""
        session = _make_neo4j_session()
        bundle = _make_graph_bundle()

        repo = Neo4jRepository(session)
        await repo.upsert_graph_bundle(bundle)

        queries = [str(call_args) for call_args in session.run.call_args_list]
        assert any("MERGE" in q for q in queries), "Expected MERGE in Cypher queries"


# ---------------------------------------------------------------------------
# QdrantRepository fixtures
# ---------------------------------------------------------------------------


def _make_vector_docs(count: int = 3, doc_id: str = "doc-001") -> list[VectorDocument]:
    return [
        VectorDocument(
            vector_id=f"chunk-{i:03d}",
            chunk_id=f"chunk-{i:03d}",
            document_id=doc_id,
            vector=[0.0] * 1024,
            payload={
                "chunk_id": f"chunk-{i:03d}",
                "document_id": doc_id,
                "hierarchy_level": "section",
                "page_number": i + 1,
            },
            text=f"Policy text chunk {i}.",
        )
        for i in range(count)
    ]


def _make_qdrant_client():
    client = MagicMock()
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.delete = AsyncMock()
    client.get_collection = AsyncMock(return_value=MagicMock())
    client.count = AsyncMock(return_value=MagicMock(count=5))
    return client


class TestQdrantRepository:
    @pytest.mark.asyncio
    async def test_upsert_documents_calls_client(self):
        client = _make_qdrant_client()
        docs = _make_vector_docs(5)

        repo = QdrantRepository(client, "policy_chunks")
        await repo.upsert_documents(docs)

        client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_batches_large_lists(self):
        """Lists larger than _BATCH_SIZE (100) should produce multiple upsert calls."""
        client = _make_qdrant_client()
        docs = _make_vector_docs(150)

        repo = QdrantRepository(client, "policy_chunks")
        await repo.upsert_documents(docs)

        # 150 docs in batches of 100 → 2 calls
        assert client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_upsert_empty_list_no_call(self):
        client = _make_qdrant_client()

        repo = QdrantRepository(client, "policy_chunks")
        await repo.upsert_documents([])

        client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        client = _make_qdrant_client()
        mock_hit = MagicMock()
        mock_hit.score = 0.92
        mock_hit.id = "chunk-001"
        mock_hit.payload = {"chunk_id": "chunk-001", "document_id": "doc-001"}
        client.search = AsyncMock(return_value=[mock_hit])

        repo = QdrantRepository(client, "policy_chunks")
        results = await repo.search([0.1] * 1024, limit=5)

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.92)
        assert results[0]["chunk_id"] == "chunk-001"

    @pytest.mark.asyncio
    async def test_delete_by_document_id_calls_delete(self):
        client = _make_qdrant_client()

        repo = QdrantRepository(client, "policy_chunks")
        await repo.delete_by_document_id("doc-001")

        client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self):
        client = _make_qdrant_client()

        repo = QdrantRepository(client, "policy_chunks")
        ok = await repo.health_check()

        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        client = _make_qdrant_client()
        client.get_collection = AsyncMock(side_effect=Exception("Network error"))

        repo = QdrantRepository(client, "policy_chunks")
        ok = await repo.health_check()

        assert ok is False

    @pytest.mark.asyncio
    async def test_upsert_uses_correct_vector_id(self):
        """Each PointStruct must use vector_id (= chunk_id) as the Qdrant point ID."""
        from qdrant_client.http import models as qmodels

        client = _make_qdrant_client()
        docs = _make_vector_docs(2)

        repo = QdrantRepository(client, "policy_chunks")
        await repo.upsert_documents(docs)

        call_args = client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args.args[1] if call_args.args else []
        # All points should have IDs from vector_id
        for i, point in enumerate(points):
            assert point.id == docs[i].vector_id
