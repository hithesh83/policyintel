"""
Neo4j Repository
================

``Neo4jRepository`` persists ``GraphBundle`` objects (nodes + relationships)
to Neo4j using MERGE statements for idempotent upserts.

All Cypher operations use parameterised queries — no string interpolation.

Node MERGE Strategy
-------------------
``MERGE (n:Label {node_id: $node_id}) SET n += $props``

Idempotent — running the same bundle twice produces identical graph state.
The ``node_id`` (UUID5) is the stable identifier guaranteed by ``GraphBuilder``.

Relationship MERGE Strategy
----------------------------
``MATCH source, MATCH target → MERGE (s)-[r:TYPE {rel_id: $rel_id}]->(t) SET r += $props``

Only created if both endpoints exist.

Batching
--------
Nodes and relationships are written in batches of 100 using
``UNWIND $batch AS item`` for performance.

Usage::

    async with neo4j_manager.session() as session:
        repo = Neo4jRepository(session)
        await repo.upsert_graph_bundle(bundle)
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.pipeline import GraphBundle, GraphNode, GraphRelationship, NodeType

logger = logging.getLogger(__name__)

# Cypher label for each NodeType
_LABEL_MAP: dict[NodeType, str] = {
    NodeType.SCHEME: "Scheme",
    NodeType.MINISTRY: "Ministry",
    NodeType.STATE: "State",
    NodeType.BENEFICIARY: "Beneficiary",
    NodeType.ELIGIBILITY_RULE: "EligibilityRule",
    NodeType.BENEFIT: "Benefit",
    NodeType.CLAUSE: "Clause",
    NodeType.AMENDMENT: "Amendment",
}

_BATCH_SIZE = 100


class Neo4jRepository:
    """
    Repository for persisting ``GraphBundle`` objects to Neo4j.

    Parameters
    ----------
    session :
        An open Neo4j async session.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upsert_graph_bundle(self, bundle: GraphBundle) -> None:
        """
        Upsert all nodes and relationships from a ``GraphBundle``.

        Nodes are written first (relationships require both endpoints to exist).

        Parameters
        ----------
        bundle :
            Output of ``GraphBuilder.build()``.
        """
        logger.info(
            "Neo4jRepository.upsert_graph_bundle() | document_id=%s | nodes=%d | rels=%d",
            bundle.document_id,
            bundle.node_count,
            bundle.relationship_count,
        )

        # Group nodes by label for efficient batch MERGE
        nodes_by_label: dict[str, list[GraphNode]] = {}
        for node in bundle.nodes:
            label = _LABEL_MAP.get(node.node_type, "Unknown")
            nodes_by_label.setdefault(label, []).append(node)

        # Upsert nodes per label
        for label, nodes in nodes_by_label.items():
            await self._upsert_nodes_batch(label, nodes)

        # Upsert relationships
        await self._upsert_relationships_batch(bundle.relationships)

        logger.info(
            "Neo4jRepository.upsert_graph_bundle() DONE | document_id=%s",
            bundle.document_id,
        )

    async def get_scheme_by_name(self, name: str) -> dict[str, Any] | None:
        """Return a Scheme node by its scheme_name or None."""
        result = await self._session.run(
            "MATCH (s:Scheme) WHERE s.scheme_name = $name RETURN s LIMIT 1",
            name=name,
        )
        record = await result.single()
        if record:
            return dict(record["s"])
        return None

    async def health_check(self) -> bool:
        """Return True if a simple Cypher query succeeds."""
        try:
            result = await self._session.run("RETURN 1 AS n")
            record = await result.single()
            return record is not None and record["n"] == 1
        except Exception as exc:
            logger.warning("Neo4jRepository.health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    async def _upsert_nodes_batch(self, label: str, nodes: list[GraphNode]) -> None:
        """
        MERGE a batch of nodes with the given label.

        Uses ``UNWIND`` for efficient bulk operations.
        """
        for i in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[i : i + _BATCH_SIZE]
            items = [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "source_document_id": n.source_document_id,
                    "source_chunk_id": n.source_chunk_id,
                    **{k: _safe_value(v) for k, v in n.properties.items()},
                }
                for n in batch
            ]
            # Cypher MERGE on node_id; SET n += props (additive update)
            query = (
                f"UNWIND $items AS item "
                f"MERGE (n:{label} {{node_id: item.node_id}}) "
                f"SET n += item, n.label = item.label, "
                f"    n.source_document_id = item.source_document_id"
            )
            await self._session.run(query, items=items)
            logger.debug(
                "Neo4jRepository: upserted %d %s nodes", len(batch), label
            )

    async def _upsert_relationships_batch(
        self, relationships: list[GraphRelationship]
    ) -> None:
        """
        MERGE a batch of relationships.

        Requires both source and target nodes to already exist.
        Groups relationships by type for efficient batching.
        """
        # Group by relationship type
        by_type: dict[str, list[GraphRelationship]] = {}
        for rel in relationships:
            by_type.setdefault(rel.rel_type.value.upper(), []).append(rel)

        for rel_type_str, rels in by_type.items():
            for i in range(0, len(rels), _BATCH_SIZE):
                batch = rels[i : i + _BATCH_SIZE]
                items = [
                    {
                        "rel_id": r.rel_id,
                        "source_id": r.source_node_id,
                        "target_id": r.target_node_id,
                        "source_document_id": r.source_document_id,
                        **{k: _safe_value(v) for k, v in r.properties.items()},
                    }
                    for r in batch
                ]
                query = (
                    f"UNWIND $items AS item "
                    f"MATCH (s {{node_id: item.source_id}}) "
                    f"MATCH (t {{node_id: item.target_id}}) "
                    f"MERGE (s)-[r:{rel_type_str} {{rel_id: item.rel_id}}]->(t) "
                    f"SET r += item, r.source_document_id = item.source_document_id"
                )
                try:
                    await self._session.run(query, items=items)
                    logger.debug(
                        "Neo4jRepository: upserted %d %s relationships",
                        len(batch),
                        rel_type_str,
                    )
                except Exception as exc:
                    logger.warning(
                        "Neo4jRepository: relationship upsert failed | type=%s | error=%s",
                        rel_type_str,
                        exc,
                    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_value(v: Any) -> Any:
    """
    Coerce Python values to Neo4j-compatible primitives.

    Neo4j supports: str, int, float, bool, list of primitives.
    Lists of dicts (Pydantic models serialised to dicts) are converted to JSON strings.
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        # Neo4j supports lists of primitives — stringify complex items
        safe = []
        for item in v:
            if isinstance(item, (str, int, float, bool)):
                safe.append(item)
            else:
                import json
                safe.append(json.dumps(item, default=str))
        return safe
    import json
    return json.dumps(v, default=str)
