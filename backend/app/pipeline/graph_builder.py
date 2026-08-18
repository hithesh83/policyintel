"""
Graph Builder
=============

``GraphBuilder`` converts a list of ``ExtractionResult`` objects into a
``GraphBundle`` containing ``GraphNode`` and ``GraphRelationship`` objects
ready to be upserted into Neo4j.

Design Principles
-----------------
1. **Pure transformation** — no I/O, no network calls, no external
   dependencies.  Input → output.  This makes it trivially testable and
   deterministic.

2. **Idempotent node IDs** — every ``GraphNode.node_id`` is a UUID5 derived
   from ``(NodeType.value, canonical_label)``.  Running the pipeline twice
   on the same document produces identical node IDs so Neo4j MERGE upserts
   are safe.

3. **Idempotent relationship IDs** — ``GraphRelationship.rel_id`` is a UUID5
   derived from ``(RelationshipType, source_node_id, target_node_id)``.

4. **Deduplication** — nodes are deduplicated in a ``dict[node_id, GraphNode]``
   before building the final list.  The first occurrence of each node wins
   (properties from later occurrences are dropped).

Node Types Generated
--------------------
- Scheme          — the top-level policy/scheme
- Ministry        — the single issuing ministry (``issuing_ministry``)
- Organisation    — implementing organisations, supporting agencies, departments
- Benefit         — one Benefit instance per extracted benefit
- EligibilityRule — one rule per eligibility criterion
- Beneficiary     — one per beneficiary category
- State           — one per state in geographic scope
- Clause          — one per CLAUSE/SECTION-level chunk with significant content
- Amendment       — one per amendment reference found in any chunk

Relationship Types Generated
-----------------------------
Organisational (replaces the legacy single ADMINISTERED_BY):
- Scheme  -[ISSUED_BY]→       Ministry      (from ``issuing_ministry``)
- Scheme  -[IMPLEMENTED_BY]→  Organisation  (from ``implementing_organizations``)
- Scheme  -[IMPLEMENTED_BY]→  Organisation  (from ``departments``)
- Scheme  -[SUPPORTED_BY]→    Organisation  (from ``supporting_agencies``)

Other:
- Scheme  -[HAS_RULE]→         EligibilityRule
- Scheme  -[HAS_BENEFIT]→      Benefit
- Scheme  -[TARGETS]→          Beneficiary
- Scheme  -[APPLIES_IN]→       State
- Scheme  -[CONTAINS_CLAUSE]→  Clause
- Scheme  -[AMENDS]→           Amendment   (when amendment_references found)
- Scheme  -[SUPERSEDED_BY]→    Scheme      (when supersedes list is non-empty)

Graph Retrieval Capabilities
-----------------------------
With the new topology, the retrieval layer can answer:
  - Which ministry issued PM-KISAN?          Traverse ISSUED_BY
  - Which departments implement PM-KISAN?    Traverse IMPLEMENTED_BY (departments)
  - Which banks support PM-KISAN?            Traverse SUPPORTED_BY
  - Which schemes involve State Governments? MATCH (s)-[:IMPLEMENTED_BY]->(o) WHERE o.org_type = 'state_agency'

Usage
-----
::

    from app.pipeline.graph_builder import GraphBuilder

    builder = GraphBuilder()
    bundle = builder.build(extraction_results, document_id="doc-uuid")
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.schemas.pipeline import (
    Benefit,
    EligibilityCriterion,
    ExtractionResult,
    GraphBundle,
    GraphNode,
    GraphRelationship,
    HierarchyLevel,
    NodeType,
    RelationshipType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ID generation helpers
# ---------------------------------------------------------------------------


def _node_id(node_type: NodeType, label: str) -> str:
    """
    Derive a stable UUID5 for a graph node.

    UUID5 is deterministic: the same (type, label) pair always produces the
    same ID.  Label is lowercased and stripped for normalisation.
    """
    name = f"{node_type.value}:{label.strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _rel_id(rel_type: RelationshipType, source_id: str, target_id: str) -> str:
    """Derive a stable UUID5 for a graph relationship."""
    name = f"{rel_type.value}:{source_id}:{target_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """
    Pure transformation from ``ExtractionResult`` list to ``GraphBundle``.

    No external I/O.  Inject no dependencies.

    Usage
    -----
    ::

        builder = GraphBuilder()
        bundle = builder.build(results, document_id="…")
    """

    def __init__(self) -> None:
        logger.info("GraphBuilder initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        results: list[ExtractionResult],
        document_id: str,
    ) -> GraphBundle:
        """
        Build a deduplicated ``GraphBundle`` from extraction results.

        Parameters
        ----------
        results :
            Output of ``PolicyExtractor.extract()``.
        document_id :
            The source document ID — attached to every node and relationship
            for provenance tracking.

        Returns
        -------
        GraphBundle
            Deduplicated nodes and relationships, ready for Neo4j upsert.
        """
        logger.info(
            "GraphBuilder.build() START | document_id=%s | results=%d",
            document_id,
            len(results),
        )

        # Accumulate nodes (deduped by node_id) and relationships (deduped by rel_id)
        nodes: dict[str, GraphNode] = {}
        relationships: dict[str, GraphRelationship] = {}

        # Identify the primary scheme node — use the first non-empty scheme_name
        scheme_node = self._find_or_create_scheme_node(results, document_id)
        if scheme_node:
            nodes[scheme_node.node_id] = scheme_node

        # Process every extraction result
        for result in results:
            self._process_result(
                result=result,
                scheme_node=scheme_node,
                document_id=document_id,
                nodes=nodes,
                relationships=relationships,
            )

        bundle = GraphBundle(
            document_id=document_id,
            nodes=list(nodes.values()),
            relationships=list(relationships.values()),
        )

        logger.info(
            "GraphBuilder.build() DONE | document_id=%s | nodes=%d | relationships=%d",
            document_id,
            bundle.node_count,
            bundle.relationship_count,
        )
        return bundle

    # ------------------------------------------------------------------
    # Per-Result Processing
    # ------------------------------------------------------------------

    def _process_result(
        self,
        *,
        result: ExtractionResult,
        scheme_node: GraphNode | None,
        document_id: str,
        nodes: dict[str, GraphNode],
        relationships: dict[str, GraphRelationship],
    ) -> None:
        """Process one ``ExtractionResult`` — mutates nodes and relationships dicts."""
        entities = result.entities

        # --- Issuing Ministry (singular, ISSUED_BY) ---
        if entities.issuing_ministry:
            ministry_node = self._build_ministry_node(entities.issuing_ministry, document_id)
            _add_node(nodes, ministry_node)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.ISSUED_BY,
                        scheme_node.node_id,
                        ministry_node.node_id,
                        document_id,
                        properties={"ministry_name": entities.issuing_ministry},
                    ),
                )

        # --- Implementing Organisations (plural, IMPLEMENTED_BY) ---
        for org_name in entities.implementing_organizations:
            org_node = self._build_organisation_node(
                org_name, document_id, org_type="implementing_organization"
            )
            _add_node(nodes, org_node)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.IMPLEMENTED_BY,
                        scheme_node.node_id,
                        org_node.node_id,
                        document_id,
                        properties={"org_type": "implementing_organization"},
                    ),
                )

        # --- Departments (plural, IMPLEMENTED_BY with org_type=department) ---
        for dept_name in entities.departments:
            dept_node = self._build_organisation_node(
                dept_name, document_id, org_type="department"
            )
            _add_node(nodes, dept_node)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.IMPLEMENTED_BY,
                        scheme_node.node_id,
                        dept_node.node_id,
                        document_id,
                        properties={"org_type": "department"},
                    ),
                )

        # --- Supporting Agencies (plural, SUPPORTED_BY) ---
        for agency_name in entities.supporting_agencies:
            agency_node = self._build_organisation_node(
                agency_name, document_id, org_type="supporting_agency"
            )
            _add_node(nodes, agency_node)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.SUPPORTED_BY,
                        scheme_node.node_id,
                        agency_node.node_id,
                        document_id,
                        properties={"org_type": "supporting_agency"},
                    ),
                )

        # --- States ---
        for state_name in entities.states:
            state = self._build_state_node(state_name, document_id)
            _add_node(nodes, state)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.APPLIES_IN,
                        scheme_node.node_id,
                        state.node_id,
                        document_id,
                    ),
                )

        # --- Beneficiary categories ---
        for category in entities.beneficiary_categories:
            beneficiary = self._build_beneficiary_node(category, document_id)
            _add_node(nodes, beneficiary)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.TARGETS,
                        scheme_node.node_id,
                        beneficiary.node_id,
                        document_id,
                    ),
                )

        # --- Eligible categories (merge with beneficiary) ---
        for category in entities.eligible_categories:
            beneficiary = self._build_beneficiary_node(category, document_id)
            _add_node(nodes, beneficiary)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.TARGETS,
                        scheme_node.node_id,
                        beneficiary.node_id,
                        document_id,
                    ),
                )

        # --- Eligibility rules ---
        for criterion in entities.eligibility_criteria:
            rule = self._build_eligibility_rule_node(criterion, document_id, result.chunk_id)
            _add_node(nodes, rule)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.HAS_RULE,
                        scheme_node.node_id,
                        rule.node_id,
                        document_id,
                    ),
                )

        # --- Benefits ---
        for benefit in entities.benefits:
            benefit_node = self._build_benefit_node(benefit, document_id, result.chunk_id)
            _add_node(nodes, benefit_node)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.HAS_BENEFIT,
                        scheme_node.node_id,
                        benefit_node.node_id,
                        document_id,
                    ),
                )

        # --- Clause node (for significant non-DOCUMENT chunks) ---
        if result.hierarchy_level not in (HierarchyLevel.DOCUMENT,) and result.title:
            clause = self._build_clause_node(result, document_id)
            _add_node(nodes, clause)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.CONTAINS_CLAUSE,
                        scheme_node.node_id,
                        clause.node_id,
                        document_id,
                    ),
                )

        # --- Amendment references ---
        for ref in entities.amendment_references:
            amendment = self._build_amendment_node(ref, document_id)
            _add_node(nodes, amendment)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.AMENDS,
                        scheme_node.node_id,
                        amendment.node_id,
                        document_id,
                    ),
                )

        # --- Supersedes relationships ---
        for superseded_name in entities.supersedes:
            superseded = self._build_scheme_node_from_name(superseded_name, document_id)
            _add_node(nodes, superseded)
            if scheme_node:
                _add_rel(
                    relationships,
                    self._build_rel(
                        RelationshipType.SUPERSEDED_BY,
                        superseded.node_id,
                        scheme_node.node_id,
                        document_id,
                    ),
                )

    # ------------------------------------------------------------------
    # Node Builders
    # ------------------------------------------------------------------

    def _find_or_create_scheme_node(
        self,
        results: list[ExtractionResult],
        document_id: str,
    ) -> GraphNode | None:
        """
        Find the first non-empty scheme name across all results and build a
        Scheme node.  Returns None if no scheme name was extracted.
        """
        for result in results:
            name = result.entities.scheme_name
            if name:
                return self._build_scheme_node_from_extraction(result, document_id)
        logger.warning(
            "GraphBuilder: no scheme_name found | document_id=%s", document_id
        )
        return None

    def _build_scheme_node_from_extraction(
        self, result: ExtractionResult, document_id: str
    ) -> GraphNode:
        """Build a Scheme node from the richest available extraction result."""
        entities = result.entities
        name = entities.scheme_name or "Unknown Scheme"
        props: dict[str, Any] = {
            "scheme_name": name,
            "issuing_ministry": entities.issuing_ministry,
            "scheme_code": entities.scheme_code,
            "effective_date": entities.effective_date,
            "issue_date": entities.issue_date,
            "policy_type": entities.policy_type,
            "geographic_scope": entities.geographic_scope,
            "funding_pattern": entities.funding_pattern,
            "is_direct_benefit_transfer": entities.is_direct_benefit_transfer,
            "total_annual_benefit_inr": entities.total_annual_benefit_inr,
            "income_limit_annual": entities.income_limit_annual,
            "age_min": entities.age_min,
            "age_max": entities.age_max,
        }
        nid = _node_id(NodeType.SCHEME, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.SCHEME,
            label=name,
            properties={k: v for k, v in props.items() if v is not None},
            source_document_id=document_id,
            source_chunk_id=result.chunk_id,
        )

    def _build_scheme_node_from_name(
        self, name: str, document_id: str
    ) -> GraphNode:
        """Build a minimal Scheme node from a name string (for supersedes links)."""
        nid = _node_id(NodeType.SCHEME, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.SCHEME,
            label=name,
            properties={"scheme_name": name},
            source_document_id=document_id,
            source_chunk_id=None,
        )

    def _build_ministry_node(self, name: str, document_id: str) -> GraphNode:
        """Build a Ministry node for the single issuing ministry."""
        nid = _node_id(NodeType.MINISTRY, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.MINISTRY,
            label=name,
            properties={"ministry_name": name, "org_type": "issuing_ministry"},
            source_document_id=document_id,
        )

    def _build_organisation_node(
        self, name: str, document_id: str, org_type: str
    ) -> GraphNode:
        """
        Build an Organisation node for implementing/supporting bodies.

        Uses NodeType.MINISTRY as the backing type so existing Neo4j queries
        that pattern-match on (:Ministry) still work.  The ``org_type``
        property lets the retrieval layer differentiate between:
        - ``implementing_organization`` — state agencies, district offices
        - ``department``                — central/state departments
        - ``supporting_agency``         — banks, NABARD, NPCI, etc.
        """
        # Scope the node_id by org_type so a bank and a department with the
        # same name produce distinct nodes.
        nid = _node_id(NodeType.MINISTRY, f"{org_type}:{name}")
        return GraphNode(
            node_id=nid,
            node_type=NodeType.MINISTRY,
            label=name,
            properties={"ministry_name": name, "org_type": org_type},
            source_document_id=document_id,
        )

    def _build_state_node(self, name: str, document_id: str) -> GraphNode:
        nid = _node_id(NodeType.STATE, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.STATE,
            label=name,
            properties={"state_name": name},
            source_document_id=document_id,
        )

    def _build_beneficiary_node(self, category: str, document_id: str) -> GraphNode:
        nid = _node_id(NodeType.BENEFICIARY, category)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.BENEFICIARY,
            label=category,
            properties={"category": category},
            source_document_id=document_id,
        )

    def _build_eligibility_rule_node(
        self,
        criterion: EligibilityCriterion,
        document_id: str,
        chunk_id: str,
    ) -> GraphNode:
        label = criterion.description[:80]
        props: dict[str, Any] = {
            "criterion_type": criterion.criterion_type,
            "description": criterion.description,
            "mandatory": criterion.mandatory,
        }
        if criterion.condition:
            props["condition"] = criterion.condition
        if criterion.min_value is not None:
            props["min_value"] = criterion.min_value
        if criterion.max_value is not None:
            props["max_value"] = criterion.max_value
        if criterion.unit:
            props["unit"] = criterion.unit

        # Include chunk_id in the name to allow duplicate criterion types
        # across different clauses to produce distinct nodes
        name = f"{criterion.criterion_type}:{criterion.description[:60]}:{chunk_id[:8]}"
        nid = _node_id(NodeType.ELIGIBILITY_RULE, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.ELIGIBILITY_RULE,
            label=label,
            properties=props,
            source_document_id=document_id,
            source_chunk_id=chunk_id,
        )

    def _build_benefit_node(
        self,
        benefit: Benefit,
        document_id: str,
        chunk_id: str,
    ) -> GraphNode:
        label = benefit.description[:80]
        props: dict[str, Any] = {
            "benefit_type": benefit.benefit_type,
            "description": benefit.description,
        }
        if benefit.amount_inr is not None:
            props["amount_inr"] = benefit.amount_inr
        if benefit.frequency:
            props["frequency"] = benefit.frequency
        if benefit.duration_months is not None:
            props["duration_months"] = benefit.duration_months
        if benefit.conditions:
            props["conditions"] = benefit.conditions

        name = f"{benefit.benefit_type}:{benefit.description[:60]}:{chunk_id[:8]}"
        nid = _node_id(NodeType.BENEFIT, name)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.BENEFIT,
            label=label,
            properties=props,
            source_document_id=document_id,
            source_chunk_id=chunk_id,
        )

    def _build_clause_node(
        self, result: ExtractionResult, document_id: str
    ) -> GraphNode:
        label = result.title or result.section or result.chunk_id[:16]
        props: dict[str, Any] = {
            "title": result.title,
            "section": result.section,
            "hierarchy_level": result.hierarchy_level.value,
            "page_number": result.page_number,
            "chunk_id": result.chunk_id,
        }
        nid = _node_id(NodeType.CLAUSE, result.chunk_id)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.CLAUSE,
            label=label,
            properties={k: v for k, v in props.items() if v is not None},
            source_document_id=document_id,
            source_chunk_id=result.chunk_id,
        )

    def _build_amendment_node(self, ref: str, document_id: str) -> GraphNode:
        nid = _node_id(NodeType.AMENDMENT, ref)
        return GraphNode(
            node_id=nid,
            node_type=NodeType.AMENDMENT,
            label=ref[:80],
            properties={"reference": ref},
            source_document_id=document_id,
        )

    # ------------------------------------------------------------------
    # Relationship Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rel(
        rel_type: RelationshipType,
        source_node_id: str,
        target_node_id: str,
        document_id: str,
        *,
        properties: dict[str, Any] | None = None,
    ) -> GraphRelationship:
        rid = _rel_id(rel_type, source_node_id, target_node_id)
        return GraphRelationship(
            rel_id=rid,
            rel_type=rel_type,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            properties=properties or {},
            source_document_id=document_id,
        )


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _add_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    """Add a node if not already present (first-occurrence wins)."""
    if node.node_id not in nodes:
        nodes[node.node_id] = node


def _add_rel(
    relationships: dict[str, GraphRelationship], rel: GraphRelationship
) -> None:
    """Add a relationship if not already present (first-occurrence wins)."""
    if rel.rel_id not in relationships:
        relationships[rel.rel_id] = rel
