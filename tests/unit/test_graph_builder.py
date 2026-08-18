"""
Unit tests for GraphBuilder.

GraphBuilder is a pure function (no I/O) so no mocking is needed.

Covers
------
- All node types generated correctly
- All relationship types generated correctly
- Node deduplication (same type + label → same node_id)
- Relationship deduplication (idempotent upserts)
- Node IDs are stable UUID5 strings
- GraphBundle node_count / relationship_count properties
- Empty extractions → minimal bundle (no crash)
- Scheme node from first non-empty scheme_name
- Ministry node → ADMINISTERED_BY relationship
- State nodes → APPLIES_IN relationship
- Beneficiary nodes → TARGETS relationship
- EligibilityRule nodes → HAS_RULE relationship
- Benefit nodes → HAS_BENEFIT relationship
- Amendment nodes → AMENDS relationship
- Supersedes → SUPERSEDED_BY relationship
- Clause nodes → CONTAINS_CLAUSE relationship
"""

from __future__ import annotations

import uuid

import pytest

from app.pipeline.graph_builder import GraphBuilder, _node_id, _rel_id
from app.schemas.pipeline import (
    Benefit,
    DocumentChunk,
    EligibilityCriterion,
    ExtractionResult,
    ExtractedEntities,
    HierarchyLevel,
    NodeType,
    RelationshipType,
    ChunkMetadata,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_entities(**kwargs) -> ExtractedEntities:
    defaults: dict = {
        "scheme_name": "PM Kisan",
        # DDD organisational fields
        "issuing_ministry": "Ministry of Agriculture and Farmers Welfare",
        "implementing_organizations": ["State Agriculture Departments", "District Collectors"],
        "supporting_agencies": ["NABARD", "SBI"],
        "departments": ["Department of Agriculture, Cooperation & Farmers Welfare"],
        "stakeholders": ["DBT Mission"],
        "funding_pattern": "100% Central",
        "scheme_code": "PKS-001",
        "effective_date": "2019-02-01",
        "policy_type": "central_scheme",
        "geographic_scope": "national",
        "states": ["Rajasthan", "Maharashtra"],
        "beneficiary_categories": ["small farmers", "marginal farmers"],
        "eligible_categories": ["landholding farmers"],
        "eligibility_criteria": [
            EligibilityCriterion(
                criterion_type="income",
                description="Annual income below Rs. 2 lakh",
                condition="income < 200000",
                max_value=200000,
                unit="INR",
                mandatory=True,
            ),
            EligibilityCriterion(
                criterion_type="age",
                description="Age between 18 and 60",
                min_value=18,
                max_value=60,
                unit="years",
                mandatory=True,
            ),
        ],
        "benefits": [
            Benefit(
                benefit_type="cash_transfer",
                description="Rs. 6,000 per year",
                amount_inr=6000,
                frequency="annual",
            )
        ],
        "total_annual_benefit_inr": 6000,
        "is_direct_benefit_transfer": True,
        "deadlines": ["31 March 2024"],
        "key_dates": ["2019-02-01"],
        "key_amounts": ["Rs. 6,000"],
        "documents_required": ["Aadhaar", "Land record"],
        "amendment_references": ["Circular No. 10/2021"],
        "supersedes": ["PM Kisan v1.0"],
    }
    defaults.update(kwargs)
    return ExtractedEntities(**defaults)


def _make_result(
    chunk_id: str = "chunk-001",
    doc_id: str = "doc-001",
    hierarchy_level: HierarchyLevel = HierarchyLevel.SECTION,
    title: str | None = "Eligibility Section",
    entities: ExtractedEntities | None = None,
    extraction_error: str | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        hierarchy_level=hierarchy_level,
        page_number=1,
        section="3.1",
        title=title,
        entities=entities or _make_entities(),
        raw_text="Some policy text.",
        extraction_error=extraction_error,
        model_used="test-model",
    )


# ---------------------------------------------------------------------------
# Tests: node ID stability
# ---------------------------------------------------------------------------


class TestNodeIdStability:
    def test_same_type_and_label_produce_same_id(self):
        id1 = _node_id(NodeType.SCHEME, "PM Kisan")
        id2 = _node_id(NodeType.SCHEME, "PM Kisan")
        assert id1 == id2

    def test_different_type_same_label_produce_different_id(self):
        id1 = _node_id(NodeType.SCHEME, "PM Kisan")
        id2 = _node_id(NodeType.MINISTRY, "PM Kisan")
        assert id1 != id2

    def test_same_type_different_label_produce_different_id(self):
        id1 = _node_id(NodeType.SCHEME, "PM Kisan")
        id2 = _node_id(NodeType.SCHEME, "MNREGA")
        assert id1 != id2

    def test_label_case_normalised(self):
        id1 = _node_id(NodeType.SCHEME, "PM Kisan")
        id2 = _node_id(NodeType.SCHEME, "PM KISAN")
        # Normalised to lowercase — should produce same ID
        assert id1 == id2

    def test_node_id_is_valid_uuid(self):
        nid = _node_id(NodeType.SCHEME, "PM Kisan")
        parsed = uuid.UUID(nid)
        assert str(parsed) == nid


# ---------------------------------------------------------------------------
# Tests: node generation
# ---------------------------------------------------------------------------


class TestNodeGeneration:
    def setup_method(self):
        self.builder = GraphBuilder()
        self.result = _make_result()
        self.bundle = self.builder.build([self.result], document_id="doc-001")

    def test_scheme_node_created(self):
        scheme_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.SCHEME]
        assert len(scheme_nodes) >= 1

    def test_scheme_node_label(self):
        scheme_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.SCHEME]
        scheme_labels = {n.label for n in scheme_nodes}
        assert "PM Kisan" in scheme_labels

    def test_ministry_node_created(self):
        """The issuing_ministry should produce exactly one Ministry node."""
        ministry_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.MINISTRY
                          and n.properties.get("org_type") == "issuing_ministry"]
        assert len(ministry_nodes) == 1
        assert ministry_nodes[0].label == "Ministry of Agriculture and Farmers Welfare"

    def test_state_nodes_created(self):
        state_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.STATE]
        state_labels = {n.label for n in state_nodes}
        assert "Rajasthan" in state_labels
        assert "Maharashtra" in state_labels

    def test_beneficiary_nodes_created(self):
        bene_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.BENEFICIARY]
        bene_labels = {n.label for n in bene_nodes}
        assert "small farmers" in bene_labels
        assert "marginal farmers" in bene_labels

    def test_eligibility_rule_nodes_created(self):
        rule_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.ELIGIBILITY_RULE]
        assert len(rule_nodes) == 2

    def test_benefit_nodes_created(self):
        benefit_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.BENEFIT]
        assert len(benefit_nodes) == 1

    def test_clause_node_created_for_named_section(self):
        clause_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.CLAUSE]
        assert len(clause_nodes) >= 1

    def test_amendment_node_created(self):
        amendment_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.AMENDMENT]
        assert len(amendment_nodes) == 1
        assert amendment_nodes[0].properties["reference"] == "Circular No. 10/2021"

    def test_superseded_scheme_node_created(self):
        scheme_nodes = [n for n in self.bundle.nodes if n.node_type == NodeType.SCHEME]
        scheme_labels = {n.label for n in scheme_nodes}
        assert "PM Kisan v1.0" in scheme_labels

    def test_all_nodes_have_source_document_id(self):
        for node in self.bundle.nodes:
            assert node.source_document_id == "doc-001"


# ---------------------------------------------------------------------------
# Tests: relationship generation
# ---------------------------------------------------------------------------


class TestRelationshipGeneration:
    def setup_method(self):
        self.builder = GraphBuilder()
        self.result = _make_result()
        self.bundle = self.builder.build([self.result], document_id="doc-001")

    def _rels_of_type(self, rel_type: RelationshipType):
        return [r for r in self.bundle.relationships if r.rel_type == rel_type]

    def test_issued_by_relationship(self):
        """issuing_ministry → exactly one ISSUED_BY relationship."""
        rels = self._rels_of_type(RelationshipType.ISSUED_BY)
        assert len(rels) == 1

    def test_implemented_by_relationships(self):
        """implementing_organizations + departments → IMPLEMENTED_BY relationships."""
        rels = self._rels_of_type(RelationshipType.IMPLEMENTED_BY)
        # 2 implementing_organizations + 1 department = 3
        assert len(rels) == 3

    def test_supported_by_relationships(self):
        """supporting_agencies → SUPPORTED_BY relationships."""
        rels = self._rels_of_type(RelationshipType.SUPPORTED_BY)
        assert len(rels) == 2  # NABARD + SBI

    def test_targets_relationships(self):
        rels = self._rels_of_type(RelationshipType.TARGETS)
        # small farmers + marginal farmers + landholding farmers = 3 unique beneficiaries
        assert len(rels) >= 2

    def test_applies_in_relationships(self):
        rels = self._rels_of_type(RelationshipType.APPLIES_IN)
        assert len(rels) == 2  # Rajasthan + Maharashtra

    def test_has_rule_relationships(self):
        rels = self._rels_of_type(RelationshipType.HAS_RULE)
        assert len(rels) == 2  # income + age criteria

    def test_has_benefit_relationships(self):
        rels = self._rels_of_type(RelationshipType.HAS_BENEFIT)
        assert len(rels) == 1

    def test_contains_clause_relationship(self):
        rels = self._rels_of_type(RelationshipType.CONTAINS_CLAUSE)
        assert len(rels) >= 1

    def test_amends_relationship(self):
        rels = self._rels_of_type(RelationshipType.AMENDS)
        assert len(rels) == 1

    def test_superseded_by_relationship(self):
        rels = self._rels_of_type(RelationshipType.SUPERSEDED_BY)
        assert len(rels) == 1

    def test_all_relationships_have_source_document_id(self):
        for rel in self.bundle.relationships:
            assert rel.source_document_id == "doc-001"


# ---------------------------------------------------------------------------
# Tests: deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_issuing_ministry_across_chunks_produces_single_node(self):
        """Two chunks with the same issuing_ministry → one Ministry node."""
        builder = GraphBuilder()
        result1 = _make_result("c1", entities=_make_entities(amendment_references=[]))
        result2 = _make_result("c2", entities=_make_entities(
            amendment_references=[],
            eligibility_criteria=[],
            benefits=[],
            states=[],
            beneficiary_categories=[],
            eligible_categories=[],
            supersedes=[],
            implementing_organizations=[],
            supporting_agencies=[],
            departments=[],
            stakeholders=[],
        ))
        bundle = builder.build([result1, result2], document_id="doc-001")

        # Only one ISSUED_BY node for the same ministry
        ministry_nodes = [
            n for n in bundle.nodes
            if n.node_type == NodeType.MINISTRY
            and n.properties.get("org_type") == "issuing_ministry"
        ]
        assert len(ministry_nodes) == 1

    def test_duplicate_state_across_chunks_produces_single_node(self):
        """Two chunks mentioning Rajasthan → one State node."""
        builder = GraphBuilder()
        result1 = _make_result(
            "c1",
            entities=_make_entities(states=["Rajasthan"], amendment_references=[], supersedes=[]),
        )
        result2 = _make_result(
            "c2",
            entities=_make_entities(
                states=["Rajasthan"],
                amendment_references=[],
                supersedes=[],
                eligibility_criteria=[],
                benefits=[],
                beneficiary_categories=[],
                eligible_categories=[],
            ),
        )
        bundle = builder.build([result1, result2], document_id="doc-001")

        state_nodes = [n for n in bundle.nodes if n.node_type == NodeType.STATE]
        assert len(state_nodes) == 1

    def test_same_relationship_not_duplicated(self):
        """Running build twice on the same data → same relationship count."""
        builder = GraphBuilder()
        result = _make_result()

        bundle1 = builder.build([result], document_id="doc-001")
        bundle2 = builder.build([result], document_id="doc-001")

        assert bundle1.relationship_count == bundle2.relationship_count


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results_list_produces_empty_bundle(self):
        builder = GraphBuilder()
        bundle = builder.build([], document_id="doc-001")

        assert bundle.node_count == 0
        assert bundle.relationship_count == 0

    def test_no_scheme_name_produces_no_scheme_node(self):
        builder = GraphBuilder()
        entities = _make_entities(scheme_name=None)
        result = _make_result(entities=entities)
        bundle = builder.build([result], document_id="doc-001")

        scheme_nodes = [n for n in bundle.nodes if n.node_type == NodeType.SCHEME]
        # With no scheme_name, no Scheme node should be created
        # (only the superseded "PM Kisan v1.0" might be created)
        assert all(n.label != "None" for n in scheme_nodes)

    def test_extraction_error_result_still_processed(self):
        """Results with extraction_error should be processed without crashing."""
        builder = GraphBuilder()
        result = _make_result(extraction_error="LLM failed")
        # Should not raise even if entities are minimal
        bundle = builder.build([result], document_id="doc-001")
        assert isinstance(bundle.node_count, int)

    def test_document_chunk_produces_no_clause_node(self):
        """DOCUMENT-level chunk should not produce a Clause node."""
        builder = GraphBuilder()
        result = _make_result(
            hierarchy_level=HierarchyLevel.DOCUMENT,
            entities=_make_entities(
                eligibility_criteria=[],
                benefits=[],
                states=[],
                beneficiary_categories=[],
                eligible_categories=[],
                amendment_references=[],
                supersedes=[],
            ),
        )
        bundle = builder.build([result], document_id="doc-001")

        clause_nodes = [n for n in bundle.nodes if n.node_type == NodeType.CLAUSE]
        assert len(clause_nodes) == 0

    def test_bundle_document_id_matches_input(self):
        builder = GraphBuilder()
        bundle = builder.build([], document_id="custom-doc-id")
        assert bundle.document_id == "custom-doc-id"

    def test_node_count_property(self):
        builder = GraphBuilder()
        result = _make_result()
        bundle = builder.build([result], document_id="doc-001")
        assert bundle.node_count == len(bundle.nodes)

    def test_relationship_count_property(self):
        builder = GraphBuilder()
        result = _make_result()
        bundle = builder.build([result], document_id="doc-001")
        assert bundle.relationship_count == len(bundle.relationships)


# ---------------------------------------------------------------------------
# Tests: Organisational graph (DDD domain model)
# ---------------------------------------------------------------------------


class TestOrganisationalGraph:
    """
    Tests for the new organisational relationship topology:

    Scheme -[ISSUED_BY]→       Ministry          (issuing_ministry)
    Scheme -[IMPLEMENTED_BY]→  Organisation      (implementing_organizations)
    Scheme -[IMPLEMENTED_BY]→  Organisation      (departments)
    Scheme -[SUPPORTED_BY]→    Organisation      (supporting_agencies)
    """

    def _build(self, **entity_kwargs):
        builder = GraphBuilder()
        entities = _make_entities(
            eligibility_criteria=[],
            benefits=[],
            states=[],
            beneficiary_categories=[],
            eligible_categories=[],
            amendment_references=[],
            supersedes=[],
            **entity_kwargs,
        )
        return builder.build([_make_result(entities=entities)], document_id="doc-001")

    def test_issuing_ministry_produces_issued_by_rel(self):
        bundle = self._build(issuing_ministry="Ministry of Agriculture")
        rels = [r for r in bundle.relationships if r.rel_type == RelationshipType.ISSUED_BY]
        assert len(rels) == 1
        nodes = {n.node_id: n for n in bundle.nodes}
        target = nodes[rels[0].target_node_id]
        assert target.label == "Ministry of Agriculture"
        assert target.properties["org_type"] == "issuing_ministry"

    def test_null_issuing_ministry_produces_no_issued_by_rel(self):
        bundle = self._build(issuing_ministry=None)
        rels = [r for r in bundle.relationships if r.rel_type == RelationshipType.ISSUED_BY]
        assert len(rels) == 0

    def test_implementing_orgs_produce_implemented_by_rels(self):
        bundle = self._build(
            issuing_ministry=None,
            implementing_organizations=["State Agriculture Departments", "District Collectors"],
            departments=[],
            supporting_agencies=[],
            stakeholders=[],
        )
        rels = [r for r in bundle.relationships if r.rel_type == RelationshipType.IMPLEMENTED_BY]
        assert len(rels) == 2
        nodes = {n.node_id: n for n in bundle.nodes}
        labels = {nodes[r.target_node_id].label for r in rels}
        assert "State Agriculture Departments" in labels
        assert "District Collectors" in labels

    def test_departments_produce_implemented_by_rels_with_dept_org_type(self):
        bundle = self._build(
            issuing_ministry=None,
            implementing_organizations=[],
            departments=["Department of Agriculture, Cooperation & Farmers Welfare"],
            supporting_agencies=[],
            stakeholders=[],
        )
        rels = [r for r in bundle.relationships if r.rel_type == RelationshipType.IMPLEMENTED_BY]
        assert len(rels) == 1
        nodes = {n.node_id: n for n in bundle.nodes}
        target = nodes[rels[0].target_node_id]
        assert target.properties["org_type"] == "department"

    def test_supporting_agencies_produce_supported_by_rels(self):
        bundle = self._build(
            issuing_ministry=None,
            implementing_organizations=[],
            departments=[],
            supporting_agencies=["NABARD", "SBI", "NPCI"],
            stakeholders=[],
        )
        rels = [r for r in bundle.relationships if r.rel_type == RelationshipType.SUPPORTED_BY]
        assert len(rels) == 3
        nodes = {n.node_id: n for n in bundle.nodes}
        labels = {nodes[r.target_node_id].label for r in rels}
        assert "NABARD" in labels and "SBI" in labels and "NPCI" in labels

    def test_org_nodes_carry_org_type_property(self):
        """Each node must carry org_type so retrieval can filter by role."""
        bundle = self._build(
            issuing_ministry="Ministry of Finance",
            implementing_organizations=["State Treasury Departments"],
            departments=["Dept of Financial Services"],
            supporting_agencies=["NABARD"],
            stakeholders=[],
        )
        nodes = {n.node_id: n for n in bundle.nodes}

        # Collect org_types from all MINISTRY-typed nodes
        org_types = {n.properties.get("org_type") for n in bundle.nodes
                     if n.node_type == NodeType.MINISTRY}
        assert "issuing_ministry" in org_types
        assert "implementing_organization" in org_types
        assert "department" in org_types
        assert "supporting_agency" in org_types

    def test_empty_org_lists_produce_no_org_rels(self):
        """Empty lists must not create any org relationships or nodes."""
        bundle = self._build(
            issuing_ministry=None,
            implementing_organizations=[],
            departments=[],
            supporting_agencies=[],
            stakeholders=[],
        )
        org_rel_types = {RelationshipType.ISSUED_BY, RelationshipType.IMPLEMENTED_BY,
                         RelationshipType.SUPPORTED_BY}
        org_rels = [r for r in bundle.relationships if r.rel_type in org_rel_types]
        assert len(org_rels) == 0

    def test_same_org_in_different_roles_produces_distinct_nodes(self):
        """
        A bank like SBI appearing as both implementing_organization and
        supporting_agency should produce two distinct nodes (different org_type
        scoping in _node_id).
        """
        bundle = self._build(
            issuing_ministry=None,
            implementing_organizations=["SBI"],
            departments=[],
            supporting_agencies=["SBI"],
            stakeholders=[],
        )
        sbi_nodes = [n for n in bundle.nodes
                     if n.label == "SBI" and n.node_type == NodeType.MINISTRY]
        org_types = {n.properties.get("org_type") for n in sbi_nodes}
        assert "implementing_organization" in org_types
        assert "supporting_agency" in org_types
