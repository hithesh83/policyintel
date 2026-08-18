"""
Week 2 Pipeline — Pydantic V2 Schema Definitions
=================================================

Single source of truth for all data shapes that flow through the document
understanding pipeline.  Every model is immutable (frozen=True) and fully
typed so downstream consumers get IDE autocompletion and runtime validation
at every boundary.

Hierarchy
---------
ParsedDocument  (parser.py output / pipeline input)
    │
    ▼
DocumentChunk   (chunker.py output)
    │
    ├──▶ ExtractionResult   (extractor.py output)
    │         │
    │         ├──▶ GraphBundle        (graph_builder.py output)
    │         │        ├── GraphNode
    │         │        └── GraphRelationship
    │         │
    │         └──▶ VectorDocument     (indexer.py output)
    │
    └──▶ VectorDocument     (indexer.py output)

Design Notes
------------
- All models use ``model_config = ConfigDict(frozen=True)`` for immutability.
- Optional fields use ``None`` as sentinel — never empty strings.
- Enum values are lowercase strings for JSON serialisation compatibility.
- ``chunk_id``, ``node_id``, ``document_id`` are UUID strings (str, not UUID)
  so they round-trip through JSON without custom serialisers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# INPUT MODELS — Parser Contract
# ===========================================================================


class ParsedPage(BaseModel):
    """
    A single page extracted from a policy document.

    This is the contract that ``parser.py`` (owned by another engineer) must
    produce.  The chunker accepts a list of these inside a ``ParsedDocument``.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(..., ge=1, description="1-based page number.")
    text: str = Field(..., description="Raw extracted text of this page.")
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured table data found on this page.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Page-level metadata (e.g., rotation, image flags).",
    )


class ParsedDocument(BaseModel):
    """
    Output of ``parser.py`` and primary input to the chunking pipeline.

    Encapsulates the full parsed representation of a single policy PDF or
    structured text document.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        ...,
        description="Stable UUID string identifying this document.",
    )
    filename: str = Field(..., description="Original filename, e.g. 'PM_Kisan_SOP_2023.pdf'.")
    title: str | None = Field(default=None, description="Document title if extractable.")
    total_pages: int = Field(..., ge=1, description="Total number of pages.")
    pages: list[ParsedPage] = Field(..., min_length=1, description="Ordered list of pages.")
    raw_text: str = Field(
        default="",
        description="Full concatenated text (fallback when page-level is unavailable).",
    )
    file_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-level metadata (size, hash, mime-type, etc.).",
    )


# ===========================================================================
# CHUNKER MODELS
# ===========================================================================


class HierarchyLevel(str, Enum):
    """
    Hierarchical depth of a document chunk.

    The numeric ordering reflects nesting depth (DOCUMENT=0 … PARAGRAPH=4).
    """

    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    CLAUSE = "clause"
    PARAGRAPH = "paragraph"


class ChunkMetadata(BaseModel):
    """
    Extensible metadata bag attached to every ``DocumentChunk``.

    Populated by the chunker and optionally enriched by the extractor.
    """

    model_config = ConfigDict(frozen=True)

    content_type: str | None = Field(
        default=None,
        description=(
            "Semantic type: eligibility_criteria | procedure | definition | "
            "benefit | penalty | general_information | schedule | amendment."
        ),
    )
    topic: str | None = Field(default=None, description="Short topic label (5–10 words).")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Named entities found in this chunk.",
    )
    key_dates: list[str] = Field(
        default_factory=list,
        description="ISO-formatted dates or time periods found in this chunk.",
    )
    key_amounts: list[str] = Field(
        default_factory=list,
        description="Monetary amounts or limits found in this chunk.",
    )
    has_eligibility_criteria: bool = Field(default=False)
    has_procedure_steps: bool = Field(default=False)
    summary: str | None = Field(
        default=None,
        description="1–2 sentence AI-generated summary of this chunk.",
    )
    word_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata for future extensibility.",
    )


class DocumentChunk(BaseModel):
    """
    A single node in the hierarchical document tree produced by the chunker.

    Every chunk carries enough context to be independently indexed in Qdrant,
    written to PostgreSQL, or linked as a graph node in Neo4j.

    Field Contract
    --------------
    ``chunk_id``   — UUID5(document_id, sha256(text[:200])), stable across re-runs.
    ``parent_id``  — ``chunk_id`` of the immediate parent; ``None`` at DOCUMENT level.
    ``page_number``— first page this chunk appears on (1-based).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Stable UUID5 identifier for this chunk.")
    document_id: str = Field(..., description="UUID of the source document.")
    parent_id: str | None = Field(
        default=None,
        description="chunk_id of the parent chunk; None for DOCUMENT-level chunks.",
    )
    hierarchy_level: HierarchyLevel = Field(
        ..., description="Position of this chunk in the document hierarchy."
    )
    page_number: int = Field(..., ge=1, description="First page this chunk appears on.")
    title: str | None = Field(
        default=None,
        description="Heading / title of this chunk extracted from the document.",
    )
    section: str | None = Field(
        default=None,
        description="Section identifier (e.g., '3.2', 'Chapter IV').",
    )
    text: str = Field(..., description="Raw text content of this chunk.")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,  # type: ignore[arg-type]
        description="Structured metadata for this chunk.",
    )


# ===========================================================================
# EXTRACTOR MODELS
# ===========================================================================


class EligibilityCriterion(BaseModel):
    """A single parsed eligibility condition from a policy section."""

    model_config = ConfigDict(frozen=True)

    criterion_type: str = Field(
        ...,
        description=(
            "Type: age | income | occupation | land_holding | residence | "
            "caste_category | gender | disability | marital_status | other."
        ),
    )
    description: str = Field(..., description="Plain English description of the criterion.")
    condition: str | None = Field(
        default=None,
        description="Exact logical condition, e.g. 'age >= 18 AND age <= 60'.",
    )
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)
    unit: str | None = Field(default=None, description="e.g. 'years', 'INR', 'acres'.")
    mandatory: bool = Field(default=True)


class Benefit(BaseModel):
    """A single financial or in-kind benefit described in a policy section."""

    model_config = ConfigDict(frozen=True)

    benefit_type: str = Field(
        ...,
        description=(
            "Type: cash_transfer | subsidy | loan | insurance | pension | "
            "training | equipment | land | housing | other."
        ),
    )
    description: str = Field(..., description="Plain English description of the benefit.")
    amount_inr: float | None = Field(default=None, description="Amount in Indian Rupees.")
    frequency: str | None = Field(
        default=None,
        description="one_time | monthly | quarterly | annual | on_demand.",
    )
    duration_months: float | None = Field(default=None)
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions attached to this benefit.",
    )


class ExtractedEntities(BaseModel):
    """
    The complete structured extraction produced by ``PolicyExtractor`` for a
    single ``DocumentChunk``.

    All fields are optional — a partial extraction is valid and preferable to
    a hard failure when the LLM cannot infer a value.

    Government Policy Domain Model
    -------------------------------
    Indian government schemes involve multiple distinct organisational roles:

    ``issuing_ministry``
        The single central/state ministry that issued the policy.  Singular
        because every scheme has exactly one issuing authority (e.g.
        "Ministry of Agriculture & Farmers Welfare").  ``str | None``.

    ``implementing_organizations``
        Departments, state agencies, or PSBs that operationally deliver the
        scheme (e.g. "State Agriculture Departments", "District Collectors").
        ``list[str]`` — many schemes have multiple implementing bodies.

    ``supporting_agencies``
        Banks, insurance companies, technology providers, and other
        third-party bodies that support but do not directly administer
        (e.g. "NABARD", "SBI", "NPCI").
        ``list[str]``.

    ``departments``
        Sub-ministerial or cross-ministerial departments named explicitly
        in the document (e.g. "Department of Agriculture, Cooperation &
        Farmers Welfare").
        ``list[str]``.

    ``funding_pattern``
        Cost-sharing ratio string, e.g. "60:40 Centre:State",
        "100% Central".  ``str | None``.

    ``stakeholders``
        Any other named stakeholders not captured above (nodal officers,
        local bodies, gram panchayats, etc.).  ``list[str]``.

    Together these fields allow the retrieval layer to answer:
        - Which ministry issued PM-KISAN?  → ``issuing_ministry``
        - Which departments implement it?  → ``departments``
        - Which banks are involved?        → ``supporting_agencies``
        - Which state agencies deliver it? → ``implementing_organizations``
    without schema changes.
    """

    model_config = ConfigDict(frozen=True)

    # ── Document-level identity fields ──────────────────────────────────────
    scheme_name: str | None = Field(default=None)
    scheme_code: str | None = Field(default=None)
    effective_date: str | None = Field(default=None, description="YYYY-MM-DD or null.")
    issue_date: str | None = Field(default=None, description="YYYY-MM-DD or null.")
    policy_type: str | None = Field(default=None)
    geographic_scope: str | None = Field(default=None)
    states: list[str] = Field(default_factory=list, description="States mentioned.")
    supersedes: list[str] = Field(
        default_factory=list, description="Documents this policy supersedes."
    )

    # ── Government organisational fields (DDD domain model) ─────────────────
    issuing_ministry: str | None = Field(
        default=None,
        description=(
            "The single ministry or department that issued/owns this policy. "
            "e.g. 'Ministry of Agriculture and Farmers Welfare'. "
            "Always singular — every scheme has exactly one issuing authority."
        ),
    )
    implementing_organizations: list[str] = Field(
        default_factory=list,
        description=(
            "Organisations that operationally deliver the scheme. "
            "e.g. ['State Agriculture Departments', 'District Collectors', "
            "'Block Development Officers']. Typically plural."
        ),
    )
    supporting_agencies: list[str] = Field(
        default_factory=list,
        description=(
            "Banks, insurance companies, technology providers, and other "
            "third-party support bodies. e.g. ['NABARD', 'SBI', 'NPCI', "
            "'India Post Payments Bank']."
        ),
    )
    departments: list[str] = Field(
        default_factory=list,
        description=(
            "Sub-ministerial or cross-ministerial departments named explicitly. "
            "e.g. ['Department of Agriculture, Cooperation & Farmers Welfare', "
            "'Department of Financial Services']."
        ),
    )
    funding_pattern: str | None = Field(
        default=None,
        description="Cost-sharing ratio e.g. '60:40 Centre:State', '100% Central'.",
    )
    stakeholders: list[str] = Field(
        default_factory=list,
        description=(
            "Other named stakeholders: nodal officers, local bodies, "
            "gram panchayats, DBT Mission, etc."
        ),
    )

    # ── Chunk-level extraction ───────────────────────────────────────────────
    eligibility_criteria: list[EligibilityCriterion] = Field(default_factory=list)
    eligible_categories: list[str] = Field(default_factory=list)
    ineligible_categories: list[str] = Field(default_factory=list)
    income_limit_annual: float | None = Field(default=None, description="Annual INR cap.")
    age_min: float | None = Field(default=None)
    age_max: float | None = Field(default=None)
    beneficiary_categories: list[str] = Field(default_factory=list)
    benefits: list[Benefit] = Field(default_factory=list)
    total_annual_benefit_inr: float | None = Field(default=None)
    is_direct_benefit_transfer: bool | None = Field(default=None)

    # Temporal entities
    deadlines: list[str] = Field(default_factory=list, description="Application deadlines.")
    key_dates: list[str] = Field(default_factory=list)

    # Amounts
    key_amounts: list[str] = Field(default_factory=list)

    # Relational entities
    relationships: list[str] = Field(
        default_factory=list,
        description="Relationships between entities (free-form descriptions).",
    )

    # Documents required
    documents_required: list[str] = Field(
        default_factory=list,
        description="Documents required to apply for this scheme.",
    )

    # Amendments
    amendment_references: list[str] = Field(
        default_factory=list,
        description="References to amendments, circulars, or notifications.",
    )


class ExtractionResult(BaseModel):
    """
    The fully validated extraction output for a single ``DocumentChunk``.

    Wraps ``ExtractedEntities`` with source tracing fields so downstream
    consumers know which chunk produced which data.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="References DocumentChunk.chunk_id.")
    document_id: str = Field(..., description="References ParsedDocument.document_id.")
    hierarchy_level: HierarchyLevel = Field(...)
    page_number: int = Field(..., ge=1)
    section: str | None = Field(default=None)
    title: str | None = Field(default=None)
    entities: ExtractedEntities = Field(
        ..., description="All extracted structured entities."
    )
    raw_text: str = Field(..., description="The original chunk text that was extracted from.")
    extraction_error: str | None = Field(
        default=None,
        description="Set to the error message if extraction partially or fully failed.",
    )
    model_used: str = Field(
        default="unknown",
        description="LLM model name used for this extraction.",
    )
    latency_ms: int = Field(default=0, ge=0)


# ===========================================================================
# GRAPH BUILDER MODELS
# ===========================================================================


class NodeType(str, Enum):
    """Node labels used in the Neo4j knowledge graph."""

    SCHEME = "Scheme"
    CLAUSE = "Clause"
    BENEFIT = "Benefit"
    ELIGIBILITY_RULE = "EligibilityRule"
    MINISTRY = "Ministry"
    BENEFICIARY = "Beneficiary"
    STATE = "State"
    AMENDMENT = "Amendment"


class RelationshipType(str, Enum):
    """Relationship types used in the Neo4j knowledge graph."""

    HAS_RULE = "HAS_RULE"
    HAS_BENEFIT = "HAS_BENEFIT"
    # Organisational relationships (replaces the generic ADMINISTERED_BY)
    ISSUED_BY      = "ISSUED_BY"        # Scheme → issuing Ministry (singular)
    IMPLEMENTED_BY = "IMPLEMENTED_BY"   # Scheme → implementing Dept/Agency
    SUPPORTED_BY   = "SUPPORTED_BY"     # Scheme → supporting Bank/Agency
    # Legacy — kept for Neo4j backward compatibility with existing data
    ADMINISTERED_BY = "ADMINISTERED_BY"
    TARGETS = "TARGETS"
    AMENDS = "AMENDS"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    CONTAINS_CLAUSE = "CONTAINS_CLAUSE"
    APPLIES_IN = "APPLIES_IN"


class GraphNode(BaseModel):
    """
    A single node ready to be upserted into Neo4j.

    ``node_id`` is UUID5(NodeType.value, canonical_label) — stable across
    pipeline re-runs so Neo4j upserts are idempotent.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(..., description="UUID5-derived stable identifier.")
    node_type: NodeType = Field(..., description="Neo4j node label.")
    label: str = Field(..., description="Human-readable canonical name.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="All node properties to persist in Neo4j.",
    )
    source_document_id: str = Field(
        ..., description="ID of the document that produced this node."
    )
    source_chunk_id: str | None = Field(
        default=None,
        description="ID of the chunk that produced this node (for traceability).",
    )


class GraphRelationship(BaseModel):
    """
    A directed relationship between two ``GraphNode`` objects.

    ``rel_id`` is UUID5(rel_type, source_node_id, target_node_id) for
    idempotent upserts.
    """

    model_config = ConfigDict(frozen=True)

    rel_id: str = Field(..., description="UUID5-derived stable relationship identifier.")
    rel_type: RelationshipType = Field(..., description="Neo4j relationship type.")
    source_node_id: str = Field(..., description="node_id of the source node.")
    target_node_id: str = Field(..., description="node_id of the target node.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship properties (e.g., since, confidence).",
    )
    source_document_id: str = Field(
        ..., description="ID of the document that produced this relationship."
    )


class GraphBundle(BaseModel):
    """
    The complete graph representation of all extraction results from one
    document ingestion run.

    Passed to the Neo4j writer in Week 3.  Contains deduplicated nodes and
    relationships — safe to upsert directly.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(...)
    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)


# ===========================================================================
# INDEXER MODELS
# ===========================================================================


class VectorDocument(BaseModel):
    """
    A single document ready to be upserted into a Qdrant collection.

    ``vector`` is populated by ``DocumentIndexer``.  In Week 2 it carries a
    placeholder (list of zeros) with the correct dimension so the Qdrant
    writer (Week 3) can validate the schema without a live embedding model.
    In Week 3, swap the embedding call to a dedicated encoder.

    ``payload`` mirrors Qdrant's payload JSON — kept flat for efficient
    filtering without additional lookups.
    """

    model_config = ConfigDict(frozen=True)

    vector_id: str = Field(
        ...,
        description="Qdrant point ID — same as chunk_id for 1-to-1 mapping.",
    )
    chunk_id: str = Field(...)
    document_id: str = Field(...)
    vector: list[float] = Field(
        ...,
        description="Embedding vector (dimension determined by the embedding model).",
    )
    payload: dict[str, Any] = Field(
        ...,
        description=(
            "Flat metadata payload for Qdrant filtering. "
            "Includes: title, section, hierarchy_level, page_number, "
            "ministry, scheme_name, states, content_type, summary, "
            "has_eligibility_criteria, has_procedure_steps."
        ),
    )
    text: str = Field(..., description="Original chunk text for retrieval display.")
