#!/usr/bin/env python3
"""
PolicyIntel AI — Pipeline Validation Script
============================================

Executes the complete ingestion pipeline against a REAL government policy PDF
without any mocks, fakes, or placeholder objects.

Stages
------
  PDF → PDFParser → DocumentChunker → PolicyExtractor (→ AIService → Ollama)
      → GraphBuilder → DocumentIndexer

Outputs JSON artefacts to ``data/processed/`` and prints a structured progress
report.  Exits with a non-zero status code if any stage fails.

Usage
-----
    cd backend
    python scripts/validate_pipeline.py "../data/raw/OPERATIONAL GUIDELINES.pdf"

Requirements
------------
- Ollama must be running locally (default: http://localhost:11434).
- The model configured in OLLAMA_MODEL (default: qwen2.5:7b) must be pulled.
- All backend Python dependencies must be installed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: add the backend directory to sys.path so that all app.* imports
# resolve correctly when the script is run from the backend/ directory.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # .../backend
sys.path.insert(0, str(_BACKEND_DIR))

# ---------------------------------------------------------------------------
# App imports — must come after sys.path is set
# ---------------------------------------------------------------------------

from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.llm.ollama import LLMSettings, OllamaClient  # noqa: E402
from app.pipeline.chunker import DocumentChunker  # noqa: E402
from app.pipeline.extractor import PolicyExtractor  # noqa: E402
from app.pipeline.graph_builder import GraphBuilder  # noqa: E402
from app.pipeline.indexer import DocumentIndexer  # noqa: E402
from app.pipeline.parser import PDFParser  # noqa: E402
from app.schemas.pipeline import (  # noqa: E402
    DocumentChunk,
    ExtractionResult,
    GraphBundle,
    ParsedDocument,
    VectorDocument,
)
from app.services.ai_service import AIService  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — suppress library noise; script uses structured print output
# ---------------------------------------------------------------------------

configure_logging(level="WARNING")
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _banner(text: str) -> None:
    bar = "=" * 51
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  {text}{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")


def _step(label: str) -> None:
    print(f"\n{_CYAN}▶  {label}...{_RESET}", flush=True)


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  {_DIM}{detail}{_RESET}" if detail else ""
    print(f"   {_GREEN}✓ PASS{_RESET}  {label}{suffix}", flush=True)


def _fail(label: str, error: str) -> None:
    print(f"   {_RED}✗ FAIL{_RESET}  {label}", flush=True)
    print(f"          {_RED}{error}{_RESET}", flush=True)


def _info(key: str, value: Any) -> None:
    v = str(value)
    if len(v) > 120:
        v = v[:117] + "…"
    print(f"   {_DIM}{key:<32}{_RESET} {v}", flush=True)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: Any) -> None:
    """Serialise *data* to *path* using Pydantic-aware JSON encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        raw: Any = data.model_dump()
    elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        raw = [item.model_dump() for item in data]
    else:
        raw = data
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Stage 0: PDF file verification (sync)
# ---------------------------------------------------------------------------


def stage_verify_pdf(pdf_path: Path) -> bytes:
    """Verify the PDF exists, is readable, and return its raw bytes."""
    _step("Verifying PDF")

    if not pdf_path.exists():
        _fail("PDF exists", f"File not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        _fail("PDF is a file", f"Path is not a file: {pdf_path}")
        raise ValueError(f"Not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        _fail("PDF extension", f"Expected .pdf, got {pdf_path.suffix!r}")
        raise ValueError(f"Not a PDF: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    size_kb = len(pdf_bytes) / 1024
    _ok("PDF found and readable", f"{size_kb:,.1f} KB — {pdf_path.name}")

    # Magic-byte validation
    if not pdf_bytes.startswith(b"%PDF"):
        _fail("PDF magic bytes", "File does not start with %PDF — may be corrupted")
        raise ValueError("File header does not match PDF format")

    _ok("PDF header valid (%PDF)")
    return pdf_bytes


# ---------------------------------------------------------------------------
# Stage 0.5: Ollama connectivity pre-check
# ---------------------------------------------------------------------------


async def stage_check_ollama(settings: LLMSettings) -> None:
    """Raise RuntimeError if Ollama is not reachable before starting the pipeline."""
    _step("Checking Ollama connectivity")
    client = OllamaClient(settings)
    try:
        health = await client.health()
    except Exception as exc:
        _fail("Ollama health", str(exc))
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.url}. "
            "Start Ollama with: ollama serve"
        ) from exc
    finally:
        await client.aclose()

    _ok(
        "Ollama reachable",
        f"url={settings.url}  model={settings.model}  status={health.status}",
    )


# ---------------------------------------------------------------------------
# Stage 1: PDFParser
# ---------------------------------------------------------------------------


async def stage_parser(
    pdf_bytes: bytes,
    pdf_path: Path,
    output_dir: Path,
) -> ParsedDocument:
    _step("Running PDFParser")
    parser = PDFParser(prefer_pymupdf=True)
    try:
        parsed = await parser.parse_bytes(pdf_bytes, filename=pdf_path.name)
    except Exception as exc:
        _fail("PDFParser.parse_bytes()", str(exc))
        raise

    _ok("PDFParser completed")
    _info("document_id",          parsed.document_id)
    _info("filename",             parsed.filename)
    _info("title",                parsed.title or "(inferred from content)")
    _info("total_pages",          parsed.total_pages)
    _info("raw_text_length",      f"{len(parsed.raw_text):,} chars")
    _info("sha256 (prefix)",      str(parsed.file_metadata.get("sha256", "—"))[:20] + "…")
    _info("size_bytes",           f"{parsed.file_metadata.get('size_bytes', 0):,}")
    _info("author",               parsed.file_metadata.get("author", "—"))
    _info("creator",              parsed.file_metadata.get("creator", "—"))

    _save_json(output_dir / "parsed_document.json", parsed)
    _ok("Saved → data/processed/parsed_document.json")
    return parsed


# ---------------------------------------------------------------------------
# Stage 2: DocumentChunker
# ---------------------------------------------------------------------------


async def stage_chunker(
    parsed: ParsedDocument,
    ai_service: AIService,
    output_dir: Path,
) -> list[DocumentChunk]:
    _step("Running DocumentChunker")
    chunker = DocumentChunker(ai_service=ai_service)
    try:
        chunks = await chunker.chunk(parsed)
    except Exception as exc:
        _fail("DocumentChunker.chunk()", str(exc))
        raise

    if not chunks:
        _fail("Chunk output", "DocumentChunker returned 0 chunks")
        raise ValueError("Zero chunks produced — parsing may have failed")

    total_chars = sum(len(c.text) for c in chunks)
    avg_chars   = total_chars / len(chunks) if chunks else 0
    level_counts = Counter(c.hierarchy_level.value for c in chunks)

    _ok("DocumentChunker completed")
    _info("total_chunks",         len(chunks))
    _info("total_chars",          f"{total_chars:,}")
    _info("avg_chunk_chars",      f"{avg_chars:,.0f}")
    _info("hierarchy_levels",     dict(level_counts))

    print(f"\n   {_BOLD}First 3 chunks:{_RESET}")
    for i, chunk in enumerate(chunks[:3], 1):
        print(
            f"   {_DIM}[{i}] level={chunk.hierarchy_level.value:<12}"
            f"page={chunk.page_number}  "
            f"chars={len(chunk.text)}  "
            f"title={chunk.title!r}  "
            f"id={chunk.chunk_id[:12]}…{_RESET}"
        )
        preview = chunk.text[:130].replace("\n", " ")
        print(f"       {_DIM}«{preview}…»{_RESET}")

    _save_json(output_dir / "chunks.json", chunks)
    _ok("Saved → data/processed/chunks.json")
    return chunks


# ---------------------------------------------------------------------------
# Stage 3: PolicyExtractor  (real AIService → real Ollama)
# ---------------------------------------------------------------------------


async def stage_extractor(
    chunks: list[DocumentChunk],
    ai_service: AIService,
    output_dir: Path,
) -> list[ExtractionResult]:
    _step("Running PolicyExtractor  [AIService → Ollama]")
    print(
        f"   {_YELLOW}Note: This stage calls Ollama for every chunk. "
        f"Processing {len(chunks)} chunks — may take several minutes.{_RESET}"
    )

    extractor = PolicyExtractor(ai_service=ai_service, max_concurrent_chunks=2)
    try:
        results = await extractor.extract(chunks)
    except Exception as exc:
        _fail("PolicyExtractor.extract()", str(exc))
        raise

    # ── JSON round-trip validation for every result ──────────────────────
    for i, result in enumerate(results):
        try:
            raw = result.model_dump_json()
            json.loads(raw)  # must parse cleanly
        except Exception as exc:
            _fail(
                f"JSON validation (result {i})",
                f"ExtractionResult[{i}] is not valid JSON: {exc}",
            )
            raise ValueError(f"Invalid JSON at result index {i}") from exc

    # ── Aggregate statistics ─────────────────────────────────────────────
    failed     = [r for r in results if r.extraction_error]
    successful = len(results) - len(failed)

    scheme_names   = {r.entities.scheme_name for r in results if r.entities.scheme_name}
    ministries     = {r.entities.ministry for r in results if r.entities.ministry}
    all_benefits   = [b for r in results for b in r.entities.benefits]
    all_eligibility= [e for r in results for e in r.entities.eligibility_criteria]
    all_ineligible = [c for r in results for c in r.entities.ineligible_categories]
    all_docs       = [d for r in results for d in r.entities.documents_required]
    all_dates      = [d for r in results for d in r.entities.key_dates]
    all_amounts    = [a for r in results for a in r.entities.key_amounts]
    all_states     = {s for r in results for s in r.entities.states}

    _ok("JSON validation passed for all ExtractionResult objects")
    _ok("PolicyExtractor completed")
    _info("chunks_processed",           len(results))
    _info("successful_extractions",     successful)
    _info("failed_extractions (non-fatal)", len(failed))
    _info("scheme_name(s)",             scheme_names or "—")
    _info("ministry / ministries",      ministries or "—")
    _info("benefits_found",             len(all_benefits))
    _info("eligibility_criteria_found", len(all_eligibility))
    _info("ineligible_categories",      len(all_ineligible))
    _info("documents_required",         len(all_docs))
    _info("key_dates",                  len(all_dates))
    _info("key_amounts",                len(all_amounts))
    _info("states_mentioned",           all_states or "—")

    # ── Detailed view: first chunk with real data ─────────────────────────
    primary = next(
        (r for r in results if r.entities.scheme_name or r.entities.ministry),
        results[0] if results else None,
    )
    if primary:
        e = primary.entities
        print(f"\n   {_BOLD}Extracted entities (chunk {primary.chunk_id[:12]}…):{_RESET}")
        _info("  scheme_name",              e.scheme_name or "—")
        _info("  ministry",                 e.ministry or "—")
        _info("  policy_type",              e.policy_type or "—")
        _info("  geographic_scope",         e.geographic_scope or "—")
        _info("  effective_date",           e.effective_date or "—")
        _info("  income_limit_annual (₹)",  e.income_limit_annual or "—")
        _info("  is_direct_benefit_transfer", e.is_direct_benefit_transfer)
        _info("  total_annual_benefit_inr", e.total_annual_benefit_inr or "—")

        if e.benefits:
            print(f"   {_DIM}  benefits:{_RESET}")
            for b in e.benefits[:6]:
                print(f"       {_DIM}• [{b.benefit_type}] {b.description}  ₹{b.amount_inr}{_RESET}")

        if e.eligibility_criteria:
            print(f"   {_DIM}  eligibility criteria:{_RESET}")
            for crit in e.eligibility_criteria[:6]:
                print(f"       {_DIM}• [{crit.criterion_type}] {crit.description}{_RESET}")

        if e.ineligible_categories:
            print(f"   {_DIM}  exclusions / ineligible categories:{_RESET}")
            for cat in e.ineligible_categories[:6]:
                print(f"       {_DIM}• {cat}{_RESET}")

        if e.documents_required:
            print(f"   {_DIM}  documents required:{_RESET}")
            for doc in e.documents_required[:6]:
                print(f"       {_DIM}• {doc}{_RESET}")

        if e.key_dates:
            print(f"   {_DIM}  key dates: {', '.join(e.key_dates[:6])}{_RESET}")

        if e.key_amounts:
            print(f"   {_DIM}  key amounts: {', '.join(e.key_amounts[:6])}{_RESET}")

        if e.amendment_references:
            print(f"   {_DIM}  amendment references:{_RESET}")
            for ref in e.amendment_references[:4]:
                print(f"       {_DIM}• {ref}{_RESET}")

    if failed:
        print(f"\n   {_YELLOW}Non-fatal extraction errors ({len(failed)}):{_RESET}")
        for r in failed[:3]:
            snippet = (r.extraction_error or "")[:120]
            print(f"   {_DIM}  chunk {r.chunk_id[:12]}…: {snippet}{_RESET}")

    _save_json(output_dir / "extracted_entities.json", results)
    _ok("Saved → data/processed/extracted_entities.json")
    return results


# ---------------------------------------------------------------------------
# Stage 4: GraphBuilder  (pure transformation — no I/O)
# ---------------------------------------------------------------------------


def stage_graph_builder(
    results: list[ExtractionResult],
    document_id: str,
    output_dir: Path,
) -> GraphBundle:
    _step("Running GraphBuilder")
    builder = GraphBuilder()
    try:
        bundle = builder.build(results, document_id=document_id)
    except Exception as exc:
        _fail("GraphBuilder.build()", str(exc))
        raise

    node_type_counts = Counter(n.node_type.value for n in bundle.nodes)
    rel_type_counts  = Counter(r.rel_type.value   for r in bundle.relationships)

    _ok("GraphBuilder completed")
    _info("total_nodes",          bundle.node_count)
    _info("total_relationships",  bundle.relationship_count)
    _info("node_types",           dict(node_type_counts))
    _info("relationship_types",   dict(rel_type_counts))

    if bundle.nodes:
        print(f"\n   {_BOLD}Sample nodes (up to 6):{_RESET}")
        for node in bundle.nodes[:6]:
            print(
                f"   {_DIM}  [{node.node_type.value:<20}] {node.label[:55]}"
                f"  id={node.node_id[:12]}…{_RESET}"
            )

    if bundle.relationships:
        print(f"\n   {_BOLD}Sample relationships (up to 6):{_RESET}")
        for rel in bundle.relationships[:6]:
            print(
                f"   {_DIM}  {rel.source_node_id[:8]}…"
                f" ─[{rel.rel_type.value}]→"
                f" {rel.target_node_id[:8]}…"
                f"  id={rel.rel_id[:12]}…{_RESET}"
            )

    _save_json(output_dir / "graph.json", bundle)
    _ok("Saved → data/processed/graph.json")
    return bundle


# ---------------------------------------------------------------------------
# Stage 5: DocumentIndexer  (generates VectorDocuments — NO Qdrant write)
# ---------------------------------------------------------------------------


async def stage_indexer(
    chunks: list[DocumentChunk],
    results: list[ExtractionResult],
    ai_service: AIService,
    output_dir: Path,
) -> list[VectorDocument]:
    _step("Running DocumentIndexer  [embedding stub — Qdrant NOT written]")
    indexer = DocumentIndexer(ai_service=ai_service, max_concurrent_embeddings=3)
    try:
        vector_docs = await indexer.index(chunks=chunks, results=results)
    except Exception as exc:
        _fail("DocumentIndexer.index()", str(exc))
        raise

    if not vector_docs:
        _fail("Indexer output", "DocumentIndexer returned 0 VectorDocuments")
        raise ValueError("Zero VectorDocuments produced")

    # Validate all vectors have a consistent dimension
    dims = {len(v.vector) for v in vector_docs}
    if len(dims) != 1:
        _fail("Vector dimensions", f"Inconsistent dimensions: {dims}")
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")

    embedding_dim  = next(iter(dims))
    payload_keys   = sorted({k for v in vector_docs for k in v.payload.keys()})

    _ok("DocumentIndexer completed")
    _info("total_vector_docs",    len(vector_docs))
    _info("embedding_dimension",  embedding_dim)
    _info("payload_fields",       payload_keys)

    print(f"\n   {_BOLD}Sample vector payloads (up to 3):{_RESET}")
    for vd in vector_docs[:3]:
        level   = vd.payload.get("hierarchy_level", "?")
        page    = vd.payload.get("page_number", "?")
        topic   = str(vd.payload.get("topic", "—"))[:55]
        summary = str(vd.payload.get("summary", "—"))[:80]
        preview = [round(x, 6) for x in vd.vector[:4]]
        print(
            f"   {_DIM}  vector_id={vd.vector_id[:12]}…"
            f"  level={level}  page={page}"
            f"  topic={topic!r}{_RESET}"
        )
        print(f"         {_DIM}summary: {summary}{_RESET}")
        print(f"         {_DIM}vector[0:4]={preview}  (dim={len(vd.vector)}){_RESET}")

    _save_json(output_dir / "vectors.json", vector_docs)
    _ok("Saved → data/processed/vectors.json  (Qdrant NOT written)")
    return vector_docs


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _print_summary(
    *,
    pdf_path: Path,
    parsed: ParsedDocument,
    chunks: list[DocumentChunk],
    results: list[ExtractionResult],
    bundle: GraphBundle,
    vector_docs: list[VectorDocument],
    elapsed_s: float,
    output_dir: Path,
) -> None:
    _banner("Validation Summary")

    failed_extractions = sum(1 for r in results if r.extraction_error)
    scheme_names = {r.entities.scheme_name for r in results if r.entities.scheme_name}
    embedding_dim = len(vector_docs[0].vector) if vector_docs else "—"

    rows: list[tuple[str, Any]] = [
        ("PDF file",              pdf_path.name),
        ("Document ID",           parsed.document_id),
        ("Title",                 parsed.title or "(inferred)"),
        ("Scheme name(s)",        ", ".join(scheme_names) if scheme_names else "—"),
        ("Pages parsed",          parsed.total_pages),
        ("Total chunks",          len(chunks)),
        ("Extractions",           f"{len(results) - failed_extractions} ok / {failed_extractions} errors"),
        ("Graph nodes",           bundle.node_count),
        ("Graph relationships",   bundle.relationship_count),
        ("Vector documents",      len(vector_docs)),
        ("Embedding dimension",   embedding_dim),
        ("Processing time",       f"{elapsed_s:.1f}s"),
        ("Artefacts saved to",    str(output_dir)),
    ]

    max_k = max(len(k) for k, _ in rows)
    for key, val in rows:
        print(f"  {_BOLD}{key:<{max_k}}{_RESET}  {val}")

    print()
    print(f"  {_GREEN}{_BOLD}✓  All pipeline stages completed successfully.{_RESET}")
    print()


# ---------------------------------------------------------------------------
# Main async orchestrator
# ---------------------------------------------------------------------------


async def run_validation(pdf_path: Path, output_dir: Path) -> None:
    start = time.monotonic()

    _banner("PolicyIntel AI — Pipeline Validation")
    print(f"  {_DIM}PDF     : {pdf_path}{_RESET}")
    print(f"  {_DIM}Output  : {output_dir}{_RESET}")

    # ── Stage 0: PDF verification ─────────────────────────────────────────
    pdf_bytes = stage_verify_pdf(pdf_path)

    # ── Stage 0.5: Ollama pre-check ───────────────────────────────────────
    llm_settings = LLMSettings()  # reads OLLAMA_URL / OLLAMA_MODEL / etc. from env
    await stage_check_ollama(llm_settings)

    # ── Bootstrap AIService ───────────────────────────────────────────────
    _step("Initialising AIService")
    ai_client  = OllamaClient(llm_settings)
    ai_service = AIService(client=ai_client)
    _ok("AIService ready", f"model={llm_settings.model}  timeout={llm_settings.timeout}s")

    try:
        # Stage 1
        parsed = await stage_parser(pdf_bytes, pdf_path, output_dir)

        # Stage 2
        chunks = await stage_chunker(parsed, ai_service, output_dir)

        # Stage 3
        results = await stage_extractor(chunks, ai_service, output_dir)

        # Stage 4
        bundle = stage_graph_builder(results, parsed.document_id, output_dir)

        # Stage 5
        vector_docs = await stage_indexer(chunks, results, ai_service, output_dir)

    finally:
        # Always release the HTTP connection pool even if a stage fails
        await ai_client.aclose()

    elapsed = time.monotonic() - start
    _print_summary(
        pdf_path=pdf_path,
        parsed=parsed,
        chunks=chunks,
        results=results,
        bundle=bundle,
        vector_docs=vector_docs,
        elapsed_s=elapsed,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"{_RED}Usage: python scripts/validate_pipeline.py <path/to/policy.pdf>"
            f" [output_dir]{_RESET}\n"
            "Example:\n"
            "    cd backend\n"
            '    python scripts/validate_pipeline.py '
            '"../data/raw/OPERATIONAL GUIDELINES.pdf"',
            file=sys.stderr,
        )
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()

    # Optional second argument overrides the output directory
    output_dir = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else (_BACKEND_DIR.parent / "data" / "processed").resolve()
    )

    try:
        asyncio.run(run_validation(pdf_path, output_dir))
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Interrupted by user.{_RESET}", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"\n{_RED}File error: {exc}{_RESET}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        # Covers Ollama not running, AIService not initialised, etc.
        print(f"\n{_RED}Runtime error: {exc}{_RESET}", file=sys.stderr)
        sys.exit(3)
    except Exception as exc:
        print(f"\n{_RED}Pipeline validation FAILED: {exc}{_RESET}", file=sys.stderr)
        logger.exception("Unhandled exception in validate_pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
