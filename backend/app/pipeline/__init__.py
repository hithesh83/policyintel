"""
Pipeline Package
================

Week 2 document understanding pipeline modules.

Public API
----------
- ``DocumentChunker``  — hierarchical document chunking
- ``PolicyExtractor``  — AI-powered entity extraction via AIService
- ``GraphBuilder``     — pure extraction→graph transformation
- ``DocumentIndexer``  — vector document preparation for Qdrant

Import pattern::

    from app.pipeline.chunker import DocumentChunker
    from app.pipeline.extractor import PolicyExtractor
    from app.pipeline.graph_builder import GraphBuilder
    from app.pipeline.indexer import DocumentIndexer
"""

from app.pipeline.chunker import DocumentChunker
from app.pipeline.extractor import PolicyExtractor
from app.pipeline.graph_builder import GraphBuilder
from app.pipeline.indexer import DocumentIndexer

__all__ = [
    "DocumentChunker",
    "PolicyExtractor",
    "GraphBuilder",
    "DocumentIndexer",
]
