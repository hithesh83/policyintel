"""
Core DB Package
===============

Exports the three database manager singletons.
"""

from app.core.db.neo4j import Neo4jManager, neo4j_manager, get_neo4j_session
from app.core.db.postgres import PostgresManager, postgres_manager, get_db_session
from app.core.db.qdrant import QdrantManager, qdrant_manager

__all__ = [
    "Neo4jManager",
    "neo4j_manager",
    "get_neo4j_session",
    "PostgresManager",
    "postgres_manager",
    "get_db_session",
    "QdrantManager",
    "qdrant_manager",
]
