import os
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

# Load environment variables from .env file (if present)
load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================
POSTGRES_USER = os.getenv("POSTGRES_USER", "policy_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "policy_password")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "policyintel_db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "policy_password")


# ==============================================================================
# Database Clients Initialization
# ==============================================================================

# 1. PostgreSQL - Async Engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Handle disconnected connections gracefully
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# 2. Qdrant - Thread-safe Synchronous Client
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

# 3. Neo4j - Async Driver
neo4j_driver = AsyncGraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ==============================================================================
# Dependency Injectors
# ==============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding asynchronous SQLAlchemy sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_qdrant() -> QdrantClient:
    """FastAPI dependency for yielding the thread-safe Qdrant client."""
    return qdrant_client

def get_neo4j():
    """FastAPI dependency for yielding the Neo4j async driver."""
    return neo4j_driver
