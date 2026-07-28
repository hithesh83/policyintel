import asyncio
import logging
from qdrant_client.http import models as qdrant_models
from app.db.session import qdrant_client, neo4j_driver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "policy_chunks"
VECTOR_SIZE = 768  # Specifically sized for Google Gemini text-embedding-004


async def init_qdrant() -> None:
    """
    Ensure the Qdrant collection exists and is configured properly.
    Uses a thread-safe synchronous call wrapped for logical grouping.
    """
    logger.info("Checking Qdrant collections...")
    try:
        collections_response = qdrant_client.get_collections()
        exists = any(c.name == COLLECTION_NAME for c in collections_response.collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection: '{COLLECTION_NAME}' (Size: {VECTOR_SIZE})")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info("Qdrant collection created successfully.")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists. Skipping creation.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        raise


async def init_neo4j() -> None:
    """
    Create necessary constraints and indexes in the Neo4j knowledge graph.
    Requires APOC for some advanced schema rules, but standard constraints are native.
    """
    logger.info("Checking Neo4j schema constraints...")
    
    # Define exact Cypher queries for creating idempotently
    constraint_queries = [
        "CREATE CONSTRAINT scheme_id_unique IF NOT EXISTS FOR (s:Scheme) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE",
    ]
    
    try:
        async with neo4j_driver.session() as session:
            for query in constraint_queries:
                await session.run(query)
                logger.info(f"Executed Neo4j query: {query}")
        logger.info("Neo4j schema constraints ensured successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        raise


async def main() -> None:
    """Run all initialization steps asynchronously."""
    logger.info("Starting database initialization sequence...")
    try:
        # We can run these sequentially to ensure easy log tracing
        await init_qdrant()
        await init_neo4j()
        logger.info("Database initialization completed successfully.")
    finally:
        # Gracefully close the Neo4j driver
        await neo4j_driver.close()


if __name__ == "__main__":
    asyncio.run(main())
