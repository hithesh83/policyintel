import logging
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db, get_qdrant, get_neo4j, engine, neo4j_driver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="PolicyIntel AI",
    description="Enterprise Policy Intelligence Engine (Explainable AI & Knowledge Graph)",
    version="1.0.0",
)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources explicitly on application shutdown."""
    logger.info("Shutting down application, cleaning up connections...")
    await engine.dispose()
    await neo4j_driver.close()


@app.get("/health", response_model=Dict[str, Any], tags=["System"])
async def health_check(
    db: AsyncSession = Depends(get_db),
    # Note: Qdrant client and Neo4j driver are loaded directly or via DI
):
    """
    Readiness probe validating connectivity to PostgreSQL, Qdrant, and Neo4j.
    Returns 200 OK if all systems are healthy, otherwise 503 Service Unavailable.
    """
    health_payload = {
        "status": "healthy",
        "postgres": "down",
        "qdrant": "down",
        "neo4j": "down",
    }
    
    # 1. PostgreSQL Check
    try:
        await db.execute(text("SELECT 1"))
        health_payload["postgres"] = "up"
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        health_payload["status"] = "unhealthy"

    # 2. Qdrant Check
    try:
        qdrant = get_qdrant()
        # Fetching collections acts as a lightweight ping
        qdrant.get_collections()
        health_payload["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        health_payload["status"] = "unhealthy"

    # 3. Neo4j Check
    try:
        neo4j_async_driver = get_neo4j()
        async with neo4j_async_driver.session() as session:
            result = await session.run("RETURN 1 AS num")
            record = await result.single()
            if record and record["num"] == 1:
                health_payload["neo4j"] = "up"
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        health_payload["status"] = "unhealthy"

    # Return HTTP 503 if any critical service is down
    if health_payload["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_payload
        )

    return health_payload
