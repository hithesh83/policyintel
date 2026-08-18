"""
Health Check Endpoint
=====================

``GET /api/v1/health`` — readiness probe for all three databases.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.db.init_db import health_check_all

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="System health check",
    description="Validates connectivity to PostgreSQL, Neo4j, and Qdrant.",
    response_model=dict[str, Any],
)
async def health_check() -> dict[str, Any]:
    """
    Returns 200 OK if all databases are reachable, 503 otherwise.
    """
    db_status = await health_check_all()

    payload: dict[str, Any] = {
        "status": "healthy" if all(db_status.values()) else "degraded",
        "databases": {
            "postgres": "up" if db_status["postgres"] else "down",
            "neo4j": "up" if db_status["neo4j"] else "down",
            "qdrant": "up" if db_status["qdrant"] else "down",
        },
    }

    if not all(db_status.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )

    return payload
