"""
API v1 Router
=============

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.ingestion import router as ingestion_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(ingestion_router)
