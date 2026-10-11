"""API Router aggregator."""
from __future__ import annotations

from fastapi import APIRouter

from api.config_router import router as config_router
from api.documents import router as documents_router
from api.graph import router as graph_router
from api.health import router as health_router
from api.index_router import router as index_router
from api.logs import router as logs_router
from api.query import router as query_router
from api.search import router as search_router
from api.stats import router as stats_router
from api.upload import router as upload_router


api_router = APIRouter(prefix="/api/v1")

api_router.include_router(upload_router)
api_router.include_router(index_router)
api_router.include_router(search_router)
api_router.include_router(query_router)
api_router.include_router(documents_router)
api_router.include_router(graph_router)
api_router.include_router(stats_router)
api_router.include_router(config_router)
api_router.include_router(logs_router)
api_router.include_router(health_router)
