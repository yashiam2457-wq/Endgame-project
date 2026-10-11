"""Health & Metrics API router."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from core import (
    get_app_settings,
    get_encoder,
    get_faiss,
    get_llm,
    get_neo4j,
    get_sqlite,
)
from models import APIResponse, HealthCheck, MetricsResponse

router = APIRouter(prefix="", tags=["Health"])

_START_TIME = time.time()


@router.get("/health", response_model=APIResponse)
async def get_health_status(
    settings=Depends(get_app_settings),
    neo4j=Depends(get_neo4j),
    faiss_store=Depends(get_faiss),
    encoder=Depends(get_encoder),
    llm=Depends(get_llm),
) -> APIResponse:
    """Check subsystem connection and health statuses."""
    neo4j_status = "connected" if (neo4j and neo4j.is_connected) else "offline"
    faiss_status = "healthy" if faiss_store.is_built else "empty"
    sqlite_status = "healthy"

    # Neo4j is REQUIRED — overall health degrades if offline
    overall_status = "healthy" if neo4j_status == "connected" else "degraded"

    # Use actual Ollama model name if available
    llm_model_name = getattr(llm, "_ollama_model", None) or settings.model.generation_model_name

    health = HealthCheck(
        status=overall_status,
        version=settings.app_version,
        neo4j=neo4j_status,
        faiss=faiss_status,
        embedding_model=settings.model.embedding_model_name,
        llm_model=llm_model_name,
        sqlite=sqlite_status,
        uptime_seconds=time.time() - _START_TIME,
    )
    return APIResponse(success=True, message="Health check complete", data=health)



@router.get("/metrics", response_model=APIResponse)
async def get_metrics(
    sqlite=Depends(get_sqlite),
    faiss_store=Depends(get_faiss),
) -> APIResponse:
    """Get system operational metrics."""
    q_stats = sqlite.get_query_stats()
    metrics = {
        "total_queries": q_stats.get("total_queries", 0),
        "avg_latency_ms": q_stats.get("avg_latency_ms", 0.0),
        "faiss_total_vectors": faiss_store.total_vectors,
        "uptime_seconds": time.time() - _START_TIME,
    }
    return APIResponse(success=True, message="Metrics retrieved", data=metrics)
