"""System Statistics API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_cache, get_faiss, get_neo4j, get_sqlite
from models import APIResponse, SystemStats
from utils import get_gpu_memory_info, get_ram_info

router = APIRouter(prefix="", tags=["Stats"], dependencies=[Depends(verify_api_key)])


@router.get("/stats", response_model=APIResponse)
async def get_system_stats(
    sqlite: SQLiteManager = Depends(get_sqlite),
    neo4j: Neo4jClient = Depends(get_neo4j),
    faiss_store: FAISSStore = Depends(get_faiss),
    cache: CacheStore = Depends(get_cache),
) -> APIResponse:
    """Get system health metrics, storage counts, memory utilization, and cache rates."""
    _, doc_count = sqlite.list_documents(page_size=1)
    chunk_count = sqlite.count_chunks()
    q_stats = sqlite.get_query_stats()

    gpu_info = get_gpu_memory_info()
    ram_info = get_ram_info()
    cache_stats = cache.get_stats()

    nodes = sqlite.get_all_graph_nodes(limit=100000)
    edges = sqlite.get_all_graph_edges(limit=100000)

    stats = SystemStats(
        document_count=doc_count,
        chunk_count=chunk_count,
        entity_count=len(nodes),
        relationship_count=len(edges),
        query_count=q_stats.get("total_queries", 0),
        avg_query_latency_ms=q_stats.get("avg_latency_ms", 0.0) or 0.0,
        faiss_index_size=faiss_store.total_vectors,
        graph_node_count=len(nodes),
        graph_edge_count=len(edges),
        gpu_memory_used_mb=gpu_info["used_mb"] if gpu_info else None,
        gpu_memory_total_mb=gpu_info["total_mb"] if gpu_info else None,
        ram_used_mb=ram_info["used_mb"],
        ram_total_mb=ram_info["total_mb"],
        cache_hit_rate=cache_stats.get("hit_rate", 0.0),
        uptime_seconds=0.0,
    )

    return APIResponse(success=True, message="System stats retrieved", data=stats)
