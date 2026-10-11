"""Query API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_retrieval_pipeline, get_sqlite
from models import APIResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="", tags=["Query"], dependencies=[Depends(verify_api_key)])


@router.post("/query", response_model=APIResponse)
async def process_query(
    request: QueryRequest,
    pipeline=Depends(get_retrieval_pipeline),
    sqlite=Depends(get_sqlite),
) -> APIResponse:
    """Execute full GraphRAG query pipeline (Intent → Hybrid Search → Graph Traversal → Gemma Generation)."""
    try:
        response: QueryResponse = await pipeline.execute_query(request)
    except Exception as exc:
        err_msg = str(exc)
        if "not built" in err_msg.lower() or "faiss" in err_msg.lower():
            return APIResponse(
                success=False,
                message="The vector index is empty. Upload and index documents before querying.",
            )
        return APIResponse(success=False, message=f"Query failed: {err_msg}")

    try:
        sqlite.log_query(
            {
                "id": response.query_id,
                "query": response.query,
                "intent": response.intent.value,
                "answer": response.answer,
                "confidence_score": response.confidence_score,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
                "session_id": response.session_id,
                "created_at": response.created_at.isoformat(),
            }
        )
    except Exception:
        pass  # Non-critical — don't fail the query response if logging fails

    return APIResponse(success=True, message="Query processed successfully", data=response)
