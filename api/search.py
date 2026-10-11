"""Search API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_encoder, get_faiss
from models import APIResponse

router = APIRouter(prefix="", tags=["Search"], dependencies=[Depends(verify_api_key)])


@router.get("/search", response_model=APIResponse)
async def search_vector_store(
    q: str = "",
    top_k: int = 10,
    encoder=Depends(get_encoder),
    faiss_store=Depends(get_faiss),
) -> APIResponse:
    """Execute raw FAISS vector search across chunk embeddings."""
    if not q.strip():
        return APIResponse(success=False, message="Query parameter 'q' cannot be empty.")

    q_emb = encoder.encode_single(q)
    results = faiss_store.search_mmr(q_emb, top_k=top_k)

    return APIResponse(
        success=True,
        message=f"Found {len(results)} results",
        data={"query": q, "results": results},
    )
