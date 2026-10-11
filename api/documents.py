"""Documents API router."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_faiss, get_ingestion_pipeline, get_neo4j, get_sqlite
from models import APIResponse, DocumentDeleteResponse

router = APIRouter(prefix="", tags=["Documents"], dependencies=[Depends(verify_api_key)])


@router.get("/documents", response_model=APIResponse)
async def list_documents(
    sqlite=Depends(get_sqlite),
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> APIResponse:
    """List ingested documents with optional status filter and pagination."""
    docs, total = sqlite.list_documents(status=status, page=page, page_size=page_size)
    return APIResponse(
        success=True,
        message=f"Retrieved {len(docs)} documents",
        data={
            "items": docs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 1,
        },
    )


@router.get("/document/{id}", response_model=APIResponse)
async def get_document(
    id: str,
    sqlite=Depends(get_sqlite),
) -> APIResponse:
    """Get full document metadata and chunks."""
    doc = sqlite.get_document(id)
    if not doc:
        return APIResponse(success=False, message=f"Document '{id}' not found.")
    chunks = sqlite.get_chunks_for_document(id)
    return APIResponse(success=True, message="Document retrieved", data={"document": doc, "chunks": chunks})


@router.delete("/delete/{id}", response_model=APIResponse)
async def delete_document(
    id: str,
    sqlite=Depends(get_sqlite),
    faiss_store=Depends(get_faiss),
    neo4j=Depends(get_neo4j),
) -> APIResponse:
    """Delete document, its chunks, associated FAISS vector index embeddings, and knowledge graph nodes & edges."""
    doc = sqlite.get_document(id)
    if not doc:
        return APIResponse(success=False, message=f"Document '{id}' not found.")

    deleted_chunks = sqlite.delete_chunks_for_document(id)
    deleted_nodes, deleted_edges = sqlite.delete_graph_data_for_document(id)
    sqlite.delete_document(id)
    faiss_store.delete_by_document(id)

    # Purge graph tables if no active documents remain
    sqlite.purge_all_graph_data_if_empty()

    if neo4j and neo4j.is_connected:
        try:
            neo4j.execute_write("MATCH (e:Entity {document_id: $doc_id}) DETACH DELETE e", {"doc_id": id})
            neo4j.execute_write("MATCH ()-[r:RELATED_TO {document_id: $doc_id}]->() DELETE r", {"doc_id": id})
        except Exception:
            pass

    res = DocumentDeleteResponse(
        document_id=id,
        deleted_chunks=deleted_chunks,
        deleted_entities=deleted_nodes,
        deleted_relationships=deleted_edges,
    )
    return APIResponse(success=True, message=f"Document '{doc.get('filename', id)}' and graph data deleted", data=res)



@router.post("/reindex", response_model=APIResponse)
async def reindex_documents(
    ingestion=Depends(get_ingestion_pipeline),
    sqlite=Depends(get_sqlite),
    document_id: str | None = None,
) -> APIResponse:
    """Reindex specified document or all documents."""
    if document_id:
        doc = sqlite.get_document(document_id)
        if not doc:
            return APIResponse(success=False, message=f"Document '{document_id}' not found.")
        res = await ingestion.ingest_file(Path(doc["file_path"]), force=True)
        return APIResponse(success=True, message="Reindexed document", data=res)

    docs, _ = sqlite.list_documents(page_size=500)
    reindexed = []
    for d in docs:
        try:
            r = await ingestion.ingest_file(Path(d["file_path"]), force=True)
            reindexed.append(r)
        except Exception as exc:
            reindexed.append({"document_id": d["id"], "error": str(exc)})

    return APIResponse(success=True, message="Reindexed all documents", data={"reindexed": reindexed})
