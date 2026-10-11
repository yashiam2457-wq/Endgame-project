"""Indexing API router."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends

from api.auth import verify_api_key
from core import DuplicateDocumentError, get_ingestion_pipeline, get_sqlite
from models import APIResponse, IndexRequest, IndexResponse

import uuid

router = APIRouter(prefix="", tags=["Index"], dependencies=[Depends(verify_api_key)])


@router.get("/index/progress/{job_id}", response_model=APIResponse)
async def get_indexing_progress(
    job_id: str,
    ingestion=Depends(get_ingestion_pipeline),
) -> APIResponse:
    """Get live progress percentage and stage for an indexing job."""
    progress_info = ingestion.get_job_progress(job_id)
    return APIResponse(
        success=True,
        message=f"Indexing progress for job {job_id}",
        data=progress_info,
    )


import asyncio


async def _run_async_ingest(ingestion, file_path: Path, force: bool, job_id: str):
    """Background task wrapper for document ingestion running off the main event loop thread."""
    try:
        def _sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(ingestion.ingest_file(file_path, force=force, job_id=job_id))
            finally:
                loop.close()

        await asyncio.to_thread(_sync_wrapper)
    except Exception as exc:
        ingestion.update_job_progress(
            job_id, "Failed", 100, f"Error: {exc}", status="failed"
        )



@router.post("/index", response_model=APIResponse)
async def trigger_indexing(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    ingestion=Depends(get_ingestion_pipeline),
    sqlite=Depends(get_sqlite),
) -> APIResponse:
    """Trigger document indexing with instant job_id return for live progress polling."""
    job_id = str(uuid.uuid4())

    if request.document_id:
        doc_record = sqlite.get_document(request.document_id)
        if not doc_record:
            return APIResponse(
                success=False,
                message=f"Document '{request.document_id}' not found. Upload the file first via POST /upload.",
            )
        file_path_str = doc_record.get("file_path")
        if not file_path_str:
            return APIResponse(
                success=False,
                message=f"Document '{request.document_id}' has no associated file path. Re-upload the document.",
            )
        file_path = Path(file_path_str)
        if not file_path.exists():
            return APIResponse(
                success=False,
                message=f"File not found on disk: {file_path}. Re-upload the document.",
            )

        # Initialize progress tracker immediately
        ingestion.update_job_progress(
            job_id, "Queued", 5, "Job queued for ingestion...", filename=doc_record.get("filename", "")
        )

        # Execute in background task so response returns instantly and frontend polls progress bar
        background_tasks.add_task(_run_async_ingest, ingestion, file_path, True, job_id)

        return APIResponse(
            success=True,
            message="Indexing started in background.",
            data={
                "job_id": job_id,
                "document_id": request.document_id,
                "filename": doc_record.get("filename"),
                "status": "processing",
            },
        )

    # Batch: index all non-indexed documents (pending, processing, failed)
    all_docs, _ = sqlite.list_documents(page_size=500)
    pending_docs = [d for d in all_docs if d.get("status") != "indexed"]
    for d in pending_docs:
        batch_job_id = str(uuid.uuid4())
        ingestion.update_job_progress(
            batch_job_id, "Queued", 5, "Batch job queued...", filename=d.get("filename", "")
        )
        background_tasks.add_task(_run_async_ingest, ingestion, Path(d["file_path"]), request.force or True, batch_job_id)

    return APIResponse(
        success=True,
        message=f"Batch indexing started for {len(pending_docs)} documents",
        data={"job_id": job_id, "pending_count": len(pending_docs)},
    )
