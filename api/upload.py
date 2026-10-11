"""Document upload API router."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from api.auth import verify_api_key
from core import app_logger, get_app_settings, get_ingestion_pipeline, get_sqlite
from models import APIResponse, UploadResponse
from utils import SUPPORTED_EXTENSIONS, compute_sha256

router = APIRouter(prefix="", tags=["Upload"], dependencies=[Depends(verify_api_key)])


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion=Depends(get_ingestion_pipeline),
    settings=Depends(get_app_settings),
    sqlite=Depends(get_sqlite),
) -> APIResponse:
    """Upload a new document (PDF, DOCX, TXT, MD, HTML) to data/uploads."""
    filename = file.filename or f"uploaded_{uuid.uuid4().hex}"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return APIResponse(
            success=False,
            message=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Save file to upload directory
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename

    content = await file.read()
    dest_path.write_bytes(content)

    sha256_hash = compute_sha256(dest_path)

    # Check for duplicate by SHA-256
    existing = sqlite.get_document_by_sha256(sha256_hash)
    if existing:
        return APIResponse(
            success=True,
            message=f"Document already exists (id: {existing['id']}). Use /index to re-index.",
            data=UploadResponse(
                document_id=existing["id"],
                filename=filename,
                file_size_bytes=len(content),
                sha256=sha256_hash,
                status=existing.get("status", "indexed"),
                message="Duplicate document — already in the corpus.",
            ),
        )

    # Load document stub via loader and persist to SQLite
    try:
        doc = ingestion._loader.load_file(dest_path)
    except Exception as exc:
        app_logger.warning(f"Loader failed for {filename}: {exc}. Creating manual stub.")
        # Fallback: manually build and persist minimal Document record
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        ext_type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".txt": "txt",
            ".md": "md",
            ".markdown": "md",
            ".html": "html",
            ".htm": "html",
        }
        sqlite.upsert_document({
            "id": doc_id,
            "sha256": sha256_hash,
            "filename": filename,
            "file_type": ext_type_map.get(ext, "txt"),
            "file_path": str(dest_path),
            "status": "pending",
            "title": None,
            "author": None,
            "page_count": None,
            "word_count": None,
            "file_size_bytes": len(content),
            "version": 1,
            "metadata_json": json.dumps({}),
            "created_at": now,
            "updated_at": now,
        })
        return APIResponse(
            success=True,
            message="Document uploaded successfully.",
            data=UploadResponse(
                document_id=doc_id,
                filename=filename,
                file_size_bytes=len(content),
                sha256=sha256_hash,
                status="pending",
                message="Upload successful. Call POST /index to index this document.",
            ),
        )

    # Persist the properly loaded document stub
    ingestion._persist_document(doc)

    return APIResponse(
        success=True,
        message="Document uploaded successfully.",
        data=UploadResponse(
            document_id=doc.id,
            filename=filename,
            file_size_bytes=len(content),
            sha256=sha256_hash,
            status="pending",
            message="Upload successful. Call POST /index to index this document.",
        ),
    )
