"""Configuration API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_app_settings
from models import APIResponse, ConfigPatchRequest

router = APIRouter(prefix="", tags=["Config"], dependencies=[Depends(verify_api_key)])


@router.get("/config", response_model=APIResponse)
async def get_system_config(settings=Depends(get_app_settings)) -> APIResponse:
    """Retrieve current system configuration settings."""
    return APIResponse(
        success=True,
        message="System configuration retrieved",
        data={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "model": settings.model.model_dump(),
            "chunking": settings.chunking.model_dump(),
            "retrieval": settings.retrieval.model_dump(),
            "faiss": settings.faiss.model_dump(),
        },
    )


@router.patch("/config", response_model=APIResponse)
async def patch_system_config(
    request: ConfigPatchRequest,
    settings=Depends(get_app_settings),
) -> APIResponse:
    """Update configurable parameter sections (retrieval, chunking)."""
    if request.section == "retrieval":
        for k, v in request.updates.items():
            if hasattr(settings.retrieval, k):
                setattr(settings.retrieval, k, v)
    elif request.section == "chunking":
        for k, v in request.updates.items():
            if hasattr(settings.chunking, k):
                setattr(settings.chunking, k, v)

    return APIResponse(
        success=True, message=f"Config section '{request.section}' updated"
    )
