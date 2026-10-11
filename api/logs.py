"""Logs API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_app_settings
from models import APIResponse

router = APIRouter(prefix="", tags=["Logs"], dependencies=[Depends(verify_api_key)])


@router.get("/logs", response_model=APIResponse)
async def read_recent_logs(
    settings=Depends(get_app_settings),
    sink: str = "app",
    lines: int = 100,
) -> APIResponse:
    """Fetch recent log lines from specified subsystem sink."""
    log_file = settings.logging.log_dir / f"{sink}.log"
    if not log_file.exists():
        return APIResponse(success=False, message=f"Log file '{sink}.log' not found.")

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        recent_lines = content[-lines:]
        return APIResponse(
            success=True,
            message=f"Retrieved {len(recent_lines)} lines",
            data={"sink": sink, "lines": recent_lines},
        )
    except Exception as exc:
        return APIResponse(success=False, message=f"Error reading logs: {exc}")
