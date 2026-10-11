"""API key authentication middleware."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from core import AppConfig, AuthenticationError, get_settings


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str = Security(_API_KEY_HEADER),
    settings: AppConfig = Depends(get_settings),
) -> Optional[str]:
    """Verify incoming request API Key header if authentication is enabled."""
    if not settings.enable_auth:
        return None

    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Include 'X-API-Key' in headers.",
        )
    return api_key
