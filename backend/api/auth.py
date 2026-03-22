from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from backend.core.config import Settings, get_settings


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    if not isinstance(authorization, str):
        return None
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.require_api_key:
        return

    expected_key = (settings.backend_api_key or "").strip()
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API auth is enabled but BACKEND_API_KEY is not configured.",
        )

    bearer = _extract_bearer_token(authorization)
    provided_header = x_api_key if isinstance(x_api_key, str) else None
    provided = (provided_header or bearer or "").strip()
    if provided != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
