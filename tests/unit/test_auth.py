from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.api.auth import verify_api_key
from backend.core.config import Settings


def test_verify_api_key_disabled_allows_requests() -> None:
    settings = Settings(require_api_key=False)
    verify_api_key(settings=settings)


def test_verify_api_key_enabled_accepts_x_api_key() -> None:
    settings = Settings(require_api_key=True, backend_api_key="secret-key")
    verify_api_key(x_api_key="secret-key", settings=settings)


def test_verify_api_key_enabled_accepts_bearer_token() -> None:
    settings = Settings(require_api_key=True, backend_api_key="secret-key")
    verify_api_key(authorization="Bearer secret-key", settings=settings)


def test_verify_api_key_enabled_rejects_missing_credentials() -> None:
    settings = Settings(require_api_key=True, backend_api_key="secret-key")

    with pytest.raises(HTTPException) as exc:
        verify_api_key(settings=settings)

    assert exc.value.status_code == 401


def test_verify_api_key_enabled_rejects_wrong_key() -> None:
    settings = Settings(require_api_key=True, backend_api_key="secret-key")

    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="wrong", settings=settings)

    assert exc.value.status_code == 401
