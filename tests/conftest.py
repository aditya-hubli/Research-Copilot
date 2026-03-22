from __future__ import annotations

import importlib
import sys
from pathlib import Path
from uuid import uuid4
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


def _reload_backend_modules() -> None:
    module_order = [
        "backend.core.config",
        "backend.db.faiss_store",
        "backend.tools.embedding_tools",
        "backend.tools.metadata_tools",
        "backend.api.cache",
        "backend.api.job_store",
        "backend.review_store",
        "backend.tools.vector_tools",
        "backend.api.pipeline_service",
        "backend.agents.workflow",
        "backend.api.main",
    ]

    for module_name in module_order:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)


def _clear_caches() -> None:
    from backend.core.config import get_settings
    from backend.db.faiss_store import get_faiss_store
    from backend.tools.embedding_tools import _embedding_cache, _openai_client_env

    get_settings.cache_clear()
    get_faiss_store.cache_clear()
    _openai_client_env.cache_clear()
    _embedding_cache.cache_clear()


@pytest.fixture
def tmp_path() -> Path:
    root = Path.cwd() / ".tmp" / "test-runtime"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FAISS_DATA_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("GRAPH_REVIEW_DATA_PATH", str(tmp_path / "reviews" / "reviews.json"))
    monkeypatch.setenv("REQUIRE_API_KEY", "false")
    monkeypatch.setenv("BACKEND_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("QUEUE_WORKER_COUNT", "1")
    monkeypatch.setenv("QUEUE_MAX_SIZE", "20")
    monkeypatch.setenv("ANALYSIS_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("ANALYSIS_CACHE_MAX_ENTRIES", "100")
    monkeypatch.setenv("EMBEDDING_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("EMBEDDING_CACHE_MAX_ENTRIES", "1000")
    monkeypatch.setenv("METADATA_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("METADATA_CACHE_MAX_ENTRIES", "1000")

    _reload_backend_modules()
    _clear_caches()

    from backend.api.main import app

    with TestClient(app) as test_client:
        yield test_client
