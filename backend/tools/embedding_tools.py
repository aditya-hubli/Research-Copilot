from __future__ import annotations

import hashlib
from functools import lru_cache
from threading import Lock

from cachetools import TTLCache
from openai import OpenAI

from backend.core.config import get_settings

# Embedding dimensions:
# - OpenAI text-embedding-3-small -> 1536
# - Deterministic fallback -> configurable FAISS dimension
_OPENAI_EMBED_DIM = 1536

_embedding_lock = Lock()


@lru_cache(maxsize=1)
def _embedding_cache() -> TTLCache[str, list[float]]:
    settings = get_settings()
    return TTLCache(
        maxsize=max(1, settings.embedding_cache_max_entries),
        ttl=max(1, settings.embedding_cache_ttl_seconds),
    )


def _embedding_cache_key(text: str, dimension: int, key_marker: str) -> str:
    settings = get_settings()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{settings.llm_provider}:{dimension}:{key_marker}:{digest}"


def _key_marker(openai_api_key: str | None) -> str:
    normalized = (openai_api_key or "").strip()
    if not normalized:
        return "default"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _deterministic_embedding(text: str, dimension: int) -> list[float]:
    """
    Deterministic pseudo-embedding for when no API key is available.
    This is consistent per text but not semantically meaningful.
    """
    if not text:
        text = "__empty__"
    vector: list[float] = []
    counter = 0
    while len(vector) < dimension:
        digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        counter += 1
        for byte in digest:
            vector.append((byte / 255.0) * 2.0 - 1.0)
            if len(vector) >= dimension:
                break
    return vector[:dimension]


@lru_cache(maxsize=1)
def _openai_client_env() -> OpenAI | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    return OpenAI(api_key=settings.openai_api_key)


def has_embedding_provider(openai_api_key: str | None = None) -> bool:
    if str(openai_api_key or "").strip():
        return True
    settings = get_settings()
    return bool(str(settings.openai_api_key or "").strip())


def create_embedding(
    text: str,
    dimension: int | None = None,
    openai_api_key: str | None = None,
) -> list[float]:
    """
    Create an embedding vector for text.

    - Uses OpenAI embeddings when a key is available.
    - Falls back to deterministic hash-based embeddings otherwise.

    The optional `dimension` parameter is accepted for backward compatibility,
    but FAISS store dimension from settings is used to keep vectors consistent.
    """
    settings = get_settings()
    store_dim = int(settings.faiss_dimension)
    normalized_text = text or ""

    resolved_key = str(openai_api_key or "").strip() or str(settings.openai_api_key or "").strip()
    marker = _key_marker(resolved_key or None)
    cache_key = _embedding_cache_key(normalized_text, store_dim, marker)
    cache = _embedding_cache()

    with _embedding_lock:
        cached = cache.get(cache_key)
        if cached is not None:
            return list(cached)

    if not resolved_key:
        vector = _deterministic_embedding(normalized_text, dimension=store_dim)
        with _embedding_lock:
            cache[cache_key] = list(vector)
        return vector

    client = OpenAI(api_key=resolved_key) if resolved_key else _openai_client_env()
    if client is None:
        vector = _deterministic_embedding(normalized_text, dimension=store_dim)
        with _embedding_lock:
            cache[cache_key] = list(vector)
        return vector

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=normalized_text,
        )
        embedding = response.data[0].embedding

        if len(embedding) >= store_dim:
            vector = embedding[:store_dim]
        else:
            vector = embedding + [0.0] * (store_dim - len(embedding))

        with _embedding_lock:
            cache[cache_key] = list(vector)
        return vector
    except Exception:
        vector = _deterministic_embedding(normalized_text, dimension=store_dim)
        with _embedding_lock:
            cache[cache_key] = list(vector)
        return vector
