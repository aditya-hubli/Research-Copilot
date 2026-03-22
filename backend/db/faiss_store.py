from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import faiss
import numpy as np

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
_WHITESPACE_RE = re.compile(r"\s+")


class FaissStore:
    def __init__(self, dimension: int = 256, data_dir: str = "backend/data/faiss") -> None:
        self._dimension = dimension
        self._data_dir = Path(data_dir)
        self._index_path = self._data_dir / "index.faiss"
        self._records_path = self._data_dir / "records.json"
        self._index = faiss.IndexFlatIP(dimension)
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._lock = Lock()
        self._ensure_storage_dir()
        self._load_from_disk()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_storage_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _persist_to_disk(self) -> None:
        records = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(self._texts, self._metadata, strict=False)
        ]
        faiss.write_index(self._index, str(self._index_path))
        self._records_path.write_text(
            json.dumps(records, ensure_ascii=True),
            encoding="utf-8",
        )

    def _load_from_disk(self) -> None:
        if not self._index_path.exists() or not self._records_path.exists():
            return

        try:
            loaded_index = faiss.read_index(str(self._index_path))
            if loaded_index.d != self._dimension:
                logger.warning(
                    "Skipping FAISS load due to dimension mismatch: expected %s, found %s",
                    self._dimension,
                    loaded_index.d,
                )
                return

            raw_records = self._records_path.read_text(encoding="utf-8").strip()
            if not raw_records:
                return

            parsed = json.loads(raw_records)
            if not isinstance(parsed, list):
                return

            texts: list[str] = []
            metadata: list[dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                item_metadata = item.get("metadata", {})
                if not text or not isinstance(item_metadata, dict):
                    continue
                texts.append(text)
                metadata.append(item_metadata)

            if int(loaded_index.ntotal) != len(texts):
                logger.warning(
                    "Skipping FAISS load due to index/metadata size mismatch: %s vs %s",
                    loaded_index.ntotal,
                    len(texts),
                )
                return

            self._index = loaded_index
            self._texts = texts
            self._metadata = metadata
            logger.info("Loaded FAISS store with %s vectors", loaded_index.ntotal)
        except Exception as exc:
            logger.warning("Failed to load FAISS persisted state: %s", exc)

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not embeddings:
            return

        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D list.")
        if matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {matrix.shape[1]}"
            )

        faiss.normalize_L2(matrix)
        with self._lock:
            self._index.add(matrix)
            self._texts.extend(texts)
            self._metadata.extend(metadatas)
            try:
                self._persist_to_disk()
            except Exception as exc:
                logger.warning("Failed to persist FAISS store: %s", exc)

    def contains_paper_url(self, paper_url: str, user_id: str | None = None) -> bool:
        with self._lock:
            for item in self._metadata:
                if item.get("paper_url") != paper_url:
                    continue
                if user_id and item.get("user_id") != user_id:
                    continue
                return True
            return False

    def vector_count(self, user_id: str | None = None) -> int:
        with self._lock:
            if user_id:
                return sum(1 for item in self._metadata if item.get("user_id") == user_id)
            return int(self._index.ntotal)

    def unique_paper_count(self, user_id: str | None = None) -> int:
        with self._lock:
            urls = {
                str(item.get("paper_url", "")).strip()
                for item in self._metadata
                if str(item.get("paper_url", "")).strip()
                and (not user_id or item.get("user_id") == user_id)
            }
            return len(urls)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        with self._lock:
            if self._index.ntotal == 0:
                return []

            vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
            if vector.shape[1] != self._dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self._dimension}, got {vector.shape[1]}"
                )
            faiss.normalize_L2(vector)
            scores, indices = self._index.search(vector, top_k)

            results: list[dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self._texts):
                    continue
                result = {
                    "score": float(score),
                    "text": self._texts[idx],
                    "metadata": self._metadata[idx],
                }
                results.append(result)
            return results

    def records(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items: list[dict[str, Any]] = []
            for text, metadata in zip(self._texts, self._metadata, strict=False):
                if user_id and str(metadata.get("user_id", "")).strip() != user_id:
                    continue
                items.append({"text": text, "metadata": dict(metadata)})
            return items

    @staticmethod
    def _dedupe_key(text: str, metadata: dict[str, Any]) -> str:
        normalized_text = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
        key_payload = {
            "paper_url": str(metadata.get("paper_url", "")).strip(),
            "title": str(metadata.get("title", "")).strip(),
            "user_id": str(metadata.get("user_id", "")).strip(),
            "section": str(metadata.get("section", "")).strip(),
            "text": normalized_text,
        }
        return json.dumps(key_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def compact(self) -> dict[str, int]:
        with self._lock:
            before = len(self._texts)
            if before == 0:
                return {
                    "vectors_before": 0,
                    "vectors_after": 0,
                    "duplicates_removed": 0,
                    "unique_papers_after": 0,
                }

            keep_embeddings: list[np.ndarray] = []
            keep_texts: list[str] = []
            keep_metadata: list[dict[str, Any]] = []
            seen_keys: set[str] = set()

            for idx, (text, metadata) in enumerate(zip(self._texts, self._metadata, strict=False)):
                dedupe_key = self._dedupe_key(text, metadata)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                keep_texts.append(text)
                keep_metadata.append(dict(metadata))
                keep_embeddings.append(np.asarray(self._index.reconstruct(idx), dtype="float32"))

            rebuilt_index = faiss.IndexFlatIP(self._dimension)
            if keep_embeddings:
                matrix = np.vstack(keep_embeddings).astype("float32")
                faiss.normalize_L2(matrix)
                rebuilt_index.add(matrix)

            self._index = rebuilt_index
            self._texts = keep_texts
            self._metadata = keep_metadata
            self._persist_to_disk()

            unique_papers = {
                str(item.get("paper_url", "")).strip()
                for item in self._metadata
                if str(item.get("paper_url", "")).strip()
            }
            after = len(self._texts)
            return {
                "vectors_before": before,
                "vectors_after": after,
                "duplicates_removed": max(0, before - after),
                "unique_papers_after": len(unique_papers),
            }


@lru_cache(maxsize=1)
def get_faiss_store() -> FaissStore:
    settings = get_settings()
    return FaissStore(
        dimension=settings.faiss_dimension,
        data_dir=settings.faiss_data_dir,
    )
