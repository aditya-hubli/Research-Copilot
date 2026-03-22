from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any
from uuid import uuid4

from backend.core.config import get_settings


@dataclass
class GraphReviewRecord:
    review_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    note: str = ""
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "payload": dict(self.payload),
            "status": self.status,
            "note": self.note,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphReviewRecord | None:
        review_id = str(payload.get("review_id", "")).strip()
        if not review_id:
            return None
        return cls(
            review_id=review_id,
            payload=dict(payload.get("payload", {})),
            status=str(payload.get("status", "pending")).strip().lower() or "pending",
            note=str(payload.get("note", "")).strip(),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            decided_at=float(payload["decided_at"]) if payload.get("decided_at") is not None else None,
        )


class GraphReviewStore:
    def __init__(self, data_path: str | None = None) -> None:
        self._lock = Lock()
        settings = get_settings()
        self._data_path = Path(data_path or settings.graph_review_data_path)
        self._reviews: dict[str, GraphReviewRecord] = {}
        self._ensure_storage_dir()
        self._load_from_disk()

    @property
    def storage_path(self) -> Path:
        return self._data_path

    def _ensure_storage_dir(self) -> None:
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

    def _persist_to_disk(self) -> None:
        payload = [record.as_dict() for record in self._reviews.values()]
        self._data_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    def _load_from_disk(self) -> None:
        if not self._data_path.exists():
            return
        try:
            raw = self._data_path.read_text(encoding="utf-8").strip()
            if not raw:
                return
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                record = GraphReviewRecord.from_dict(item)
                if record is None:
                    continue
                self._reviews[record.review_id] = record
        except Exception:
            return

    def create_review(self, payload: dict[str, Any]) -> GraphReviewRecord:
        record = GraphReviewRecord(
            review_id=str(uuid4()),
            payload=dict(payload),
        )
        with self._lock:
            self._reviews[record.review_id] = record
            self._persist_to_disk()
        return record

    def get(self, review_id: str) -> GraphReviewRecord | None:
        with self._lock:
            return self._reviews.get(review_id)

    def list(self, status: str | None = None) -> list[GraphReviewRecord]:
        with self._lock:
            reviews = list(self._reviews.values())

        if status:
            normalized = status.strip().lower()
            reviews = [item for item in reviews if item.status == normalized]

        return sorted(reviews, key=lambda item: item.created_at, reverse=True)

    def set_status(self, review_id: str, status: str, note: str = "") -> GraphReviewRecord | None:
        normalized = status.strip().lower()
        with self._lock:
            record = self._reviews.get(review_id)
            if record is None:
                return None
            record.status = normalized
            record.note = note.strip()
            record.decided_at = time.time()
            self._persist_to_disk()
            return record

    def counts(self) -> dict[str, int]:
        base = {"pending": 0, "approved": 0, "rejected": 0}
        with self._lock:
            for record in self._reviews.values():
                base[record.status] = base.get(record.status, 0) + 1
        base["total"] = sum(base.values())
        return base


graph_review_store = GraphReviewStore()
