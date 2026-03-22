from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import Lock

from cachetools import TTLCache

from backend.api.models import AnalyzePaperRequest, FastStageResult, FullStageResult
from backend.core.config import get_settings


@dataclass(frozen=True)
class CachedAnalysis:
    fast_stage: FastStageResult
    full_stage: FullStageResult


class AnalysisResultCache:
    def __init__(self) -> None:
        settings = get_settings()
        self._cache = TTLCache[str, CachedAnalysis](
            maxsize=max(1, settings.analysis_cache_max_entries),
            ttl=max(1, settings.analysis_cache_ttl_seconds),
        )
        self._lock = Lock()

    @staticmethod
    def _key_fingerprint(openai_api_key: str | None) -> str:
        normalized = (openai_api_key or "").strip()
        if not normalized:
            return "default"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return digest[:16]

    @staticmethod
    def make_key(payload: AnalyzePaperRequest, openai_api_key: str | None = None) -> str:
        canonical = {
            "paper_url": payload.paper_url.strip(),
            "title": payload.title.strip(),
            "abstract": payload.abstract.strip(),
            "user_id": payload.user_id.strip(),
            "key_fp": AnalysisResultCache._key_fingerprint(openai_api_key),
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, payload: AnalyzePaperRequest, openai_api_key: str | None = None) -> CachedAnalysis | None:
        key = self.make_key(payload, openai_api_key=openai_api_key)
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            return CachedAnalysis(
                fast_stage=item.fast_stage,
                full_stage=item.full_stage,
            )

    def set(
        self,
        payload: AnalyzePaperRequest,
        fast_stage: FastStageResult,
        full_stage: FullStageResult,
        openai_api_key: str | None = None,
    ) -> None:
        key = self.make_key(payload, openai_api_key=openai_api_key)
        with self._lock:
            self._cache[key] = CachedAnalysis(
                fast_stage=fast_stage,
                full_stage=full_stage,
            )

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


result_cache = AnalysisResultCache()
