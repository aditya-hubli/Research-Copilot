from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
import time
from typing import Any
from uuid import uuid4

from backend.api.models import AnalysisStatus, FastStageResult, FullStageResult


@dataclass
class JobRecord:
    job_id: str
    request_payload: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    status: AnalysisStatus = AnalysisStatus.queued
    fast_stage: FastStageResult | None = None
    full_stage: FullStageResult | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create_job(
        self,
        request_payload: dict[str, Any],
        fast_stage: FastStageResult | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> JobRecord:
        job_id = str(uuid4())
        status = AnalysisStatus.partial if fast_stage else AnalysisStatus.queued
        record = JobRecord(
            job_id=job_id,
            request_payload=request_payload,
            runtime_context=runtime_context or {},
            status=status,
            fast_stage=fast_stage,
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def set_status(self, job_id: str, status: AnalysisStatus) -> None:
        with self._lock:
            if job_id in self._jobs:
                record = self._jobs[job_id]
                record.status = status
                if status == AnalysisStatus.running and record.started_at is None:
                    record.started_at = time.time()
                if status in (AnalysisStatus.complete, AnalysisStatus.failed, AnalysisStatus.canceled):
                    record.completed_at = time.time()

    def set_full_stage(self, job_id: str, result: FullStageResult) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].full_stage = result

    def set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                record = self._jobs[job_id]
                record.status = AnalysisStatus.failed
                record.error = error
                if record.completed_at is None:
                    record.completed_at = time.time()

    def set_canceled(self, job_id: str, detail: str = "Job canceled") -> None:
        with self._lock:
            if job_id in self._jobs:
                record = self._jobs[job_id]
                record.status = AnalysisStatus.canceled
                record.error = detail
                if record.completed_at is None:
                    record.completed_at = time.time()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def metrics_snapshot(self) -> dict[str, int | float | dict[str, int]]:
        with self._lock:
            counts = {status.value: 0 for status in AnalysisStatus}
            queue_waits: list[float] = []
            run_durations: list[float] = []

            for record in self._jobs.values():
                counts[record.status.value] = counts.get(record.status.value, 0) + 1
                if record.started_at is not None:
                    queue_waits.append(max(0.0, record.started_at - record.created_at))
                if record.started_at is not None and record.completed_at is not None:
                    run_durations.append(max(0.0, record.completed_at - record.started_at))

            average_queue_wait = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
            average_run_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0

            return {
                "total_jobs": len(self._jobs),
                "status_counts": counts,
                "average_queue_wait_seconds": round(average_queue_wait, 4),
                "average_run_duration_seconds": round(average_run_duration, 4),
            }


job_store = JobStore()
