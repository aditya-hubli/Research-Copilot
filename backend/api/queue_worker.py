from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(order=True)
class PrioritizedJob:
    priority: int
    sequence: int
    job_id: str = field(compare=False)


class BackgroundJobQueue:
    def __init__(
        self,
        process_job: Callable[[str], None],
        max_size: int,
        worker_count: int,
        on_cancel: Callable[[str, str], None] | None = None,
    ) -> None:
        self._process_job = process_job
        self._on_cancel = on_cancel
        self._max_size = max(1, max_size)
        self._worker_count = max(1, worker_count)
        self._queue: queue.PriorityQueue[PrioritizedJob] = queue.PriorityQueue(maxsize=self._max_size)
        self._threads: list[threading.Thread] = []
        self._active_jobs: set[str] = set()
        self._active_lock = threading.Lock()
        self._cancelled_jobs: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sequence = 0
        self._sequence_lock = threading.Lock()

    def start(self) -> None:
        if self._threads:
            return

        self._stop_event.clear()
        for idx in range(self._worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"analysis-worker-{idx + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop_event.set()

        for _ in self._threads:
            self._put_internal("__STOP__", priority=99)

        for thread in self._threads:
            thread.join(timeout=3)
        self._threads.clear()

    def is_running(self) -> bool:
        return bool(self._threads) and not self._stop_event.is_set()

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _put_internal(self, job_id: str, priority: int) -> bool:
        try:
            item = PrioritizedJob(priority=priority, sequence=self._next_sequence(), job_id=job_id)
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def enqueue(self, job_id: str, priority: int = 5) -> bool:
        if self._stop_event.is_set():
            return False
        bounded_priority = max(1, min(priority, 10))
        return self._put_internal(job_id, priority=bounded_priority)

    def cancel(self, job_id: str) -> bool:
        with self._active_lock:
            if job_id in self._active_jobs:
                return False

        with self._cancel_lock:
            self._cancelled_jobs.add(job_id)
        return True

    def _is_canceled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled_jobs

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            job_id = item.job_id
            if job_id == "__STOP__":
                self._queue.task_done()
                break

            if self._is_canceled(job_id):
                if self._on_cancel:
                    self._on_cancel(job_id, "canceled-before-start")
                self._queue.task_done()
                continue

            with self._active_lock:
                self._active_jobs.add(job_id)

            try:
                self._process_job(job_id)
            finally:
                with self._active_lock:
                    self._active_jobs.discard(job_id)
                self._queue.task_done()

    def stats(self) -> dict[str, int]:
        with self._active_lock:
            active = len(self._active_jobs)
        with self._cancel_lock:
            canceled = len(self._cancelled_jobs)
        return {
            "queue_depth": self._queue.qsize(),
            "active_jobs": active,
            "worker_count": self._worker_count,
            "max_queue_size": self._max_size,
            "canceled_jobs": canceled,
        }
