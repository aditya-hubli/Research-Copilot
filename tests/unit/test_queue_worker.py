from __future__ import annotations

import time

from backend.api.queue_worker import BackgroundJobQueue


def test_priority_queue_processes_lower_priority_number_first() -> None:
    processed: list[str] = []

    def process_job(job_id: str) -> None:
        processed.append(job_id)

    queue = BackgroundJobQueue(process_job=process_job, max_size=10, worker_count=1)
    assert queue.enqueue("job-low", priority=9)
    assert queue.enqueue("job-high", priority=1)

    queue.start()

    deadline = time.monotonic() + 3
    while len(processed) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)

    queue.stop()

    assert processed == ["job-high", "job-low"]


def test_cancel_queued_job_skips_processing() -> None:
    processed: list[str] = []
    canceled: list[str] = []

    def process_job(job_id: str) -> None:
        processed.append(job_id)

    def on_cancel(job_id: str, _reason: str) -> None:
        canceled.append(job_id)

    queue = BackgroundJobQueue(
        process_job=process_job,
        max_size=10,
        worker_count=1,
        on_cancel=on_cancel,
    )

    assert queue.enqueue("job-a", priority=5)
    assert queue.enqueue("job-b", priority=5)
    assert queue.cancel("job-b") is True

    queue.start()

    deadline = time.monotonic() + 3
    while len(processed) < 1 and time.monotonic() < deadline:
        time.sleep(0.05)

    queue.stop()

    assert "job-a" in processed
    assert "job-b" not in processed
    assert "job-b" in canceled


def test_cancel_running_job_returns_false() -> None:
    def process_job(_job_id: str) -> None:
        time.sleep(0.4)

    queue = BackgroundJobQueue(process_job=process_job, max_size=10, worker_count=1)
    queue.start()
    assert queue.enqueue("job-running", priority=5)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stats = queue.stats()
        if stats["active_jobs"] > 0:
            break
        time.sleep(0.05)

    assert queue.cancel("job-running") is False
    queue.stop()
