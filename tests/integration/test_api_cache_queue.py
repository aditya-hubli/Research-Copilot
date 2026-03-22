from __future__ import annotations

import time
from typing import Any

from backend.api.models import FullStageResult


def _poll_until_terminal(client, job_id: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/analysis-status/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload["status"] in {"complete", "failed", "canceled"}:
            return payload
        time.sleep(0.15)
    return last_payload


def _simple_full_stage() -> FullStageResult:
    return FullStageResult(
        methods=["Transformer"],
        datasets=["Synthetic"],
        related_papers=[],
        research_connections=["A <-> B"],
        research_gaps=["A <-> C"],
        ideas=["Bridge A and C"],
    )


def test_analyze_paper_uses_result_cache_after_first_completion(client, monkeypatch) -> None:
    import backend.api.main as main_module

    call_counter = {"count": 0}

    def fake_run_full_stage(payload, fast_stage, openai_api_key=None):
        call_counter["count"] += 1
        return _simple_full_stage()

    monkeypatch.setattr(main_module, "run_full_stage", fake_run_full_stage)

    body = {
        "paper_url": "http://cache-test.local/paper-1.pdf",
        "title": "Queue Cache Test",
        "abstract": "Transformer graph method queue cache smoke test.",
        "user_id": "cache-user",
    }

    first = client.post("/analyze-paper", json=body)
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["from_cache"] is False
    assert first_payload["status"] == "partial"

    final_payload = _poll_until_terminal(client, first_payload["job_id"])
    assert final_payload["status"] == "complete"

    second = client.post("/analyze-paper", json=body)
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["from_cache"] is True
    assert second_payload["status"] == "complete"

    queue_stats = client.get("/queue-stats")
    assert queue_stats.status_code == 200
    stats_payload = queue_stats.json()
    assert stats_payload["cached_results"] >= 1

    assert call_counter["count"] == 1


def test_analyze_paper_forwards_openai_key_to_full_stage(client, monkeypatch) -> None:
    import backend.api.main as main_module

    captured: dict[str, Any] = {}

    def fake_run_full_stage(payload, fast_stage, openai_api_key=None):
        captured["openai_api_key"] = openai_api_key
        return _simple_full_stage()

    monkeypatch.setattr(main_module, "run_full_stage", fake_run_full_stage)

    body = {
        "paper_url": "http://cache-test.local/key-forward.pdf",
        "title": "Key Forwarding",
        "abstract": "Testing per-request OpenAI key forwarding.",
        "user_id": "key-user",
    }
    key_value = "sk-test-forward"

    response = client.post(
        "/analyze-paper",
        json=body,
        headers={"X-OpenAI-Api-Key": key_value},
    )
    assert response.status_code == 200
    payload = response.json()
    terminal = _poll_until_terminal(client, payload["job_id"])
    assert terminal["status"] == "complete"
    assert captured.get("openai_api_key") == key_value


def test_result_cache_isolated_by_openai_key(client, monkeypatch) -> None:
    import backend.api.main as main_module

    call_counter = {"count": 0}

    def fake_run_full_stage(payload, fast_stage, openai_api_key=None):
        call_counter["count"] += 1
        return _simple_full_stage()

    monkeypatch.setattr(main_module, "run_full_stage", fake_run_full_stage)

    body = {
        "paper_url": "http://cache-test.local/key-cache.pdf",
        "title": "Cache Key Isolation",
        "abstract": "Ensure cache is separated by OpenAI key fingerprint.",
        "user_id": "cache-key-user",
    }

    first = client.post(
        "/analyze-paper",
        json=body,
        headers={"X-OpenAI-Api-Key": "sk-key-1"},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["from_cache"] is False
    assert _poll_until_terminal(client, first_payload["job_id"])["status"] == "complete"

    second = client.post(
        "/analyze-paper",
        json=body,
        headers={"X-OpenAI-Api-Key": "sk-key-1"},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["from_cache"] is True

    third = client.post(
        "/analyze-paper",
        json=body,
        headers={"X-OpenAI-Api-Key": "sk-key-2"},
    )
    assert third.status_code == 200
    third_payload = third.json()
    assert third_payload["from_cache"] is False
    assert _poll_until_terminal(client, third_payload["job_id"])["status"] == "complete"

    assert call_counter["count"] == 2


def test_queue_stats_endpoint_shape(client) -> None:
    response = client.get("/queue-stats")
    assert response.status_code == 200

    payload = response.json()
    for key in ["queue_depth", "active_jobs", "worker_count", "max_queue_size", "cached_results"]:
        assert key in payload
    assert "canceled_jobs" in payload


def test_cancel_job_endpoint_cancels_queued_job(client, monkeypatch) -> None:
    import backend.api.main as main_module

    def slow_full_stage(payload, fast_stage, openai_api_key=None):
        time.sleep(0.5)
        return FullStageResult(
            methods=["Transformer"],
            datasets=[],
            related_papers=[],
            research_connections=[],
            research_gaps=[],
            ideas=[],
        )

    monkeypatch.setattr(main_module, "run_full_stage", slow_full_stage)

    job1 = client.post(
        "/analyze-paper",
        json={
            "paper_url": "http://cancel.local/job1.pdf",
            "title": "Cancel Job One",
            "abstract": "Transformer queue cancel test one.",
            "user_id": "cancel-user",
            "priority": 5,
        },
    ).json()

    job2 = client.post(
        "/analyze-paper",
        json={
            "paper_url": "http://cancel.local/job2.pdf",
            "title": "Cancel Job Two",
            "abstract": "Transformer queue cancel test two.",
            "user_id": "cancel-user",
            "priority": 5,
        },
    ).json()

    cancel_response = client.post(f"/cancel-job/{job2['job_id']}")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["canceled"] is True

    terminal = _poll_until_terminal(client, job2["job_id"], timeout_seconds=8)
    assert terminal["status"] == "canceled"

    # Ensure first job still completes.
    terminal1 = _poll_until_terminal(client, job1["job_id"], timeout_seconds=8)
    assert terminal1["status"] == "complete"
