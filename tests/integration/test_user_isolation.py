from __future__ import annotations

import time
import uuid
from typing import Any


def _poll_until_terminal(client, job_id: str, timeout_seconds: float = 12.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/analysis-status/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload["status"] in {"complete", "failed", "canceled"}:
            return payload
        time.sleep(0.2)
    return last_payload


def test_related_retrieval_is_user_isolated(client, monkeypatch) -> None:
    import backend.api.pipeline_service as pipeline_service

    # Keep ingestion deterministic and avoid external network in CI.
    monkeypatch.setattr(pipeline_service, "extract_pdf_text", lambda *_args, **_kwargs: "")

    alpha_user = f"alpha-{uuid.uuid4()}"
    beta_user = f"beta-{uuid.uuid4()}"
    alpha_seed_url = f"http://isolation.local/alpha-seed-{uuid.uuid4()}.pdf"

    seed_resp = client.post(
        "/index-paper",
        json={
            "paper_url": alpha_seed_url,
            "title": "Alpha Seed Paper",
            "abstract": "Transformer graph methods for isolation testing.",
            "user_id": alpha_user,
        },
    )
    assert seed_resp.status_code == 200
    assert seed_resp.json()["indexed_chunks"] > 0

    beta_start = client.post(
        "/analyze-paper",
        json={
            "paper_url": f"http://isolation.local/beta-query-{uuid.uuid4()}.pdf",
            "title": "Beta Query",
            "abstract": "Transformer graph methods for isolation testing.",
            "user_id": beta_user,
        },
    )
    assert beta_start.status_code == 200
    beta_final = _poll_until_terminal(client, beta_start.json()["job_id"])
    assert beta_final["status"] == "complete"

    beta_urls = [item["url"] for item in beta_final["related_papers"]]
    assert alpha_seed_url not in beta_urls

    alpha_start = client.post(
        "/analyze-paper",
        json={
            "paper_url": f"http://isolation.local/alpha-query-{uuid.uuid4()}.pdf",
            "title": "Alpha Query",
            "abstract": "Transformer graph methods for isolation testing.",
            "user_id": alpha_user,
        },
    )
    assert alpha_start.status_code == 200
    alpha_final = _poll_until_terminal(client, alpha_start.json()["job_id"])
    assert alpha_final["status"] == "complete"

    alpha_urls = [item["url"] for item in alpha_final["related_papers"]]
    assert alpha_seed_url in alpha_urls


def test_index_stats_can_scope_by_user(client, monkeypatch) -> None:
    import backend.api.pipeline_service as pipeline_service

    monkeypatch.setattr(pipeline_service, "extract_pdf_text", lambda *_args, **_kwargs: "")

    user_a = f"stats-a-{uuid.uuid4()}"
    user_b = f"stats-b-{uuid.uuid4()}"

    for user_id, suffix in [(user_a, "a"), (user_b, "b")]:
        response = client.post(
            "/index-paper",
            json={
                "paper_url": f"http://stats.local/{suffix}-{uuid.uuid4()}.pdf",
                "title": f"Stats Paper {suffix}",
                "abstract": "Transformer graph methods for stats testing.",
                "user_id": user_id,
            },
        )
        assert response.status_code == 200

    user_a_stats = client.get(f"/index-stats?user_id={user_a}")
    assert user_a_stats.status_code == 200
    assert user_a_stats.json()["unique_papers"] >= 1

    user_b_stats = client.get(f"/index-stats?user_id={user_b}")
    assert user_b_stats.status_code == 200
    assert user_b_stats.json()["unique_papers"] >= 1
