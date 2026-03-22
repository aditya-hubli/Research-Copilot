from __future__ import annotations


def test_readiness_endpoint_exposes_component_status(client) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ok", "degraded", "not_ready"}
    assert isinstance(payload["ready"], bool)
    assert isinstance(payload["degraded"], bool)
    assert "queue" in payload["components"]
    assert "vector_store" in payload["components"]
    assert "review_store" in payload["components"]
    assert "graph_store" in payload["components"]
    assert response.headers["x-request-id"]
    assert response.headers["x-process-time-ms"]


def test_metrics_summary_endpoint_shape(client) -> None:
    response = client.get("/metrics/summary")
    assert response.status_code == 200

    payload = response.json()
    assert payload["app_name"]
    assert payload["app_env"]
    assert "queue" in payload
    assert "caches" in payload
    assert "index" in payload
    assert "jobs" in payload
    assert "reviews" in payload
    assert "require_graph_review" in payload
    assert response.headers["x-request-id"]
    assert response.headers["x-process-time-ms"]


def test_graph_review_endpoints_support_approve_and_reject(client, monkeypatch) -> None:
    import backend.api.main as main_module
    from backend.review_store import graph_review_store

    first = graph_review_store.create_review(
        {
            "user_id": "review-user",
            "paper_url": "https://example.org/paper-a.pdf",
            "title": "Paper A",
            "concepts": ["Graph Neural Networks"],
            "methods": ["Transformer"],
            "datasets": ["PubMed"],
        }
    )
    second = graph_review_store.create_review(
        {
            "user_id": "review-user",
            "paper_url": "https://example.org/paper-b.pdf",
            "title": "Paper B",
            "concepts": ["Drug Discovery"],
            "methods": ["Graph Neural Network"],
            "datasets": ["MIMIC"],
        }
    )

    monkeypatch.setattr(
        main_module,
        "update_graph",
        lambda **_: {"graph_updated": True},
    )

    listed = client.get("/graph-reviews")
    assert listed.status_code == 200
    assert len(listed.json()["reviews"]) >= 2

    approved = client.post(f"/graph-reviews/{first.review_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    rejected = client.post(f"/graph-reviews/{second.review_id}/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    pending = client.get("/graph-reviews?status=pending")
    assert pending.status_code == 200
    pending_ids = {item["review_id"] for item in pending.json()["reviews"]}
    assert first.review_id not in pending_ids
    assert second.review_id not in pending_ids
