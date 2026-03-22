from __future__ import annotations

from unittest.mock import MagicMock

from backend.agents import workflow
from backend.review_store import GraphReviewStore


def test_graph_update_node_queues_review_when_enabled(monkeypatch) -> None:
    fake_store = GraphReviewStore()
    monkeypatch.setattr(workflow, "graph_review_store", fake_store)

    fake_settings = MagicMock()
    fake_settings.require_graph_review = True
    monkeypatch.setattr(workflow, "get_settings", lambda: fake_settings)

    result = workflow.graph_update_node(
        {
            "user_id": "review-user",
            "paper_url": "https://example.org/review.pdf",
            "title": "Review Paper",
            "concepts": ["Graph Neural Networks"],
            "methods": ["Transformer"],
            "datasets": ["PubMed"],
            "autonomy_notes": [],
        }
    )

    reviews = fake_store.list()
    assert result["review_required"] is True
    assert result["graph_updated"] is False
    assert reviews
    assert result["review_id"] == reviews[0].review_id
