from __future__ import annotations

from pathlib import Path

from backend.review_store import GraphReviewStore


def test_review_store_persists_and_reloads(tmp_path: Path) -> None:
    data_path = tmp_path / "reviews" / "reviews.json"

    store = GraphReviewStore(data_path=str(data_path))
    created = store.create_review(
        {
            "user_id": "u1",
            "paper_url": "https://example.org/paper.pdf",
            "title": "Example Paper",
            "concepts": ["Graph Neural Networks"],
        }
    )
    store.set_status(created.review_id, "approved", note="Looks good")

    reloaded = GraphReviewStore(data_path=str(data_path))
    review = reloaded.get(created.review_id)

    assert review is not None
    assert review.status == "approved"
    assert review.note == "Looks good"
    assert reloaded.counts()["approved"] == 1
