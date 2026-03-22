from __future__ import annotations

from pathlib import Path

from backend.db.faiss_store import FaissStore


def test_faiss_store_persists_and_reloads(tmp_path: Path) -> None:
    data_dir = tmp_path / "faiss-store"

    store = FaissStore(dimension=8, data_dir=str(data_dir))
    store.add(
        embeddings=[[1.0] * 8, [0.5] * 8],
        texts=["paper-a chunk", "paper-b chunk"],
        metadatas=[
            {"paper_url": "http://a", "user_id": "u1"},
            {"paper_url": "http://b", "user_id": "u2"},
        ],
    )

    assert store.vector_count() == 2
    assert store.unique_paper_count() == 2
    assert store.contains_paper_url("http://a", user_id="u1") is True
    assert store.contains_paper_url("http://a", user_id="u2") is False

    reloaded = FaissStore(dimension=8, data_dir=str(data_dir))
    assert reloaded.vector_count() == 2
    assert reloaded.unique_paper_count() == 2
    assert reloaded.unique_paper_count(user_id="u1") == 1

    results = reloaded.search(query_embedding=[1.0] * 8, top_k=2)
    assert len(results) >= 1
