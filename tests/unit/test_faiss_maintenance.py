from __future__ import annotations

from pathlib import Path

from backend.db.faiss_store import FaissStore


def test_faiss_store_compact_removes_duplicate_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "faiss-compact"

    store = FaissStore(dimension=8, data_dir=str(data_dir))
    store.add(
        embeddings=[[1.0] * 8, [1.0] * 8, [0.25] * 8],
        texts=["duplicate chunk", "duplicate chunk", "unique chunk"],
        metadatas=[
            {"paper_url": "http://paper-a", "title": "Paper A", "user_id": "u1", "section": "Abstract"},
            {"paper_url": "http://paper-a", "title": "Paper A", "user_id": "u1", "section": "Abstract"},
            {"paper_url": "http://paper-b", "title": "Paper B", "user_id": "u1", "section": "Body"},
        ],
    )

    result = store.compact()

    assert result["vectors_before"] == 3
    assert result["vectors_after"] == 2
    assert result["duplicates_removed"] == 1
    assert result["unique_papers_after"] == 2

    reloaded = FaissStore(dimension=8, data_dir=str(data_dir))
    assert reloaded.vector_count() == 2
    assert reloaded.unique_paper_count() == 2
