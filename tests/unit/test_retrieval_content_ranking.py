from __future__ import annotations

from backend.tools.vector_tools import retrieve_context_chunks, vector_search


class _FakeStore:
    dimension = 3

    def __init__(self, hits):
        self._hits = hits

    def search(self, query_embedding, top_k=5):
        return list(self._hits)[:top_k]

    def records(self, user_id=None):
        return []


def test_retrieve_context_chunks_penalizes_title_only_hits(monkeypatch) -> None:
    hits = [
        {
            "score": 0.99,
            "text": "Graph Retrieval Copilot",
            "metadata": {
                "paper_url": "http://paper.local/a",
                "title": "Graph Retrieval Copilot",
                "user_id": "u1",
                "section": "Title",
            },
        },
        {
            "score": 0.72,
            "text": "The method retrieves body chunks semantically and answers from evidence in the paper content.",
            "metadata": {
                "paper_url": "http://paper.local/a",
                "title": "Graph Retrieval Copilot",
                "user_id": "u1",
                "section": "Body",
            },
        },
    ]

    monkeypatch.setattr("backend.tools.vector_tools.get_faiss_store", lambda: _FakeStore(hits))
    monkeypatch.setattr("backend.tools.vector_tools.create_embedding", lambda *args, **kwargs: [0.0, 0.0, 0.0])
    monkeypatch.setattr("backend.tools.vector_tools.has_embedding_provider", lambda *_args, **_kwargs: True)

    results = retrieve_context_chunks(
        query="How does the method answer from paper content?",
        top_k=2,
        user_id="u1",
    )

    assert results
    assert results[0]["section"] == "Body"


def test_vector_search_prefers_content_supported_paper_over_title_match(monkeypatch) -> None:
    hits = [
        {
            "score": 0.97,
            "text": "Semantic Paper Matching",
            "metadata": {
                "paper_url": "http://paper.local/title-only",
                "title": "Semantic Paper Matching",
                "user_id": "u1",
                "section": "Title",
            },
        },
        {
            "score": 0.74,
            "text": "This paper compares semantic chunk retrieval, evidence ranking, and database-grounded question answering.",
            "metadata": {
                "paper_url": "http://paper.local/body-match",
                "title": "Different Title Entirely",
                "user_id": "u1",
                "section": "Body",
            },
        },
    ]

    monkeypatch.setattr("backend.tools.vector_tools.get_faiss_store", lambda: _FakeStore(hits))
    monkeypatch.setattr("backend.tools.vector_tools.create_embedding", lambda *args, **kwargs: [0.0, 0.0, 0.0])
    monkeypatch.setattr("backend.tools.vector_tools.has_embedding_provider", lambda *_args, **_kwargs: True)

    results = vector_search(
        query="database grounded question answering from semantic chunk retrieval",
        top_k=2,
        user_id="u1",
    )

    assert results
    assert results[0]["url"] == "http://paper.local/body-match"
