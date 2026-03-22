from __future__ import annotations

from backend.api.chat_service import run_chat_query
from backend.api.models import ChatRequest


def test_chat_query_seed_uses_question_not_title_or_url(monkeypatch) -> None:
    captured: list[str] = []

    def fake_retrieve_context_chunks(**kwargs):
        captured.append(kwargs["query"])
        return [
            {
                "text": "The paper answers questions from indexed body chunks.",
                "paper_url": "http://paper.local/current.pdf",
                "title": "Current Paper",
                "section": "Body",
                "score": 0.91,
            }
        ]

    monkeypatch.setattr("backend.api.chat_service.retrieve_context_chunks", fake_retrieve_context_chunks)
    monkeypatch.setattr("backend.api.chat_service.query_graph", lambda **_: {"graph_connections": [], "user_interest_topics": []})
    monkeypatch.setattr("backend.api.chat_service.call_structured_agent", lambda **_: None)
    monkeypatch.setattr("backend.api.chat_service._paper_candidates", lambda *_args, **_kwargs: [])

    payload = ChatRequest(
        user_id="unit-user",
        question="How does the method use indexed body chunks?",
        paper_url="http://paper.local/current.pdf",
        paper_title="Very Tempting Title Match",
        paper_abstract="An abstract that should not be appended to the retrieval query.",
        top_k=4,
    )

    response = run_chat_query(payload)

    assert response.answer
    assert captured
    assert captured[0] == "How does the method use indexed body chunks?"
    assert all("Very Tempting Title Match" not in item for item in captured)
    assert all("http://paper.local/current.pdf" not in item for item in captured)
