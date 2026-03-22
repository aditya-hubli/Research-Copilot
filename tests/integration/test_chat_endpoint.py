from __future__ import annotations

import uuid


def test_chat_endpoint_returns_answer_and_citations(client, monkeypatch) -> None:
    import backend.api.pipeline_service as pipeline_service

    # Keep indexing deterministic and network-free for tests.
    monkeypatch.setattr(pipeline_service, "extract_pdf_text", lambda *_args, **_kwargs: "")

    user_id = f"chat-{uuid.uuid4()}"
    seed_url = f"http://chat.local/seed-{uuid.uuid4()}.pdf"

    index_response = client.post(
        "/index-paper",
        json={
            "paper_url": seed_url,
            "title": "Chat Seed Paper",
            "abstract": "Transformer graph methods for conversational retrieval.",
            "user_id": user_id,
        },
    )
    assert index_response.status_code == 200
    assert index_response.json()["indexed_chunks"] > 0

    chat_response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "question": "What does the indexed work suggest about transformer graph methods?",
            "history": [
                {"role": "user", "content": "I am exploring graph-based research."},
                {"role": "assistant", "content": "Great, I can help compare methods."},
            ],
            "top_k": 4,
        },
    )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["answer"]
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["evidence_snippets"], list)
    assert payload["evidence_snippets"]
    assert payload["evidence_snippets"][0]["snippet"]
    assert payload["used_context_chunks"] >= 1
    assert 0.0 <= float(payload["support_score"]) <= 1.0
    assert payload["retrieval_mode"] in {"semantic", "lexical"}
    assert isinstance(payload["agent_trace"], list)
    assert payload["agent_trace"]


def test_chat_endpoint_can_auto_index_and_scope_to_current_paper(client, monkeypatch) -> None:
    import backend.api.chat_service as chat_service
    import backend.api.pipeline_service as pipeline_service

    monkeypatch.setattr(
        pipeline_service,
        "extract_pdf_text",
        lambda *_args, **_kwargs: (
            "Introduction The paper introduces a retrieval grounded copilot. "
            "Method The method indexes the currently open paper and answers questions from that paper only. "
            "Results The system returns evidence snippets from the active paper in real time."
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_paper_metadata",
        lambda **_: {"paper_url": "", "pdf_url": "", "canonical_url": "", "title": "", "abstract": ""},
    )

    user_id = f"chat-scope-{uuid.uuid4()}"
    paper_url = f"http://chat.local/current-{uuid.uuid4()}.pdf"

    chat_response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "question": "How does this paper answer questions in real time?",
            "paper_url": paper_url,
            "paper_title": "Current Paper",
            "paper_abstract": "A copilot that answers questions about the active paper.",
            "top_k": 6,
            "ensure_current_paper_indexed": True,
            "current_paper_only": True,
        },
    )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["answer"]
    assert payload["used_context_chunks"] >= 1
    assert payload["citations"]
    assert all(item["url"] == paper_url for item in payload["citations"])
    assert payload["evidence_snippets"]
    assert all(item["url"] == paper_url for item in payload["evidence_snippets"])
    assert payload["agent_trace"]
