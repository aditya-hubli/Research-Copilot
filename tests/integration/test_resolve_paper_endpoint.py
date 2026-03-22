from __future__ import annotations


def test_resolve_paper_endpoint_returns_provider_enriched_payload(client, monkeypatch) -> None:
    import backend.api.main as api_main

    def fake_resolve_paper_metadata(paper_url: str, title: str = "", abstract: str = "") -> dict[str, str]:
        return {
            "paper_url": paper_url,
            "source": "arxiv",
            "host": "arxiv.org",
            "provider": "arxiv",
            "providers": "arxiv,semantic-scholar",
            "title": title or "Resolved Paper Title",
            "abstract": abstract or "Resolved abstract text.",
            "canonical_url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            "arxiv_id": "2401.00001",
            "doi": "",
        }

    monkeypatch.setattr(api_main, "resolve_paper_metadata", fake_resolve_paper_metadata)

    response = client.get(
        "/resolve-paper",
        params={
            "paper_url": "https://arxiv.org/abs/2401.00001",
            "title": "",
            "abstract": "",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["paper_url"] == "https://arxiv.org/abs/2401.00001"
    assert payload["resolved_url"] == "https://arxiv.org/pdf/2401.00001.pdf"
    assert payload["provider"] == "arxiv"
    assert payload["providers"] == ["arxiv", "semantic-scholar"]
    assert payload["title"] == "Resolved Paper Title"
    assert payload["abstract"] == "Resolved abstract text."


def test_related_papers_preview_endpoint_returns_title_search_results(client, monkeypatch) -> None:
    import backend.api.main as api_main

    monkeypatch.setattr(
        api_main,
        "search_related_papers_by_title",
        lambda title, paper_url="", abstract="", limit=5: {
            "provider": "semantic-scholar,openalex",
            "related_papers": [
                {
                    "title": "Attention with Retrieval",
                    "url": "https://example.org/paper-a",
                    "score": 0.83,
                },
                {
                    "title": "Realtime Paper Copilots",
                    "url": "https://example.org/paper-b",
                    "score": 0.77,
                },
            ],
        },
    )

    response = client.get(
        "/related-papers-preview",
        params={
            "title": "Attention Is All You Need",
            "paper_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Attention Is All You Need"
    assert payload["provider"] == "semantic-scholar,openalex"
    assert len(payload["related_papers"]) == 2
    assert payload["related_papers"][0]["title"] == "Attention with Retrieval"
