from __future__ import annotations

from backend.tools import graph_tools


class _FakeNeo4jClient:
    def __init__(self) -> None:
        self.last_update: dict[str, object] | None = None

    def upsert_paper_graph(
        self,
        user_id: str,
        paper_url: str,
        title: str,
        concepts: list[str],
        methods: list[str],
        datasets: list[str],
    ) -> dict[str, object]:
        self.last_update = {
            "user_id": user_id,
            "paper_url": paper_url,
            "title": title,
            "concepts": concepts,
            "methods": methods,
            "datasets": datasets,
        }
        return {"graph_updated": True}

    def query_weak_connections(self, concepts: list[str], limit: int = 3) -> list[str]:
        return [f"{concepts[0]} <-> {concepts[1]} (shared papers: 2)"] if len(concepts) >= 2 else []

    def query_user_interests(self, user_id: str, limit: int = 5) -> list[str]:
        if not user_id:
            return []
        return ["Graph Neural Networks", "Drug Discovery"][:limit]


def test_update_graph_forwards_datasets(monkeypatch) -> None:
    fake_client = _FakeNeo4jClient()
    monkeypatch.setattr(graph_tools, "get_neo4j_client", lambda: fake_client)

    result = graph_tools.update_graph(
        user_id="user-1",
        paper_url="https://example.org/paper.pdf",
        title="Example Paper",
        concepts=["Graph Neural Networks"],
        methods=["Transformer"],
        datasets=["PubMed"],
    )

    assert result["graph_updated"] is True
    assert fake_client.last_update is not None
    assert fake_client.last_update["datasets"] == ["PubMed"]


def test_query_graph_returns_connections_and_user_interests(monkeypatch) -> None:
    fake_client = _FakeNeo4jClient()
    monkeypatch.setattr(graph_tools, "get_neo4j_client", lambda: fake_client)

    result = graph_tools.query_graph(
        concepts=["Graph Neural Networks", "Drug Discovery"],
        user_id="user-1",
        limit=3,
        interest_limit=5,
    )

    assert result["graph_connections"] == [
        "Graph Neural Networks <-> Drug Discovery (shared papers: 2)"
    ]
    assert result["user_interest_topics"] == [
        "Graph Neural Networks",
        "Drug Discovery",
    ]
