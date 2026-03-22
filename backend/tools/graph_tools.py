from __future__ import annotations

from backend.db.neo4j_client import get_neo4j_client


def update_graph(
    user_id: str,
    paper_url: str,
    title: str,
    concepts: list[str],
    methods: list[str],
    datasets: list[str],
) -> dict[str, object]:
    client = get_neo4j_client()
    return client.upsert_paper_graph(
        user_id=user_id,
        paper_url=paper_url,
        title=title,
        concepts=concepts,
        methods=methods,
        datasets=datasets,
    )


def query_graph(
    concepts: list[str],
    user_id: str | None = None,
    limit: int = 3,
    interest_limit: int = 5,
) -> dict[str, list[str]]:
    client = get_neo4j_client()
    connections = client.query_weak_connections(concepts=concepts, limit=limit)
    interest_topics = client.query_user_interests(user_id=user_id or "", limit=interest_limit)
    return {
        "graph_connections": connections,
        "user_interest_topics": interest_topics,
    }
