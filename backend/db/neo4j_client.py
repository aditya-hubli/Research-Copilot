from __future__ import annotations

from functools import lru_cache

from neo4j import GraphDatabase

from backend.core.config import get_settings


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
            )
        return self._driver

    def upsert_paper_graph(
        self,
        user_id: str,
        paper_url: str,
        title: str,
        concepts: list[str],
        methods: list[str],
        datasets: list[str],
    ) -> dict[str, object]:
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (p:Paper {url: $paper_url})
        SET p.title = $title,
            p.updated_at = timestamp()
        MERGE (u)-[reads:READS]->(p)
        ON CREATE SET reads.first_seen_at = timestamp(),
                      reads.read_count = 0
        SET reads.last_seen_at = timestamp(),
            reads.read_count = coalesce(reads.read_count, 0) + 1

        WITH u, p,
             [item IN $concepts WHERE trim(item) <> ""] AS concept_names,
             [item IN $methods WHERE trim(item) <> ""] AS method_names,
             [item IN $datasets WHERE trim(item) <> ""] AS dataset_names

        FOREACH (concept_name IN concept_names |
          MERGE (c:Concept {name: concept_name})
          MERGE (p)-[:INTRODUCES]->(c)
          MERGE (u)-[interest:INTERESTED_IN]->(c)
          ON CREATE SET interest.weight = 0,
                        interest.first_seen_at = timestamp()
          SET interest.weight = coalesce(interest.weight, 0) + 1,
              interest.last_seen_at = timestamp()
        )

        FOREACH (method_name IN method_names |
          MERGE (m:Method {name: method_name})
          MERGE (p)-[:USES_METHOD]->(m)
        )

        FOREACH (dataset_name IN dataset_names |
          MERGE (d:Dataset {name: dataset_name})
          MERGE (p)-[:USES_DATASET]->(d)
        )

        FOREACH (idx IN CASE WHEN size(concept_names) > 1 THEN range(0, size(concept_names) - 2) ELSE [] END |
          FOREACH (jdx IN range(idx + 1, size(concept_names) - 1) |
            MERGE (left_concept:Concept {name: concept_names[idx]})
            MERGE (right_concept:Concept {name: concept_names[jdx]})
            MERGE (left_concept)-[rel:RELATED_TO]-(right_concept)
            ON CREATE SET rel.paper_count = 0,
                          rel.created_at = timestamp()
            SET rel.paper_count = coalesce(rel.paper_count, 0) + 1,
                rel.last_seen_at = timestamp()
          )
        )
        RETURN p.url AS url
        """
        try:
            with self._get_driver().session() as session:
                session.run(
                    query,
                    user_id=user_id,
                    paper_url=paper_url,
                    title=title,
                    concepts=concepts,
                    methods=methods,
                    datasets=datasets,
                )
            return {"graph_updated": True}
        except Exception as exc:
            return {"graph_updated": False, "error": str(exc)}

    def query_weak_connections(self, concepts: list[str], limit: int = 3) -> list[str]:
        if not concepts:
            return []

        query = """
        MATCH (c1:Concept)-[r:RELATED_TO]-(c2:Concept)
        WHERE c1.name IN $concepts AND c2.name IN $concepts
        RETURN c1.name AS left_name,
               c2.name AS right_name,
               coalesce(r.paper_count, 0) AS strength
        ORDER BY strength DESC, left_name ASC, right_name ASC
        LIMIT $limit
        """

        try:
            with self._get_driver().session() as session:
                records = session.run(query, concepts=concepts, limit=limit)
                pairs = [
                    f"{record['left_name']} <-> {record['right_name']} (shared papers: {int(record['strength'])})"
                    for record in records
                    if int(record["strength"]) >= 2
                ]
            return pairs
        except Exception:
            return []

    def query_user_interests(self, user_id: str, limit: int = 5) -> list[str]:
        if not str(user_id or "").strip():
            return []

        query = """
        MATCH (:User {id: $user_id})-[interest:INTERESTED_IN]->(c:Concept)
        RETURN c.name AS concept_name, coalesce(interest.weight, 0) AS weight
        ORDER BY weight DESC, concept_name ASC
        LIMIT $limit
        """

        try:
            with self._get_driver().session() as session:
                records = session.run(query, user_id=user_id, limit=limit)
                return [str(record["concept_name"]) for record in records if str(record["concept_name"]).strip()]
        except Exception:
            return []

    def upsert_citation_edge(
        self,
        citing_url: str,
        cited_url: str,
        cited_title: str,
        user_id: str,
    ) -> None:
        """
        Ensure both Paper nodes exist and write a directed CITES edge between them.
        Also writes a lightweight RELATED_TO edge on Paper nodes for graph queries.
        """
        query = """
        MERGE (citing:Paper {url: $citing_url})
        MERGE (cited:Paper  {url: $cited_url})
        SET cited.title = CASE
              WHEN $cited_title <> '' AND (cited.title IS NULL OR cited.title = '')
              THEN $cited_title ELSE cited.title END,
            cited.discovered_by = $user_id,
            cited.updated_at = timestamp()
        MERGE (citing)-[c:CITES]->(cited)
        ON CREATE SET c.created_at  = timestamp(),
                      c.discovered_by = $user_id
        SET c.last_seen_at = timestamp()
        """
        try:
            with self._get_driver().session() as session:
                session.run(
                    query,
                    citing_url=citing_url,
                    cited_url=cited_url,
                    cited_title=cited_title or "",
                    user_id=user_id,
                )
        except Exception as exc:
            # Non-fatal — log at debug so it doesn't spam the console
            import logging
            logging.getLogger(__name__).debug("upsert_citation_edge failed: %s", exc)

    def healthcheck(self) -> dict[str, object]:
        try:
            with self._get_driver().session() as session:
                record = session.run("RETURN 1 AS ok").single()
            return {"ok": bool(record and record.get("ok") == 1), "detail": "neo4j reachable"}
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()


@lru_cache(maxsize=1)
def get_neo4j_client() -> Neo4jClient:
    settings = get_settings()
    return Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
