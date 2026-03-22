from __future__ import annotations

from typing import Any, TypedDict


class SharedState(TypedDict, total=False):
    user_id: str
    paper_url: str
    title: str
    abstract: str
    summary: str
    methods: list[str]
    datasets: list[str]
    concepts: list[str]
    retrieval_queries: list[str]
    planner: dict[str, Any]
    autonomy_notes: list[str]
    confidence: float
    context_chunks: list[dict[str, Any]]
    # Both key names supported — llm_api_key is the new unified name
    openai_api_key: str
    llm_api_key: str
    llm_provider: str
    related_papers: list[dict[str, Any]]
    graph_connections: list[str]
    user_interest_topics: list[str]
    research_gaps: list[str]
    ideas: list[str]
    graph_updated: bool
    review_required: bool
    review_id: str


def with_defaults(state: dict[str, Any]) -> SharedState:
    # Support both old (openai_api_key) and new (llm_api_key) key names
    llm_key = str(state.get("llm_api_key") or state.get("openai_api_key") or "").strip()
    return {
        "user_id":             state.get("user_id", "local-user"),
        "paper_url":           state.get("paper_url", ""),
        "title":               state.get("title", ""),
        "abstract":            state.get("abstract", ""),
        "summary":             state.get("summary", ""),
        "methods":             list(state.get("methods", [])),
        "datasets":            list(state.get("datasets", [])),
        "concepts":            list(state.get("concepts", [])),
        "retrieval_queries":   list(state.get("retrieval_queries", [])),
        "planner":             dict(state.get("planner", {})),
        "autonomy_notes":      list(state.get("autonomy_notes", [])),
        "confidence":          float(state.get("confidence", 0.0) or 0.0),
        "context_chunks":      list(state.get("context_chunks", [])),
        "openai_api_key":      llm_key,   # legacy compat
        "llm_api_key":         llm_key,   # new unified name
        "llm_provider":        str(state.get("llm_provider", "") or "").strip(),
        "related_papers":      list(state.get("related_papers", [])),
        "graph_connections":   list(state.get("graph_connections", [])),
        "user_interest_topics": list(state.get("user_interest_topics", [])),
        "research_gaps":       list(state.get("research_gaps", [])),
        "ideas":               list(state.get("ideas", [])),
        "graph_updated":       bool(state.get("graph_updated", False)),
        "review_required":     bool(state.get("review_required", False)),
        "review_id":           str(state.get("review_id", "") or ""),
    }
