from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from backend.agents.llm_runtime import call_structured_agent
from backend.agents.prompts import (
    CONCEPT_SYSTEM_PROMPT,
    GAP_SYSTEM_PROMPT,
    IDEA_SYSTEM_PROMPT,
    PARSER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from backend.agents.state import SharedState, with_defaults
from backend.core.config import get_settings
from backend.review_store import graph_review_store
from backend.tools.graph_tools import query_graph, update_graph
from backend.tools.vector_tools import vector_search

logger = logging.getLogger(__name__)

# ── NEVER cache settings at module level — always call get_settings() inside functions ──


def _first_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words]).strip()


def _sanitize_str_list(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _append_notes(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {item.strip().lower() for item in merged if item}
    for note in additions:
        normalized = str(note).strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        merged.append(normalized)
    return merged


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        return min(max(int(value), minimum), maximum)
    except Exception:
        return default


def _context_snippets(state: SharedState, max_items: int = 4, max_chars: int = 500) -> list[dict[str, str]]:
    context_chunks = state.get("context_chunks", [])
    snippets: list[dict[str, str]] = []
    if not isinstance(context_chunks, list):
        return snippets
    for raw in context_chunks:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        snippets.append({
            "title":     str(raw.get("title", "Untitled")),
            "paper_url": str(raw.get("paper_url", "")),
            "section":   str(raw.get("section", "Body")),
            "text":      text[:max_chars],
        })
        if len(snippets) >= max_items:
            break
    return snippets


def _planner_queries(state: SharedState) -> list[str]:
    settings = get_settings()
    title    = str(state.get("title", "")).strip()
    abstract = str(state.get("abstract", "")).strip()
    summary  = str(state.get("summary", "")).strip()

    base     = " ".join(p for p in [title, summary] if p).strip()
    fallback = " ".join(p for p in [title, abstract] if p).strip()
    query    = base or fallback
    if not query:
        return []

    variants = [query]
    tokens = [t for t in (title + " " + abstract).split() if len(t) >= 5]
    if tokens:
        variants.append(" ".join(tokens[:8]))
    if len(tokens) >= 10:
        variants.append(" ".join(tokens[-8:]))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = " ".join(variant.split()).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(" ".join(variant.split()).strip())
    return deduped[:max(1, settings.retrieval_query_expansions)]


def _get_llm_key(state: SharedState) -> str:
    """
    Return the LLM API key from state, trying both the new unified
    key name ('llm_api_key') and the legacy name ('openai_api_key').
    """
    return str(state.get("llm_api_key") or state.get("openai_api_key") or "").strip()


def _heuristic_plan(state: SharedState) -> dict[str, Any]:
    settings = get_settings()
    context_count = len(state.get("context_chunks", []))
    weak_context = context_count < 3
    return {
        "planner": {
            "needs_related_papers": True,
            "needs_gap_analysis": True,
            "needs_idea_generation": True,
            "related_paper_k": 5,
            "retrieval_depth": 10 if weak_context else 6,
            "agent_retry_limit": settings.agent_retry_limit,
            "reason": "heuristic plan",
        },
        "retrieval_queries": _planner_queries(state),
        "autonomy_notes": [
            "Planner: heuristic routing.",
            "Planner: increased retrieval depth (weak context)." if weak_context
            else "Planner: normal retrieval depth.",
        ],
    }


def _merge_plan(base_plan: dict[str, Any], llm_plan: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    merged = dict(base_plan)
    for bool_key in ("needs_related_papers", "needs_gap_analysis", "needs_idea_generation"):
        if bool_key in llm_plan:
            merged[bool_key] = bool(llm_plan[bool_key])
    merged["related_paper_k"]   = _clamp_int(llm_plan.get("related_paper_k",   merged["related_paper_k"]),   3, 8, 5)
    merged["retrieval_depth"]   = _clamp_int(llm_plan.get("retrieval_depth",    merged["retrieval_depth"]),   4, 16, 8)
    merged["agent_retry_limit"] = _clamp_int(llm_plan.get("agent_retry_limit",  merged["agent_retry_limit"]), 1, 4, settings.agent_retry_limit)
    if llm_plan.get("reason"):
        merged["reason"] = str(llm_plan["reason"]).strip()
    return merged


def _agent_retry_limit(state: SharedState) -> int:
    settings = get_settings()
    return _clamp_int(state.get("planner", {}).get("agent_retry_limit", settings.agent_retry_limit), 1, 4, settings.agent_retry_limit)


def _heuristic_parser(state: SharedState) -> dict[str, Any]:
    summary = state.get("summary") or _first_words(state.get("abstract", ""), 120)
    abstract_lower = state.get("abstract", "").lower()
    methods  = [h for h in ["Transformer", "RLHF", "Reinforcement Learning", "Diffusion Model",
                             "Graph Neural Network", "Contrastive Learning", "In-Context Learning",
                             "Chain-of-Thought"] if h.lower() in abstract_lower][:5]
    datasets = [h for h in ["ImageNet", "CIFAR-10", "COCO", "SQuAD", "HumanEval", "GSM8K",
                              "MMLU", "BIG-Bench", "PubMed"] if h.lower() in abstract_lower][:5]
    return {"summary": summary, "methods": methods, "datasets": datasets}


def _heuristic_concepts(state: SharedState) -> list[str]:
    seed = _sanitize_str_list(state.get("concepts", []), max_items=5)
    if seed:
        return seed
    text = f"{state.get('title', '')} {state.get('summary', '')}"
    concepts: list[str] = []
    seen: set[str] = set()
    stopwords = {"this", "that", "with", "from", "have", "been", "their", "which", "paper", "model"}
    for token in text.split():
        t = token.strip(".,:;()[]\"'")
        if len(t) < 4 or t.lower() in seen or t.lower() in stopwords:
            continue
        seen.add(t.lower())
        concepts.append(t)
        if len(concepts) >= 5:
            break
    return concepts


def _heuristic_gaps(concepts: list[str], related_papers: list[dict[str, Any]], graph_connections: list[str]) -> list[str]:
    gaps: list[str] = []
    if len(concepts) >= 2:
        for i in range(min(3, len(concepts) - 1)):
            pair = f"{concepts[i]} ↔ {concepts[i + 1]}"
            if pair not in graph_connections:
                gaps.append(f"Limited work combining {concepts[i]} and {concepts[i + 1]} in a unified framework.")
    if not gaps and concepts:
        gaps.append(f"Scalability and deployment challenges for {concepts[0]} in real-world settings remain underexplored.")
    return gaps[:3]


def _heuristic_ideas(concepts: list[str], gaps: list[str]) -> list[str]:
    ideas: list[str] = []
    for gap in gaps[:2]:
        ideas.append(f"Research direction addressing: {gap}")
    if not ideas and concepts:
        ideas.append(f"Develop a robust evaluation benchmark for {concepts[0]} that tests out-of-distribution generalization.")
    if len(ideas) < 2 and len(concepts) >= 2:
        ideas.append(f"Investigate cross-domain transfer between {concepts[0]} and {concepts[1]} using meta-learning.")
    return ideas[:3]


def _autonomous_agent_call(
    state: SharedState,
    *,
    system_prompt: str,
    payload: dict[str, Any],
    max_output_tokens: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    retry_limit = _agent_retry_limit(state)
    notes: list[str] = []
    context_total = max(1, len(state.get("context_chunks", [])))
    llm_key = _get_llm_key(state)

    for attempt in range(1, retry_limit + 1):
        context_window  = min(context_total, 2 + (attempt * 2))
        attempt_payload = dict(payload)
        attempt_payload["retrieval_context"] = _context_snippets(state, max_items=context_window)
        attempt_payload["attempt"] = attempt

        result = call_structured_agent(
            system_prompt=system_prompt,
            payload=attempt_payload,
            max_output_tokens=max_output_tokens,
            llm_api_key=llm_key,   # unified key — provider auto-detected
        )
        if result:
            if attempt > 1:
                notes.append(f"Agent recovered on attempt {attempt}.")
            return result, notes
        notes.append(f"Attempt {attempt} returned no output — retrying.")

    notes.append("All retry attempts exhausted — using heuristic fallback.")
    return None, notes


# ── Agent nodes ───────────────────────────────────────────────────────────────

def planner_node(state: SharedState) -> SharedState:
    heuristic = _heuristic_plan(state)
    context_count = len(state.get("context_chunks", []))

    llm_response, retry_notes = _autonomous_agent_call(
        state,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        payload={
            "title": state.get("title", ""),
            "abstract": state.get("abstract", ""),
            "summary": state.get("summary", ""),
            "context_count": context_count,
            "candidate_queries": heuristic["retrieval_queries"],
        },
        max_output_tokens=512,
    )

    planner = heuristic["planner"]
    retrieval_queries = list(heuristic["retrieval_queries"])
    notes = _append_notes(list(state.get("autonomy_notes", [])), list(heuristic.get("autonomy_notes", [])) + retry_notes)

    if isinstance(llm_response, dict):
        planner = _merge_plan(planner, llm_response)
        llm_queries = _sanitize_str_list(llm_response.get("retrieval_queries", []), max_items=4)
        if llm_queries:
            retrieval_queries = llm_queries
        notes = _append_notes(notes, ["Planner: LLM-guided routing active."])

    confidence = min(0.95, 0.45 + (0.08 if context_count else 0.0) + (0.1 if llm_response else 0.0))
    return {"planner": planner, "retrieval_queries": retrieval_queries, "autonomy_notes": notes,
            "confidence": max(float(state.get("confidence", 0.0)), confidence)}


def parser_node(state: SharedState) -> SharedState:
    heuristic = _heuristic_parser(state)
    llm_response, retry_notes = _autonomous_agent_call(
        state,
        system_prompt=PARSER_SYSTEM_PROMPT,
        payload={"paper_url": state.get("paper_url", ""), "title": state.get("title", ""),
                 "abstract": state.get("abstract", ""), "summary_hint": state.get("summary", "")},
        max_output_tokens=900,
    )
    notes = _append_notes(list(state.get("autonomy_notes", [])), retry_notes)
    confidence = min(0.96, 0.5 + (0.2 if llm_response else 0.0) + (0.07 if state.get("context_chunks") else 0.0))

    if not llm_response:
        return {**heuristic, "autonomy_notes": notes, "confidence": max(float(state.get("confidence", 0.0)), confidence)}

    summary  = _first_words(str(llm_response.get("summary", heuristic["summary"])), 150)
    methods  = _sanitize_str_list(llm_response.get("methods",  []), max_items=5) or heuristic["methods"]
    datasets = _sanitize_str_list(llm_response.get("datasets", []), max_items=5) or heuristic["datasets"]
    return {"summary": summary, "methods": methods, "datasets": datasets,
            "autonomy_notes": notes, "confidence": max(float(state.get("confidence", 0.0)), confidence)}


def concept_node(state: SharedState) -> SharedState:
    heuristic_concepts = _heuristic_concepts(state)
    llm_response, retry_notes = _autonomous_agent_call(
        state,
        system_prompt=CONCEPT_SYSTEM_PROMPT,
        payload={"title": state.get("title", ""), "summary": state.get("summary", ""),
                 "methods": state.get("methods", []), "concept_hints": state.get("concepts", [])},
        max_output_tokens=512,
    )
    notes = _append_notes(list(state.get("autonomy_notes", [])), retry_notes)
    confidence = min(0.96, 0.52 + (0.18 if llm_response else 0.0))
    llm_concepts = _sanitize_str_list((llm_response or {}).get("concepts", []), max_items=6)
    return {"concepts": llm_concepts or heuristic_concepts[:6], "autonomy_notes": notes,
            "confidence": max(float(state.get("confidence", 0.0)), confidence)}


def graph_update_node(state: SharedState) -> SharedState:
    settings = get_settings()
    if settings.require_graph_review:
        review = graph_review_store.create_review({
            "user_id": state.get("user_id", "local-user"), "paper_url": state.get("paper_url", ""),
            "title": state.get("title", ""), "concepts": list(state.get("concepts", [])),
            "methods": list(state.get("methods", [])), "datasets": list(state.get("datasets", [])),
        })
        return {"graph_updated": False, "review_required": True, "review_id": review.review_id,
                "autonomy_notes": _append_notes(list(state.get("autonomy_notes", [])),
                                                [f"Graph update queued for review: {review.review_id}"])}

    result = update_graph(
        user_id=state.get("user_id", "local-user"), paper_url=state.get("paper_url", ""),
        title=state.get("title", ""), concepts=state.get("concepts", []),
        methods=state.get("methods", []), datasets=state.get("datasets", []),
    )
    graph_updated = bool(result.get("graph_updated", False))
    return {"graph_updated": graph_updated, "review_required": False, "review_id": "",
            "autonomy_notes": _append_notes(list(state.get("autonomy_notes", [])),
                                            [] if graph_updated else ["Graph update failed — retrieval-first reasoning continues."])}


def related_paper_node(state: SharedState) -> SharedState:
    settings = get_settings()
    planner = state.get("planner", {})
    if not bool(planner.get("needs_related_papers", True)):
        return {"related_papers": [],
                "autonomy_notes": _append_notes(list(state.get("autonomy_notes", [])), ["Planner skipped related-paper retrieval."])}

    target_k = _clamp_int(planner.get("related_paper_k", 5), 3, 8, 5)
    retrieval_queries = _sanitize_str_list(state.get("retrieval_queries", []),
                                           max_items=max(1, settings.retrieval_query_expansions))
    if not retrieval_queries:
        fallback = state.get("summary") or state.get("title") or state.get("abstract", "")
        retrieval_queries = [str(fallback)] if fallback else []

    llm_key = _get_llm_key(state)
    aggregated: dict[str, dict[str, Any]] = {}
    for query in retrieval_queries:
        papers = vector_search(
            query=query, top_k=target_k,
            exclude_urls=[state.get("paper_url", "")],
            user_id=state.get("user_id", "local-user"),
            openai_api_key=llm_key,
        )
        for paper in papers:
            url = str(paper.get("url", "")).strip()
            if not url:
                continue
            score = float(paper.get("score", 0.0) or 0.0)
            existing = aggregated.get(url)
            if existing is None or score > float(existing.get("score", 0.0) or 0.0):
                aggregated[url] = dict(paper)

    ranked = sorted(aggregated.values(), key=lambda p: float(p.get("score", 0.0) or 0.0), reverse=True)[:target_k]
    return {"related_papers": ranked,
            "autonomy_notes": _append_notes(list(state.get("autonomy_notes", [])),
                                            [f"Related-paper search: {len(ranked)} results across {len(retrieval_queries)} query variants."])}


def research_gap_node(state: SharedState) -> SharedState:
    planner = state.get("planner", {})
    if not bool(planner.get("needs_gap_analysis", True)):
        return {"graph_connections": [], "research_gaps": [],
                "autonomy_notes": _append_notes(list(state.get("autonomy_notes", [])), ["Planner skipped gap analysis."])}

    concepts = state.get("concepts", [])
    graph_info = query_graph(concepts=concepts, user_id=state.get("user_id", "local-user"), limit=5, interest_limit=5)
    graph_connections    = graph_info.get("graph_connections", [])
    user_interest_topics = graph_info.get("user_interest_topics", [])
    related_papers_summary = [
        {"title": p.get("title", "Untitled"), "score": round(float(p.get("score", 0.0) or 0.0), 3)}
        for p in state.get("related_papers", [])[:8]
    ]
    heuristic_gaps = _heuristic_gaps(concepts, state.get("related_papers", []), graph_connections)
    llm_response, retry_notes = _autonomous_agent_call(
        state, system_prompt=GAP_SYSTEM_PROMPT,
        payload={"concepts": concepts, "related_papers": related_papers_summary,
                 "graph_connections": graph_connections, "paper_title": state.get("title", ""),
                 "paper_summary": state.get("summary", "")},
        max_output_tokens=768,
    )
    notes = _append_notes(list(state.get("autonomy_notes", [])), retry_notes)
    confidence = min(0.97, 0.52 + (0.18 if llm_response else 0.0) + (0.07 if graph_connections else 0.0))
    final_gaps = _sanitize_str_list((llm_response or {}).get("research_gaps", []), max_items=3) or heuristic_gaps
    return {"graph_connections": graph_connections, "user_interest_topics": user_interest_topics[:5],
            "research_gaps": final_gaps[:3], "autonomy_notes": notes,
            "confidence": max(float(state.get("confidence", 0.0)), confidence)}


def idea_node(state: SharedState) -> SharedState:
    concepts = state.get("concepts", [])
    gaps     = state.get("research_gaps", [])
    user_interest_topics = _sanitize_str_list(state.get("user_interest_topics", []), max_items=5) or concepts

    heuristic_ideas = _heuristic_ideas(concepts, gaps)
    llm_response, retry_notes = _autonomous_agent_call(
        state, system_prompt=IDEA_SYSTEM_PROMPT,
        payload={"concepts": concepts, "research_gaps": gaps,
                 "user_interest_topics": user_interest_topics,
                 "paper_title": state.get("title", ""), "methods": state.get("methods", [])},
        max_output_tokens=900,
    )
    notes     = _append_notes(list(state.get("autonomy_notes", [])), retry_notes)
    confidence = min(0.98, 0.56 + (0.17 if llm_response else 0.0) + (0.08 if gaps else 0.0))
    llm_ideas  = _sanitize_str_list((llm_response or {}).get("ideas", []), max_items=3)
    return {"ideas": llm_ideas or heuristic_ideas, "autonomy_notes": notes,
            "confidence": max(float(state.get("confidence", 0.0)), confidence)}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_graph_update(state: SharedState) -> str:
    return "related" if bool(state.get("planner", {}).get("needs_related_papers", True)) else "gap"


def _route_after_gap(state: SharedState) -> str:
    return "idea" if bool(state.get("planner", {}).get("needs_idea_generation", True)) else "end"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_workflow():
    graph = StateGraph(SharedState)
    graph.add_node("planner",      planner_node)
    graph.add_node("parser",       parser_node)
    graph.add_node("concept",      concept_node)
    graph.add_node("graph_update", graph_update_node)
    graph.add_node("related",      related_paper_node)
    graph.add_node("gap",          research_gap_node)
    graph.add_node("idea",         idea_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "parser")
    graph.add_edge("parser",  "concept")
    graph.add_edge("concept", "graph_update")
    graph.add_conditional_edges("graph_update", _route_after_graph_update, {"related": "related", "gap": "gap"})
    graph.add_edge("related", "gap")
    graph.add_conditional_edges("gap", _route_after_gap, {"idea": "idea", "end": END})
    graph.add_edge("idea", END)
    return graph.compile()


def run_agent_workflow(initial_state: dict) -> SharedState:
    workflow = build_workflow()
    state    = with_defaults(initial_state)
    return workflow.invoke(state)
