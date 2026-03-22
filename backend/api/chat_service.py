from __future__ import annotations

import re
from typing import Any

from backend.agents.llm_runtime import call_structured_agent, is_llm_available
from backend.agents.prompts import CHAT_CRITIC_SYSTEM_PROMPT, CHAT_PLANNER_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from backend.api.models import (
    AnalyzePaperRequest, ChatAgentStep, ChatCitation,
    ChatEvidenceSnippet, ChatRequest, ChatResponse,
)
from backend.api.pipeline_service import ingest_paper_for_retrieval
from backend.tools.embedding_tools import has_embedding_provider
from backend.tools.metadata_tools import resolve_paper_metadata
from backend.tools.graph_tools import query_graph
from backend.tools.vector_tools import retrieve_context_chunks

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{4,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_LOW_SIGNAL = ("provided proper attribution", "all rights reserved", "conference on", "arxiv:", "@")
_LIMIT_CUES = ("however", "although", "but", "difficult", "limited", "limitation", "cost", "trade-off")


def _resolve_llm_kwargs(openai_api_key, llm_api_key=None, llm_provider=None):
    key = str(llm_api_key or openai_api_key or "").strip() or None
    return {"openai_api_key": key, "llm_api_key": key, "llm_provider": llm_provider}


def _sanitize_str_list(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned, seen = [], set()
    for item in value:
        text = str(item).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _history_tail(history: list[Any], turns: int = 3) -> list[dict[str, str]]:
    if not history:
        return []
    formatted = []
    for item in history[-turns:]:
        role = str(getattr(item, "role", "user")).strip().lower()
        content = str(getattr(item, "content", "")).strip()
        if role in {"user", "assistant"} and content:
            formatted.append({"role": role, "content": content[:1500]})
    return formatted


def _question_intent(question: str) -> dict[str, bool]:
    q = question.lower()
    return {
        "overview":    any(t in q for t in ["what is", "what does", "main contribution", "about"]),
        "method":      any(t in q for t in ["how", "method", "approach", "architecture"]),
        "results":     any(t in q for t in ["result", "performance", "metric", "accuracy", "score"]),
        "limitations": any(t in q for t in ["limitation", "weakness", "tradeoff", "drawback"]),
        "comparison":  any(t in q for t in ["compare", "difference", "versus", " vs "]),
    }


def _is_low_signal(text: str) -> bool:
    n = " ".join(str(text or "").split()).lower()
    return not n or any(p in n for p in _LOW_SIGNAL)


def _best_excerpt(question: str, text: str, max_chars: int = 300) -> str:
    norm = " ".join(str(text or "").split()).strip()
    if not norm:
        return ""
    q_tokens = _token_set(question)
    sentences = [s.strip() for s in _SENTENCE_RE.split(norm) if s.strip()]
    best, best_score = "", -1.0
    for s in sentences[:10]:
        if _is_low_signal(s):
            continue
        score = len(q_tokens & _token_set(s)) / max(1, len(q_tokens))
        if score > best_score:
            best_score, best = score, s
    candidate = best or norm
    if len(candidate) <= max_chars:
        return candidate
    return candidate[:max_chars].rsplit(" ", 1)[0] + "..."


def _sentence_score(question: str, sentence: str, section: str, chunk_score: float) -> float:
    overlap = len(_token_set(question) & _token_set(sentence)) / max(1, len(_token_set(question)))
    low = sentence.lower()
    intent = _question_intent(question)
    bonus = 0.0
    sec = section.strip().lower()
    if intent["overview"]    and sec in {"abstract", "introduction"}: bonus += 0.08
    if intent["method"]      and sec in {"method", "methods", "body"}: bonus += 0.08
    if intent["results"]     and sec == "results": bonus += 0.12
    if intent["limitations"] and any(c in low for c in _LIMIT_CUES): bonus += 0.12
    if intent["comparison"]  and any(t in low for t in ["whereas", "while", "than", "compared"]): bonus += 0.08
    if "we propose" in low or "the paper proposes" in low: bonus += 0.06
    return overlap * 0.62 + max(0.0, float(chunk_score)) * 0.28 + bonus


def _rewrite_sentence(sentence: str) -> str:
    s = re.sub(r"\[[^\]]+\]", "", " ".join(str(sentence or "").split())).strip()
    if not s:
        return ""
    for pat, rep in (
        (r"^We propose\s+", "The paper proposes "),
        (r"^We show\s+", "The paper shows "),
        (r"^We find\s+", "The paper finds "),
        (r"^Our model\s+", "The model "),
    ):
        candidate = re.sub(pat, rep, s, flags=re.IGNORECASE)
        if candidate != s:
            s = candidate
            break
    s = s.strip(" .")
    return (s[0].upper() + s[1:] + ".") if s else ""


def _support_candidates(question: str, chunks: list[dict], max_items: int = 8) -> list[dict]:
    candidates, seen = [], set()
    for chunk in chunks[:8]:
        section = str(chunk.get("section", "Body")).strip() or "Body"
        text = " ".join(str(chunk.get("text", "")).split()).strip()
        cscore = float(chunk.get("score", 0.0) or 0.0)
        if not text:
            continue
        for s in [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()][:8]:
            if _is_low_signal(s) or s.lower() in seen:
                continue
            seen.add(s.lower())
            score = _sentence_score(question, s, section, cscore)
            if score > 0:
                candidates.append({"sentence": s, "section": section, "score": score})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    non_title = [c for c in candidates if c["section"].lower() != "title"]
    if non_title:
        candidates = non_title + [c for c in candidates if c["section"].lower() == "title"]
    return candidates[:max_items]


def _fallback_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No indexed content found. Try re-analysing the paper first."
    candidates = _support_candidates(question, chunks)
    if not candidates:
        return "Found context but couldn't extract a clean answer. Try asking more specifically."
    primary = _rewrite_sentence(candidates[0]["sentence"])
    if not primary:
        primary = candidates[0]["sentence"][:250]
    secondary = ""
    for item in candidates[1:]:
        if item["section"].lower() == "title":
            continue
        c = _rewrite_sentence(item["sentence"])
        if c and c.lower() != primary.lower():
            secondary = c
            break
    intent = _question_intent(question)
    answer = "The paper's key limitation: " + primary if intent["limitations"] else primary
    if secondary and len(answer) < 300:
        answer += " It also notes: " + secondary[0].lower() + secondary[1:]
    return answer[:400]


def _truncate(text: str, max_chars: int = 300) -> str:
    n = " ".join(str(text or "").split()).strip()
    if len(n) <= max_chars:
        return n
    return (n[:max_chars].rsplit(" ", 1)[0] or n[:max_chars]) + "..."


def _paper_candidates(paper_url, paper_title=None, paper_abstract=None) -> list[str]:
    url = str(paper_url or "").strip()
    if not url:
        return []
    if url.lower().endswith(".pdf"):
        return [url]
    resolved = resolve_paper_metadata(paper_url=url, title=str(paper_title or ""), abstract=str(paper_abstract or ""))
    seen, candidates = set(), []
    for item in (url, str(resolved.get("pdf_url", "")), str(resolved.get("canonical_url", ""))):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            candidates.append(item)
    return candidates


def _ensure_indexed(payload: ChatRequest, *, llm_kwargs: dict) -> None:
    url = str(payload.paper_url or "").strip()
    if not url or not payload.ensure_current_paper_indexed:
        return
    ingest_paper_for_retrieval(
        AnalyzePaperRequest(
            paper_url=url,
            title=str(payload.paper_title or "").strip() or "Current Paper",
            abstract=str(payload.paper_abstract or "").strip()[:30000],
            user_id=payload.user_id,
        ),
        openai_api_key=llm_kwargs.get("openai_api_key"),
    )


def _citations(chunks: list[dict], max_items: int = 5) -> list[ChatCitation]:
    result, seen = [], set()
    for chunk in chunks:
        url = str(chunk.get("paper_url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        result.append(ChatCitation(
            title=str(chunk.get("title", "Untitled")), url=url,
            section=str(chunk.get("section", "Body")),
            score=float(chunk.get("score", 0.0) or 0.0),
        ))
        if len(result) >= max_items:
            break
    return result


def _evidence_snippets(chunks: list[dict], question: str, max_items: int = 3) -> list[ChatEvidenceSnippet]:
    result, seen = [], set()
    for chunk in chunks:
        snippet = _best_excerpt(question, str(chunk.get("text", "")))
        if not snippet:
            continue
        url = str(chunk.get("paper_url", "")).strip()
        section = str(chunk.get("section", "Body")).strip() or "Body"
        key = f"{url}|{section}|{snippet[:100].lower()}"
        if key in seen:
            continue
        seen.add(key)
        result.append(ChatEvidenceSnippet(
            title=str(chunk.get("title", "Untitled")), url=url, section=section,
            snippet=snippet, score=float(chunk.get("score", 0.0) or 0.0),
        ))
        if len(result) >= max_items:
            break
    return result


def _grounding_score(answer: str, snippets: list[ChatEvidenceSnippet]) -> float:
    a_tokens = _token_set(answer)
    if not a_tokens or not snippets:
        return 0.0
    ev_tokens: set[str] = set()
    for s in snippets:
        ev_tokens.update(_token_set(s.snippet))
    return min(1.0, len(a_tokens & ev_tokens) / max(1, min(len(a_tokens), 24)))


def _support_score(*, answer, chunks, citations, snippets, used_llm, graph_connections) -> float:
    if not chunks:
        return 0.0
    scores = [float(c.get("score", 0.0) or 0.0) for c in chunks[:4] if isinstance(c, dict)]
    avg = sum(scores) / len(scores) if scores else 0.0
    score = 0.2 + min(0.35, avg * 0.45) + min(0.25, len(citations) * 0.07) + min(0.08, len(snippets) * 0.025)
    if used_llm: score += 0.1
    if graph_connections: score += 0.08
    score += min(0.14, _grounding_score(answer, snippets) * 0.2)
    return round(min(1.0, score), 3)


def _trace(agent: str, *, detail: str, used_llm=False, observations=None, status="completed") -> ChatAgentStep:
    return ChatAgentStep(agent=agent, status=status, detail=detail, used_llm=used_llm, observations=observations or [])


def _clamp_k(value: Any, default: int) -> int:
    try:
        return min(max(int(value), 2), 8)
    except Exception:
        return default


def _heuristic_plan(payload: ChatRequest, paper_candidates: list[str]) -> dict:
    q = str(payload.question or "").lower()
    comparative = any(t in q for t in ["compare", "difference", "versus", " vs "])
    queries = [payload.question]
    tokens = [t for t in _TOKEN_RE.findall(payload.question) if len(t) >= 4]
    if len(tokens) >= 2:
        queries.append(" ".join(tokens[:6]))
    return {
        "retrieval_queries": queries[:3],
        "needs_graph": not payload.current_paper_only,
        "top_k": max(payload.top_k, 5 if comparative else 4),
        "answer_style": "comparative grounded answer" if comparative else "direct grounded answer",
        "reason": "heuristic plan",
    }


def _chat_plan(payload: ChatRequest, paper_candidates: list[str], *, llm_kwargs: dict):
    heuristic = _heuristic_plan(payload, paper_candidates)
    llm_resp = call_structured_agent(
        system_prompt=CHAT_PLANNER_SYSTEM_PROMPT,
        payload={"question": payload.question, "recent_history": _history_tail(payload.history),
                 "current_paper_only": payload.current_paper_only,
                 "heuristic_queries": heuristic["retrieval_queries"]},
        max_output_tokens=300,
        **llm_kwargs,
    )
    plan = dict(heuristic)
    used_llm = isinstance(llm_resp, dict)
    if used_llm:
        llm_q = _sanitize_str_list(llm_resp.get("retrieval_queries", []), 3)
        if llm_q:
            plan["retrieval_queries"] = llm_q
        plan["needs_graph"] = bool(llm_resp.get("needs_graph", plan["needs_graph"]))
        plan["top_k"] = _clamp_k(llm_resp.get("top_k", plan["top_k"]), int(plan["top_k"]))
        if llm_resp.get("answer_style"):
            plan["answer_style"] = str(llm_resp["answer_style"])
    return plan, _trace("planner", detail=f"{plan['answer_style']} | top_k={plan['top_k']}", used_llm=used_llm), []


def _aggregate_chunks(chunk_lists: list[list[dict]], limit: int) -> list[dict]:
    agg: dict[str, dict] = {}
    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            text = str(chunk.get("text", "")).strip()
            url = str(chunk.get("paper_url", "")).strip()
            section = str(chunk.get("section", "Body")).strip() or "Body"
            if not text:
                continue
            key = f"{url}|{section}|{text[:140]}"
            if key not in agg or float(chunk.get("score", 0.0)) > float(agg[key].get("score", 0.0)):
                agg[key] = dict(chunk)
    return sorted(agg.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:max(1, limit)]


def _retrieve(payload: ChatRequest, plan: dict, paper_candidates: list[str], *, llm_kwargs: dict):
    queries = _sanitize_str_list(plan.get("retrieval_queries", []), 3) or [payload.question]
    k = _clamp_k(plan.get("top_k", payload.top_k), max(payload.top_k, 4))
    scoped_runs = []
    if paper_candidates:
        for q in queries:
            scoped_runs.append(retrieve_context_chunks(
                query=q, top_k=k, user_id=payload.user_id,
                include_urls=paper_candidates, exclude_urls=None,
                openai_api_key=llm_kwargs.get("openai_api_key"),
            ))
    scoped = _aggregate_chunks(scoped_runs, k)
    if scoped and (payload.current_paper_only or len(scoped) >= min(2, k)):
        return scoped, _trace("retrieval", detail="Scoped to current paper.", observations=[f"hits={len(scoped)}"]), []
    if payload.current_paper_only:
        return [], _trace("retrieval", detail="No chunks found.", status="warning"), []
    broad_runs = []
    for q in queries:
        broad_runs.append(retrieve_context_chunks(
            query=q, top_k=k, user_id=payload.user_id,
            openai_api_key=llm_kwargs.get("openai_api_key"),
        ))
    broad = _aggregate_chunks(broad_runs, k)
    return broad, _trace("retrieval", detail="Broad retrieval.", observations=[f"hits={len(broad)}"]), []


def _graph_context(payload: ChatRequest, plan: dict, chunks: list[dict]):
    if not plan.get("needs_graph", True):
        return {"graph_connections": [], "user_interest_topics": []}, _trace("graph", detail="Skipped."), []
    concepts = _sanitize_str_list(
        [t for t in _TOKEN_RE.findall(payload.question) if len(t) >= 4][:5] +
        [t for chunk in chunks[:3] for t in _TOKEN_RE.findall(chunk.get("text", "")) if len(t) >= 5][:5],
        max_items=6
    )
    info = query_graph(concepts=concepts, user_id=payload.user_id, limit=5, interest_limit=5)
    return (
        {"graph_connections": list(info.get("graph_connections", [])),
         "user_interest_topics": list(info.get("user_interest_topics", []))},
        _trace("graph", detail=f"connections={len(info.get('graph_connections', []))}"),
        [],
    )


def _answer_step(payload: ChatRequest, plan: dict, chunks: list[dict], graph_info: dict, *, llm_kwargs: dict):
    llm_resp = call_structured_agent(
        system_prompt=CHAT_SYSTEM_PROMPT,
        payload={
            "question": payload.question,
            "recent_history": _history_tail(payload.history),
            "retrieval_context": [{"title": c.get("title"), "section": c.get("section"), "text": c.get("text", "")[:600]} for c in chunks[:6]],
            "graph_connections": graph_info.get("graph_connections", []),
            "user_interest_topics": graph_info.get("user_interest_topics", []),
            "answer_style": str(plan.get("answer_style", "direct grounded answer")),
        },
        max_output_tokens=800,
        **llm_kwargs,
    )
    used_llm = isinstance(llm_resp, dict)
    if used_llm:
        answer = str(llm_resp.get("answer", "")).strip() or _fallback_answer(payload.question, chunks)
        followups = _sanitize_str_list(llm_resp.get("follow_up_questions", []), 3)
        detail = "LLM answer."
    else:
        answer = _fallback_answer(payload.question, chunks)
        followups = []
        detail = "Extractive fallback."
    return answer, followups, _trace("answer", detail=detail, used_llm=used_llm), [], used_llm


def _grounding_step(payload: ChatRequest, answer: str, followups: list[str], chunks: list[dict],
                    graph_info: dict, *, llm_kwargs: dict, used_answer_llm: bool):
    cits = _citations(chunks)
    snips = _evidence_snippets(chunks, payload.question)
    support = _support_score(answer=answer, chunks=chunks, citations=cits, snippets=snips,
                              used_llm=used_answer_llm, graph_connections=graph_info.get("graph_connections", []))
    notes, revised_answer, revised_followups = [], answer, list(followups)
    critic_used_llm = False

    if chunks and is_llm_available(**llm_kwargs):
        critic_resp = call_structured_agent(
            system_prompt=CHAT_CRITIC_SYSTEM_PROMPT,
            payload={
                "question": payload.question, "answer": answer, "support_score": support,
                "evidence_snippets": [{"section": s.section, "snippet": s.snippet, "score": s.score} for s in snips],
                "follow_up_questions": followups,
            },
            max_output_tokens=500,
            **llm_kwargs,
        )
        critic_used_llm = isinstance(critic_resp, dict)
        if critic_used_llm:
            issues = _sanitize_str_list(critic_resp.get("issues", []), 4)
            grounded = bool(critic_resp.get("grounded", support >= 0.45))
            candidate = str(critic_resp.get("revised_answer", "")).strip()
            if candidate and not grounded:
                revised_answer = candidate
            critic_followups = _sanitize_str_list(critic_resp.get("follow_up_questions", []), 3)
            if critic_followups:
                revised_followups = critic_followups
            if issues:
                notes.extend(issues)

    # Only fall back to extractive if support is very low AND answer looks poor
    if support < 0.25 and not used_answer_llm:
        revised_answer = _fallback_answer(payload.question, chunks)

    return (
        revised_answer, revised_followups, cits, snips, support,
        _trace("grounding", detail=f"support={support:.2f}", used_llm=critic_used_llm),
        notes,
    )


def run_chat_query(
    payload: ChatRequest,
    openai_api_key: str | None = None,
    llm_api_key: str | None = None,
    llm_provider: str | None = None,
) -> ChatResponse:
    llm_kwargs = _resolve_llm_kwargs(openai_api_key, llm_api_key, llm_provider)
    _ensure_indexed(payload, llm_kwargs=llm_kwargs)
    retrieval_mode = "semantic" if has_embedding_provider(llm_kwargs.get("openai_api_key")) else "lexical"
    paper_candidates_list = _paper_candidates(payload.paper_url, payload.paper_title, payload.paper_abstract)

    agent_trace, autonomy_notes = [], [f"retrieval_mode={retrieval_mode}", f"llm={is_llm_available(**llm_kwargs)}"]

    plan, planner_step, _ = _chat_plan(payload, paper_candidates_list, llm_kwargs=llm_kwargs)
    agent_trace.append(planner_step)

    chunks, retrieval_step, retrieval_notes = _retrieve(payload, plan, paper_candidates_list, llm_kwargs=llm_kwargs)
    agent_trace.append(retrieval_step)
    autonomy_notes.extend(retrieval_notes)

    if payload.current_paper_only and not chunks:
        return ChatResponse(
            answer="Paper not indexed yet. Click ↺ Analyse and wait for it to complete, then ask again.",
            citations=[], evidence_snippets=[], follow_up_questions=[],
            used_context_chunks=0, support_score=0.0, retrieval_mode=retrieval_mode,
            graph_connections=[], user_interest_topics=[], autonomy_notes=autonomy_notes, agent_trace=agent_trace,
        )

    graph_info, graph_step, _ = _graph_context(payload, plan, chunks)
    agent_trace.append(graph_step)

    answer, followups, answer_step, _, used_answer_llm = _answer_step(payload, plan, chunks, graph_info, llm_kwargs=llm_kwargs)
    agent_trace.append(answer_step)

    answer, followups, cits, snips, support, grounding_step, grounding_notes = _grounding_step(
        payload, answer, followups, chunks, graph_info, llm_kwargs=llm_kwargs, used_answer_llm=used_answer_llm
    )
    agent_trace.append(grounding_step)
    autonomy_notes.extend(grounding_notes)

    return ChatResponse(
        answer=answer, citations=cits, evidence_snippets=snips, follow_up_questions=followups,
        used_context_chunks=len(chunks), support_score=support, retrieval_mode=retrieval_mode,
        graph_connections=list(graph_info.get("graph_connections", []))[:5],
        user_interest_topics=list(graph_info.get("user_interest_topics", []))[:5],
        autonomy_notes=_sanitize_str_list(autonomy_notes, 12), agent_trace=agent_trace,
    )
