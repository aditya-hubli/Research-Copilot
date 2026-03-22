const DEFAULT_API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 2500;
const MAX_POLL_ATTEMPTS = 60;

const SUPPORTED_PATTERNS = [
  /^https:\/\/arxiv\.org\/(pdf|abs)\//i,
  /^https:\/\/www\.semanticscholar\.org\/paper\//i,
  /^https:\/\/openreview\.net\/(forum|pdf|attachment)(?:$|[/?#])/i,
  /^https?:\/\/[^\s]+\.pdf(?:[?#].*)?$/i,
];

const tabAnalysis      = new Map();
const activePollTimers = new Map();
const lastAnalyzedUrl  = new Map();

// ── Helpers ───────────────────────────────────────────────────────────────────

function isSupportedUrl(url) {
  return url ? SUPPORTED_PATTERNS.some(p => p.test(url)) : false;
}

function titleFromUrl(url) {
  try {
    const p = new URL(url);
    const seg = (p.pathname || "").split("/").filter(Boolean).pop() || p.hostname;
    const clean = decodeURIComponent(seg.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ")).trim();
    return clean ? clean.slice(0, 120) : `Paper from ${p.hostname}`;
  } catch { return "Untitled Paper"; }
}

function defaultState(url = "") {
  return {
    pageUrl: url, resolved_url: "", paper_title: "", abstract: "",
    source: "", provider: "", status: "idle", job_id: "", from_cache: false,
    confidence: 0, summary: "", key_concepts: [], methods: [], datasets: [],
    related_papers: [], related_papers_loading: false,
    research_connections: [], user_interest_topics: [], research_gaps: [], ideas: [],
    current_paper_indexing: false, current_paper_indexed: false, current_paper_chunks: 0,
    chat_status: "idle", chat_question: "", chat_answer: "", chat_error: "",
    chat_citations: [], chat_evidence_snippets: [], chat_follow_up_questions: [],
    chat_support_score: 0, chat_used_context_chunks: 0, error: "",
  };
}

function isGoodTitle(t) {
  return t && typeof t === "string" && t.trim().length > 4 && t.trim() !== "Untitled";
}

function mergeRelatedPapers(existing, incoming) {
  const byUrl = new Map();
  for (const p of (existing || [])) {
    if (p.url) byUrl.set(p.url, p);
  }
  for (const p of (incoming || [])) {
    if (!p.url) continue;
    const prev = byUrl.get(p.url);
    // Keep whichever has the better title; prefer incoming score if it's higher
    if (!prev || (!isGoodTitle(prev.title) && isGoodTitle(p.title))) {
      byUrl.set(p.url, p);
    } else if (prev && isGoodTitle(prev.title) && isGoodTitle(p.title)) {
      // Both good — keep higher score
      byUrl.set(p.url, (p.score || 0) > (prev.score || 0) ? p : prev);
    }
  }
  return Array.from(byUrl.values())
    .filter(p => isGoodTitle(p.title))
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, 8);
}

function updateTabState(tabId, patch) {
  const current = tabAnalysis.get(tabId) || defaultState();
  const merged  = { ...current, ...patch };
  // Smart-merge related_papers: never overwrite good titles with "Untitled" ones
  if (patch.related_papers !== undefined) {
    merged.related_papers = mergeRelatedPapers(current.related_papers, patch.related_papers);
  }
  tabAnalysis.set(tabId, merged);
  chrome.runtime.sendMessage({ type: "ANALYSIS_UPDATE", tabId, payload: merged }).catch(() => {});
}

async function getRuntimeConfig() {
  const s = await chrome.storage.local.get([
    "backendApiBase", "backendApiKey",
    "llmApiKey", "llmProvider",
    // Legacy key names — still supported
    "openaiApiKey",
    "userId",
  ]);
  return {
    apiBase:      s.backendApiBase  || DEFAULT_API_BASE,
    apiKey:       s.backendApiKey   || "",
    // Prefer new unified key; fall back to legacy openaiApiKey
    llmApiKey:    s.llmApiKey || s.openaiApiKey || "",
    llmProvider:  s.llmProvider || "",
    userId:       s.userId          || "local-user",
  };
}

function buildHeaders(cfg) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.apiKey)      headers["X-API-Key"]        = cfg.apiKey;
  if (cfg.llmApiKey)   headers["X-LLM-Api-Key"]    = cfg.llmApiKey;
  if (cfg.llmProvider) headers["X-LLM-Provider"]   = cfg.llmProvider;
  // Legacy header for backwards compat with older backend versions
  if (cfg.llmApiKey)   headers["X-OpenAI-Api-Key"] = cfg.llmApiKey;
  return headers;
}

async function fetchJson(url, options = {}) {
  const { cfg, ...rest } = options;
  const headers = cfg ? buildHeaders(cfg) : (rest.headers || {});
  const res = await fetch(url, { ...rest, headers });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

// ── Detection ─────────────────────────────────────────────────────────────────

function enableSidePanel(tabId) {
  chrome.sidePanel.setOptions({ tabId, path: "sidebar.html", enabled: true }).catch(() => {});
}

async function maybeTriggerAnalysis(tabId, url) {
  if (!url || !isSupportedUrl(url)) return;
  enableSidePanel(tabId);
  if (lastAnalyzedUrl.get(tabId) === url) return;
  await startAnalysis(tabId, { paper_url: url, title: titleFromUrl(url), abstract: "" });
}

async function startAnalysis(tabId, payload) {
  const cfg = await getRuntimeConfig();
  const url = payload.paper_url;
  lastAnalyzedUrl.set(tabId, url);
  clearPoll(tabId);

  let requestPayload = {
    paper_url: url,
    title:    payload.title || titleFromUrl(url),
    abstract: payload.abstract || "",
    user_id:  cfg.userId,
  };

  updateTabState(tabId, {
    pageUrl: url, paper_title: requestPayload.title,
    status: "running", error: "",
  });

  try {
    // Resolve metadata
    const params = new URLSearchParams({ paper_url: url });
    if (requestPayload.title) params.set("title", requestPayload.title);
    const resolved = await fetchJson(`${cfg.apiBase}/resolve-paper?${params}`, { cfg }).catch(() => null);

    if (resolved) {
      requestPayload = {
        paper_url: resolved.resolved_url || url,
        title:    resolved.title || requestPayload.title,
        abstract: resolved.abstract || "",
        user_id:  cfg.userId,
      };
      updateTabState(tabId, {
        resolved_url: resolved.resolved_url || url,
        paper_title: requestPayload.title,
        source: resolved.source || "",
        abstract: requestPayload.abstract,
      });
    }

    // Index for RAG (fire and forget)
    fetchJson(`${cfg.apiBase}/index-paper`, {
      cfg, method: "POST",
      body: JSON.stringify(requestPayload),
    }).then(d => updateTabState(tabId, {
      current_paper_indexed: true,
      current_paper_chunks:  Number(d.indexed_chunks || 0),
    })).catch(() => {});

    // Fetch citations from Semantic Scholar (fire and forget)
    // These get indexed into FAISS + Neo4j CITES edges on the backend
    const citationParams = new URLSearchParams({
      paper_url: requestPayload.paper_url,
      user_id: cfg.userId,
      limit: "15",
    });
    fetchJson(`${cfg.apiBase}/citation-papers?${citationParams}`, { cfg })
      .then(d => {
        const cited = (d.citations || []).filter(p => p.url && p.title && p.title !== "Untitled");
        if (cited.length > 0) {
          updateTabState(tabId, { related_papers: cited, related_papers_provider: "citations" });
        }
      }).catch(() => {});

    // Also prefetch related papers by title as fallback (fire and forget)
    if (requestPayload.title) {
      const rp = new URLSearchParams({ title: requestPayload.title, paper_url: requestPayload.paper_url, limit: "5" });
      fetchJson(`${cfg.apiBase}/related-papers-preview?${rp}`, { cfg })
        .then(d => {
          const preview = d.related_papers || [];
          if (preview.length > 0) {
            updateTabState(tabId, { related_papers: preview, related_papers_provider: d.provider || "" });
          }
        }).catch(() => {});
    }

    // Submit full analysis
    const data = await fetchJson(`${cfg.apiBase}/analyze-paper`, {
      cfg, method: "POST",
      body: JSON.stringify(requestPayload),
    });

    updateTabState(tabId, { ...data, status: data.status || "partial", error: "" });
    if (data.job_id) schedulePoll(tabId, data.job_id, cfg);

  } catch (err) {
    updateTabState(tabId, { status: "failed", error: String(err) });
  }
}

// ── Polling ───────────────────────────────────────────────────────────────────

function clearPoll(tabId) {
  const t = activePollTimers.get(tabId);
  if (t) { clearTimeout(t); activePollTimers.delete(tabId); }
}

function schedulePoll(tabId, jobId, cfg, attempt = 1) {
  if (attempt > MAX_POLL_ATTEMPTS) {
    updateTabState(tabId, { status: "failed", error: "Analysis timed out." });
    return;
  }
  const t = setTimeout(async () => {
    try {
      const data = await fetchJson(`${cfg.apiBase}/analysis-status/${jobId}`, { cfg });
      updateTabState(tabId, data);
      if (!["complete","failed","canceled"].includes(data.status)) {
        schedulePoll(tabId, jobId, cfg, attempt + 1);
      } else {
        clearPoll(tabId);
      }
    } catch {
      schedulePoll(tabId, jobId, cfg, attempt + 1);
    }
  }, POLL_INTERVAL_MS);
  activePollTimers.set(tabId, t);
}

// ── Chat ──────────────────────────────────────────────────────────────────────

async function handleChat(tabId, question) {
  const cfg   = await getRuntimeConfig();
  const state = tabAnalysis.get(tabId) || defaultState();
  const url   = state.resolved_url || state.pageUrl;

  if (!url) throw new Error("No paper open.");

  updateTabState(tabId, { chat_status: "running", chat_question: question, chat_error: "" });

  const data = await fetchJson(`${cfg.apiBase}/chat`, {
    cfg, method: "POST",
    body: JSON.stringify({
      user_id:                      cfg.userId,
      question,
      paper_url:                    url,
      paper_title:                  state.paper_title || titleFromUrl(url),
      paper_abstract:               state.abstract || "",
      top_k:                        6,
      ensure_current_paper_indexed: true,
      current_paper_only:           true,
    }),
  });

  updateTabState(tabId, {
    chat_status:              "complete",
    chat_answer:              data.answer              || "",
    chat_citations:           data.citations            || [],
    chat_evidence_snippets:   data.evidence_snippets    || [],
    chat_follow_up_questions: data.follow_up_questions  || [],
    chat_support_score:       Number(data.support_score || 0),
    chat_used_context_chunks: Number(data.used_context_chunks || 0),
    chat_error: "",
    current_paper_indexed: true,
  });

  return data;
}

// ── Chrome events ─────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const url = changeInfo.url || tab.url;
  if (!url || !isSupportedUrl(url)) return;
  enableSidePanel(tabId);
  if (changeInfo.url) {
    maybeTriggerAnalysis(tabId, changeInfo.url).catch(() => {});
    return;
  }
  if (changeInfo.status === "complete" && tab.url) {
    maybeTriggerAnalysis(tabId, tab.url).catch(() => {});
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url && isSupportedUrl(tab.url)) {
      enableSidePanel(tabId);
      await maybeTriggerAnalysis(tabId, tab.url);
    }
  } catch {}
});

chrome.tabs.onRemoved.addListener((tabId) => {
  clearPoll(tabId);
  tabAnalysis.delete(tabId);
  lastAnalyzedUrl.delete(tabId);
});

// ── Messages ──────────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg?.type) return;

  if (msg.type === "SIDEBAR_READY") {
    chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
      if (tab?.url && isSupportedUrl(tab.url)) {
        enableSidePanel(tab.id);
        await maybeTriggerAnalysis(tab.id, tab.url).catch(() => {});
      }
      sendResponse({ ok: true });
    });
    return true;
  }

  if (msg.type === "GET_TAB_ANALYSIS") {
    sendResponse({ ok: true, payload: tabAnalysis.get(msg.tabId) || defaultState() });
    return;
  }

  if (msg.type === "TRIGGER_ANALYSIS") {
    chrome.tabs.get(msg.tabId, async (tab) => {
      if (!tab?.url) { sendResponse({ ok: false, error: "No tab URL" }); return; }
      lastAnalyzedUrl.delete(msg.tabId);
      await maybeTriggerAnalysis(msg.tabId, tab.url).catch(() => {});
      sendResponse({ ok: true });
    });
    return true;
  }

  if (msg.type === "PAPER_DETECTED") {
    const tabId = sender.tab?.id;
    if (typeof tabId !== "number") { sendResponse({ ok: false }); return; }
    startAnalysis(tabId, msg.payload)
      .then(() => sendResponse({ ok: true }))
      .catch(e => sendResponse({ ok: false, error: String(e) }));
    return true;
  }

  if (msg.type === "ASK_PAPER_CHAT") {
    handleChat(msg.tabId, String(msg.question || "").trim())
      .then(d => sendResponse({ ok: true, payload: d }))
      .catch(e => {
        updateTabState(msg.tabId, { chat_status: "failed", chat_error: String(e) });
        sendResponse({ ok: false, error: String(e) });
      });
    return true;
  }
});
