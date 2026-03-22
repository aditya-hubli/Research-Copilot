function detectAbstract() {
  const metaSelectors = [
    'meta[name="citation_abstract"]',
    'meta[name="description"]',
    'meta[property="og:description"]',
  ];

  for (const selector of metaSelectors) {
    const element = document.querySelector(selector);
    const value = element?.getAttribute("content")?.trim();
    if (value) {
      return value;
    }
  }

  const arxivAbstract = document.querySelector("blockquote.abstract");
  if (arxivAbstract?.textContent) {
    return arxivAbstract.textContent.replace(/^\s*Abstract:\s*/i, "").trim();
  }

  const fallbackCandidates = [
    ".abstract",
    "section.abstract",
    "article p",
    "main p",
    "p",
  ];

  for (const selector of fallbackCandidates) {
    const nodes = Array.from(document.querySelectorAll(selector));
    const text = nodes
      .map((node) => node.textContent?.trim() || "")
      .find((value) => value.length >= 80);
    if (text) {
      return text.slice(0, 1200);
    }
  }

  return "";
}

function detectTitle() {
  const citationTitle = document.querySelector('meta[name="citation_title"]')?.getAttribute("content");
  if (citationTitle?.trim()) {
    return citationTitle.trim();
  }

  const ogTitle = document.querySelector('meta[property="og:title"]')?.getAttribute("content");
  if (ogTitle?.trim()) {
    return ogTitle.trim();
  }

  return document.title?.trim() || "Untitled Paper";
}

function sendDetection() {
  const payload = {
    paper_url: window.location.href,
    title: detectTitle(),
    abstract: detectAbstract(),
    user_id: "local-user",
  };

  chrome.runtime.sendMessage({
    type: "PAPER_DETECTED",
    payload,
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", sendDetection, { once: true });
} else {
  sendDetection();
}
