# Research Copilot

A Chrome extension + backend system that automatically analyzes research papers as you browse. Detects papers on arXiv, Semantic Scholar, OpenReview, and direct PDF links — then provides AI-powered summaries, key concepts, research gaps, related papers, and a conversational RAG chat grounded in the paper's full text.

## Architecture

```
Chrome Extension (Manifest V3, side panel)
    ↕ REST API
FastAPI Backend
    ├── LangGraph multi-agent pipeline (planner → parser → concept → gap → idea)
    ├── FAISS vector store (semantic search / RAG)
    ├── Neo4j knowledge graph (paper relationships, citation edges)
    ├── Semantic Scholar API (citation fetching + async BFS crawl)
    └── OpenAlex API (related papers preview)
```

## Quick Start

### 1. Configure environment

```bash
cp docker/.env.example docker/.env
cp backend/.env.example backend/.env
# Edit both files — at minimum set a HuggingFace API key
```

### 2. Start services

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

This starts:
- **Backend** at `http://localhost:8000`
- **Neo4j** at `http://localhost:7474` (bolt: `localhost:7687`)

### 3. Load the Chrome extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` directory
4. Navigate to a paper (e.g. `https://arxiv.org/abs/1706.03762`)
5. Click the extension icon to open the side panel

## Features

- **Auto-detection**: Recognizes papers on arXiv, Semantic Scholar, OpenReview, and any `.pdf` URL
- **Two-stage analysis**: Fast heuristic results (~2s) followed by full LLM-powered analysis (~10-20s)
- **Mind map**: Interactive D3.js visualization showing research domains → concepts → related paper links
- **RAG chat**: Ask questions about the paper — answers are grounded in the full text with citations
- **Citation graph**: Fetches citations from Semantic Scholar, indexes them into FAISS + Neo4j
- **Multi-LLM support**: HuggingFace (free), OpenAI, Anthropic, Google Gemini

## LLM Providers

Default: **HuggingFace** with `Qwen/Qwen2.5-7B-Instruct` (free tier, ~1000 req/day).

Set in `backend/.env`:

| Provider | `LLM_PROVIDER` | Key variable |
|---|---|---|
| HuggingFace | `huggingface` | `HUGGINGFACE_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/resolve-paper` | GET | Resolve paper metadata from URL |
| `/analyze-paper` | POST | Submit paper for analysis |
| `/analysis-status/{job_id}` | GET | Poll analysis progress |
| `/chat` | POST | RAG-grounded Q&A about a paper |
| `/index-paper` | POST | Pre-index a paper into FAISS |
| `/citation-papers` | GET | Fetch citations from Semantic Scholar |
| `/related-papers-preview` | GET | Quick related papers lookup |
| `/index-stats` | GET | Vector index statistics |

## Development

### Build the extension UI

```bash
cd extension-ui
npm install
npm run export    # builds + copies to extension/
```

### Run backend locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

### Run tests

```bash
pytest
```

## Project Structure

```
backend/
  agents/       # LLM agents: prompts, workflow, runtime
  api/          # FastAPI endpoints, services
  core/         # Settings / config
  db/           # FAISS + Neo4j clients
  pipeline/     # Text chunking, token budgets
  tools/        # PDF, metadata, citations, embeddings, graph, vector tools
docker/         # Docker Compose + env config
extension/      # Chrome extension (Manifest V3)
extension-ui/   # React + Vite + D3.js (side panel UI source)
tests/          # Unit + integration tests
```

## Supported URL Patterns

- `https://arxiv.org/abs/*` and `https://arxiv.org/pdf/*`
- `https://www.semanticscholar.org/paper/*`
- `https://openreview.net/forum*`, `/pdf*`, `/attachment*`
- Any direct PDF URL (`https://.../*.pdf`)
