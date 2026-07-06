# AI-PaperFlow

AI-PaperFlow is a second-development project based on `Futuresxy/paper-daily`. It turns an automated paper listing workflow into a lightweight AI research system for paper tracking, retrieval, reranking, Chinese QA, and DeepResearch Lite reports.

## Architecture

```text
GitHub Actions / Local Run
  -> OpenAlex Fetch
  -> Field Normalization + Deduplication
  -> web/data/papers.json
  -> RAG Corpus Export
  -> web/data/rag_corpus.json
  -> BM25 + ChromaDB/Qwen Dense Retrieval
  -> Cohere Rerank
  -> Gemini Answer / Research Report
  -> FastAPI + GitHub Pages Demo
```

## Features

- OpenAlex paper tracking for LLM agents, RAG, Deep Research, tool use, and multi-agent topics.
- Unified paper schema with `final_score` ranking.
- RAG corpus export with structured fallback summaries.
- BM25 keyword retrieval.
- ChromaDB dense retrieval with Qwen embeddings.
- Hybrid retrieval with candidate deduplication and score fusion.
- Cohere Rerank precision ranking with fallback to hybrid score.
- Gemini-powered Chinese `/ask` answers with retrieval-only fallback.
- DeepResearch Lite `/research` workflow: planner, retriever, evidence builder, writer, verifier.
- Citation verification and optional RAGAS-style evaluation output.
- GitHub Pages paper display plus local Research Demo page.

## Tech Stack

- Python, FastAPI, Uvicorn
- OpenAlex Works API
- BM25 via `rank_bm25`
- ChromaDB
- Qwen OpenAI-compatible embedding API
- Cohere Rerank
- Gemini via `google-genai`
- GitHub Actions and GitHub Pages

## Project Structure

```text
config/                    Research topic configuration
data/                      Raw OpenAlex data, summaries, ChromaDB index
docs/                      Deployment and project audit docs
evaluation/                Citation verifier and evaluation scripts
llm/                       Gemini provider
retrieval/                 BM25, dense, hybrid, rerank modules
scripts/                   Data, indexing, and test scripts
web/                       GitHub Pages frontend and data
rag_api.py                 FastAPI backend
research_agent.py          DeepResearch Lite workflow
```

## Environment Variables

Copy `.env.example` to `.env` locally and fill only the keys you need:

```text
OPENALEX_API_KEY=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_EMBEDDING_MODEL=text-embedding-v3
COHERE_API_KEY=
COHERE_RERANK_MODEL=rerank-v3.5
```

Never commit `.env`.

## Local Run

```bash
python scripts/check_env.py
python scripts/fetch_openalex_papers.py
python scripts/normalize_papers.py
python scripts/export_rag_corpus.py
python scripts/build_chroma_index.py
python scripts/test_hybrid_retrieval.py
python scripts/test_rerank.py
uvicorn rag_api:app --reload --port 8000
```

Open `web/research.html` and keep the backend at `http://127.0.0.1:8000`.

## API Examples

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What papers discuss hallucination in retrieval augmented generation?\",\"top_k\":5,\"use_rerank\":true}"
```

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d "{\"topic\":\"hallucination evaluation in retrieval augmented generation\",\"top_k\":5}"
```

## GitHub Actions

`.github/workflows/update-papers.yml` runs the low-cost static publishing pipeline:

```text
OpenAlex -> normalize papers.json -> export rag_corpus.json -> deploy web/ to GitHub Pages
```

It only needs `OPENALEX_API_KEY`. Gemini, Qwen, and Cohere are intentionally not called in GitHub Actions.

## GitHub Pages

GitHub Pages hosts static files under `web/`. It can show paper data and the Research Demo shell, but FastAPI must run elsewhere because Pages cannot execute Python or store private API keys.

## Evaluation

Run:

```bash
python -m evaluation.eval_ragas
```

Outputs:

- `eval_results/ragas_results.json`
- `eval_results/citation_verification.json`

If RAGAS or an evaluation LLM is unavailable, the script writes skipped/heuristic results instead of crashing.

## Fallbacks

- Missing Qwen or ChromaDB: fallback to BM25.
- Missing Cohere: fallback to `hybrid_score`.
- Missing Gemini: fallback to retrieval-only Chinese summaries.
- Missing optional evaluation dependencies: write skipped evaluation outputs.

## Completed Result

The current project can fetch OpenAlex papers, generate `papers.json`, export `rag_corpus.json`, build a ChromaDB index, run hybrid retrieval, rerank with Cohere, answer questions through FastAPI, and generate DeepResearch Lite reports.

## Future Work

- Railway deployment for `rag_api.py`.
- Stronger RAGAS evaluation with a dedicated evaluation LLM.
- Semantic deduplication with embeddings.
- User reading status sync.
- Email or Feishu daily paper push.
