# AI-PaperFlow Research Agent

## Project Overview

AI-PaperFlow extends the original paper-daily workflow into a local research demo for automatic paper tracking, personalized ranking, deduplication, hybrid retrieval, RAG question answering, DeepResearch Lite report generation, reading status management, daily digest email, and retrieval evaluation.

The original GitHub Pages paper display flow remains unchanged. The new research features are optional local extensions built around `web/data/rag_corpus.json`.

## System Architecture

```text
paper-daily collector
        |
        v
web/data/papers.json + web/data/conference_papers.json
        |
        v
scripts/export_rag_corpus.py
        |
        v
web/data/rag_corpus.json
        |
        +--> BM25 / lexical retrieval fallback
        +--> Qwen Embedding API + ChromaDB semantic retrieval
        +--> Cohere reranking
        +--> FastAPI Paper QA
        +--> DeepResearch Lite
        +--> Evaluation
        +--> Email digest
```

## Main Modules

### Paper Tracking

The existing paper-daily workflow tracks arXiv and conference papers, scores relevance, summarizes papers, and publishes the static paper page through GitHub Actions and GitHub Pages.

### Ranking and Deduplication

The collector preserves the original relevance `score` and adds a personalized `final_score`. It also deduplicates papers by arXiv ID, DOI, and normalized title before final display data is written.

### RAG Corpus

`scripts/export_rag_corpus.py` exports normalized records to `web/data/rag_corpus.json` without changing the original display JSON files.

### Hybrid Retrieval

`retrieval_engine.py` provides hybrid retrieval with BM25 recall, optional Qwen/OpenAI-compatible embeddings, ChromaDB vector storage, and optional Cohere reranking. If any advanced service is unavailable, the API falls back to BM25 or lexical retrieval.

### Paper QA

`rag_api.py` exposes `/ask` and `/ask_v2`. `/ask` returns hybrid retrieval results with fallback metadata. `/ask_v2` is kept as a compatibility endpoint.

### DeepResearch Lite

`research_agent.py` implements a simplified multi-step research workflow:

- Planner
- Retriever
- Evidence Aggregator
- Writer
- Verifier

### Email Digest

`scripts/send_daily_digest.py` sends the daily top papers by email when SMTP variables are configured. If configuration is incomplete, it prints the digest to logs.

### Evaluation

`eval_paperflow.py` evaluates retrieval quality and latency using `eval_queries.csv`, then writes results to `eval_results/paperflow_eval.json`. `eval_ragas.py` writes RAGAS-style fallback metrics and citation verification results.

## API Usage

Start the API:

```bash
uvicorn rag_api:app --reload --port 8000
```

### GET /health

Checks corpus, LLM, and Milvus availability.

### POST /ask

TF-IDF based paper QA with optional LLM answer generation.

```json
{
  "question": "Which paper is about inference serving?"
}
```

### POST /ask_v2

Semantic retrieval first, TF-IDF fallback.

```json
{
  "question": "Which paper is about inference serving?"
}
```

### POST /research

Generates a DeepResearch Lite report.

```json
{
  "topic": "LLM agent planning",
  "language": "zh"
}
```

## Environment Variables

LLM variables:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

Embedding variables:

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

Generation and reranking:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `COHERE_API_KEY`

SMTP variables:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `DIGEST_TO`

Optional vector store variables:

- ChromaDB uses local `chroma_db/` by default.

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Export RAG corpus:

```bash
python scripts/export_rag_corpus.py
```

Build semantic index:

```bash
python scripts/build_milvus_index.py
```

Start API:

```bash
uvicorn rag_api:app --reload --port 8000
```

Run evaluation:

```bash
python eval_paperflow.py
python eval_ragas.py
```

Send or print daily digest:

```bash
python scripts/send_daily_digest.py
```

## Security Note

API keys, embedding keys, and SMTP passwords must be stored in environment variables or GitHub Secrets. They should never be committed to the repository.
