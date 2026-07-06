# GitHub Deployment Guide

## Create A New Repository

1. Create a new GitHub repository named `AI-PaperFlow`.
2. Keep it private while configuring secrets, then switch visibility when ready.
3. Push this project to the new remote:

```bash
git remote add origin https://github.com/<your-name>/AI-PaperFlow.git
git branch -M main
git push -u origin main
```

## GitHub Secrets

Add only this secret for GitHub Actions:

- `OPENALEX_API_KEY`

Do not add Gemini, Qwen, or Cohere keys to the Pages frontend. Those APIs should run only in a local backend or a server deployment such as Railway.

## Enable GitHub Pages

1. Open repository Settings.
2. Go to Pages.
3. Select GitHub Actions as the source.
4. Run `Update AI-PaperFlow Papers` manually once.

## Why FastAPI Cannot Run On GitHub Pages

GitHub Pages only serves static files. It cannot run Python, FastAPI, ChromaDB, or private environment variables. The `web/` directory can be hosted on Pages, while `rag_api.py` must run locally or on a backend platform.

## Railway Deployment Later

For Railway, configure these environment variables in Railway settings:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_EMBEDDING_MODEL`
- `COHERE_API_KEY`
- `COHERE_RERANK_MODEL`

Then start the API with:

```bash
uvicorn rag_api:app --host 0.0.0.0 --port $PORT
```
