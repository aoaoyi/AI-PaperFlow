# Project File Audit

This audit records the cleanup performed while preparing the repository as `AI-PaperFlow`.

## Core Files To Keep

- `.git/`
- `.github/workflows/update-papers.yml`
- `.gitignore`
- `.env.example`
- `requirements.txt`
- `README.md`
- `README_MY_IMPROVEMENT.md`
- `rag_api.py`
- `research_agent.py`
- `research_prompts.py`
- `research_utils.py`
- `config/`
- `data/`
- `docs/`
- `evaluation/`
- `eval_results/`
- `llm/`
- `retrieval/`
- `scripts/`
- `tests/`
- `web/`

## Deleted Or Cleaned Files

- `__pycache__/`
- `evaluation/__pycache__/`
- `llm/__pycache__/`
- `retrieval/__pycache__/`
- `scripts/__pycache__/`

These are generated Python bytecode caches and are ignored by `.gitignore`.

## Moved To `archive/legacy/`

- `.github/workflows/paper-daily.yml` -> `archive/legacy/.github/workflows/paper-daily.yml`
- `.github/workflows/daily_digest.yml` -> `archive/legacy/.github/workflows/daily_digest.yml`

The legacy workflows were preserved for reference because they may contain upstream `paper-daily` behavior, but they are not suitable as default AI-PaperFlow workflows because they include older collection/digest behavior and broader secret usage.

## Uncertain Files Kept For Review

- `README_RESEARCH_AGENT.md`
- `eval_paperflow.py`
- `eval_ragas.py`
- `retrieval_engine.py`
- `vector_store.py`
- `scripts/build_milvus_index.py`
- `scripts/collect_openalex_papers.py`
- `scripts/send_daily_digest.py`

These may be useful as legacy references or optional extensions. They were kept instead of deleted.

## Security Notes

- `.env` was not modified.
- `.env` is ignored by `.gitignore`.
- `git ls-files .env` returned no tracked `.env` file.
- Frontend files do not contain API keys.
