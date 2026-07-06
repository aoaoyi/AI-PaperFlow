from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # Keep local smoke tests usable when optional deps are not installed yet.
    BM25Okapi = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = ROOT / "web" / "data" / "rag_corpus.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


class BM25Retriever:
    def __init__(self, corpus_path: Path = DEFAULT_CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self.items = load_corpus(corpus_path)
        self.tokenized_corpus = [tokenize(item.get("text_for_embedding", "")) for item in self.items]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi and self.items else None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.items or not query.strip():
            return []

        if self.bm25:
            scores = self.bm25.get_scores(tokenize(query))
        else:
            # Fallback keeps the command runnable; production uses rank_bm25 when installed.
            query_terms = set(tokenize(query))
            scores = [sum(tokens.count(term) for term in query_terms) for tokens in self.tokenized_corpus]
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for index, score in ranked:
            item = self.items[index]
            results.append(
                {
                    "paper_id": item.get("paper_id", ""),
                    "title": item.get("title", ""),
                    "final_score": float(item.get("final_score") or 0),
                    "bm25_score": round(float(score), 4),
                    "source_url": item.get("source_url", ""),
                    "pdf_url": item.get("pdf_url", ""),
                    "matched_topic_names": item.get("matched_topic_names", []),
                }
            )
        return results
