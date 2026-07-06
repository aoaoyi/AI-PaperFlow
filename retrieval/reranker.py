from __future__ import annotations

import os
import re
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    import cohere
except ImportError:
    cohere = None


load_dotenv()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def document_for_rerank(candidate: dict[str, Any]) -> str:
    summary = candidate.get("structured_summary") if isinstance(candidate.get("structured_summary"), dict) else {}
    topics = ", ".join(candidate.get("topics", []) or [])
    parts = [
        ("Title", candidate.get("title")),
        ("Abstract", candidate.get("abstract")),
        ("Topics", topics),
        ("Problem", summary.get("problem")),
        ("Method", summary.get("method")),
        ("Innovation", summary.get("innovation")),
        ("Why relevant", summary.get("why_relevant")),
    ]
    text = "\n".join(f"{label}: {normalize_space(value)}" for label, value in parts if normalize_space(value))
    return text[:4000]


def public_result(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": candidate.get("paper_id", ""),
        "title": candidate.get("title", ""),
        "rerank_score": candidate.get("rerank_score"),
        "hybrid_score": candidate.get("hybrid_score", 0),
        "bm25_score": candidate.get("bm25_score", 0),
        "dense_score": candidate.get("dense_score", 0),
        "final_score": candidate.get("final_score", 0),
        "retrieval_sources": candidate.get("retrieval_sources", []),
        "source_url": candidate.get("source_url", ""),
        "pdf_url": candidate.get("pdf_url", ""),
        "matched_topic_names": candidate.get("matched_topic_names", []),
    }


class CohereReranker:
    def __init__(self) -> None:
        self.api_key = os.getenv("COHERE_API_KEY")
        self.model = os.getenv("COHERE_RERANK_MODEL") or "rerank-v3.5"
        self.client = cohere.Client(self.api_key) if cohere and self.api_key else None

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
        fallback_results = [public_result(candidate) for candidate in candidates[:top_k]]
        if not self.api_key:
            return {
                "results": fallback_results,
                "rerank_used": False,
                "fallback_reason": "COHERE_API_KEY missing",
            }
        if cohere is None or self.client is None:
            return {
                "results": fallback_results,
                "rerank_used": False,
                "fallback_reason": "cohere package missing",
            }

        documents = [document_for_rerank(candidate) for candidate in candidates]
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
            reranked: list[dict[str, Any]] = []
            for result in response.results:
                candidate = dict(candidates[result.index])
                candidate["rerank_score"] = round(float(result.relevance_score), 6)
                reranked.append(public_result(candidate))
            return {
                "results": reranked,
                "rerank_used": True,
                "fallback_reason": None,
            }
        except Exception as exc:
            return {
                "results": fallback_results,
                "rerank_used": False,
                "fallback_reason": type(exc).__name__,
            }
