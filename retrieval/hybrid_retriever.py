from __future__ import annotations

from pathlib import Path
from typing import Any

from .bm25_retriever import BM25Retriever, DEFAULT_CORPUS_PATH, load_corpus
from .chroma_retriever import ChromaRetriever
from .reranker import CohereReranker, public_result


class HybridRetriever:
    def __init__(self, corpus_path: Path = DEFAULT_CORPUS_PATH) -> None:
        self.bm25 = BM25Retriever(corpus_path)
        self.chroma = ChromaRetriever()
        self.corpus_items = load_corpus(corpus_path)
        self.corpus_by_id = {str(item.get("paper_id") or ""): item for item in self.corpus_items}

    def _enrich(self, item: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(item.get("paper_id") or "")
        full = self.corpus_by_id.get(paper_id, {})
        enriched = {**full, **item}
        enriched.setdefault("structured_summary", full.get("structured_summary", {}))
        enriched.setdefault("abstract", full.get("abstract", ""))
        enriched.setdefault("topics", full.get("topics", []))
        enriched.setdefault("matched_topic_names", full.get("matched_topic_names", []))
        return enriched

    def search(self, query: str, top_k: int = 5, use_rerank: bool = True, candidate_k: int = 30) -> dict[str, Any]:
        bm25_results = self.bm25.search(query, top_k=20)
        dense_results = self.chroma.search(query, top_k=20)
        merged: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(bm25_results, start=1):
            paper_id = str(result.get("paper_id") or "")
            if not paper_id:
                continue
            item = merged.setdefault(paper_id, {**self._enrich(result), "retrieval_sources": []})
            item["bm25_score"] = result.get("bm25_score", 0)
            item["bm25_rank_score"] = 1 / rank
            item["retrieval_sources"].append("bm25")

        for rank, result in enumerate(dense_results, start=1):
            paper_id = str(result.get("paper_id") or "")
            if not paper_id:
                continue
            item = merged.setdefault(paper_id, {**self._enrich(result), "retrieval_sources": []})
            item["title"] = item.get("title") or result.get("title", "")
            item["final_score"] = max(float(item.get("final_score") or 0), float(result.get("final_score") or 0))
            item["dense_score"] = result.get("dense_score", 0)
            item["dense_rank_score"] = 1 / rank
            item["source_url"] = item.get("source_url") or result.get("source_url", "")
            item["pdf_url"] = item.get("pdf_url") or result.get("pdf_url", "")
            item["retrieval_sources"].append("dense")

        fused: list[dict[str, Any]] = []
        for item in merged.values():
            bm25_rank_score = float(item.get("bm25_rank_score") or 0)
            dense_rank_score = float(item.get("dense_rank_score") or 0)
            final_score_norm = float(item.get("final_score") or 0) / 20
            item["hybrid_score"] = round(bm25_rank_score + dense_rank_score + final_score_norm, 4)
            item.setdefault("bm25_score", 0)
            item.setdefault("dense_score", 0)
            item["retrieval_sources"] = list(dict.fromkeys(item.get("retrieval_sources", [])))
            fused.append(item)

        fused.sort(key=lambda item: item["hybrid_score"], reverse=True)
        candidates = fused[:candidate_k]
        if use_rerank:
            rerank_result = CohereReranker().rerank(query, candidates, top_k=top_k)
            return {
                "results": rerank_result["results"],
                "rerank_used": rerank_result["rerank_used"],
                "fallback_reason": rerank_result["fallback_reason"],
                "candidate_count": len(candidates),
            }

        return {
            "results": [public_result(item) for item in candidates[:top_k]],
            "rerank_used": False,
            "fallback_reason": "rerank disabled",
            "candidate_count": len(candidates),
        }
