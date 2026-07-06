#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.chroma_retriever import ChromaRetriever  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


QUERY = "retrieval augmented generation for research papers"
CORPUS_PATH = ROOT / "web" / "data" / "rag_corpus.json"


def print_result(prefix: str, rank: int, result: dict) -> None:
    title = str(result.get("title") or "")[:120]
    hybrid_score = result.get("hybrid_score", "")
    sources = ",".join(result.get("retrieval_sources", []))
    final_score = result.get("final_score", 0)
    print(f"{prefix} {rank}. title={title}")
    if hybrid_score != "":
        print(f"   hybrid_score={hybrid_score} retrieval_sources={sources} final_score={final_score}")
    else:
        print(f"   final_score={final_score}")


def main() -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    count = payload.get("count", len(payload.get("items", []))) if isinstance(payload, dict) else len(payload)
    print(f"loaded corpus count: {count}")
    print(f"query: {QUERY}")

    bm25 = BM25Retriever(CORPUS_PATH)
    bm25_results = bm25.search(QUERY, top_k=5)
    print("BM25 Top 5:")
    for rank, result in enumerate(bm25_results, start=1):
        print_result("BM25", rank, result)

    dense = ChromaRetriever()
    dense_results = dense.search(QUERY, top_k=5)
    if dense_results:
        print("Dense Top 5:")
        for rank, result in enumerate(dense_results, start=1):
            print_result("Dense", rank, result)
    else:
        print("Dense retrieval skipped, fallback to BM25 only.")

    hybrid = HybridRetriever(CORPUS_PATH)
    hybrid_response = hybrid.search(QUERY, top_k=5, use_rerank=False)
    hybrid_results = hybrid_response["results"]
    print("Hybrid Top 5:")
    for rank, result in enumerate(hybrid_results, start=1):
        print_result("Hybrid", rank, result)


if __name__ == "__main__":
    main()
