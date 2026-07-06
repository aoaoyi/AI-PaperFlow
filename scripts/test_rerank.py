#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


QUERY = "retrieval augmented generation hallucination evaluation"


def main() -> None:
    retriever = HybridRetriever()
    response = retriever.search(QUERY, use_rerank=True, candidate_k=30, top_k=5)
    print(f"query: {QUERY}")
    print(f"rerank_used: {str(response['rerank_used']).lower()}")
    print(f"fallback_reason: {response['fallback_reason']}")
    print(f"candidate_count: {response['candidate_count']}")
    print("Top 5:")
    for rank, result in enumerate(response["results"], start=1):
        title = str(result.get("title") or "")[:120]
        sources = ",".join(result.get("retrieval_sources", []))
        print(f"{rank}. title={title}")
        print(
            f"   rerank_score={result.get('rerank_score')} "
            f"hybrid_score={result.get('hybrid_score')} "
            f"retrieval_sources={sources}"
        )


if __name__ == "__main__":
    main()
