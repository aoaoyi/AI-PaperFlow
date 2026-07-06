from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from research_utils import load_corpus, tfidf_search

try:
    import retrieval_engine
except Exception:  # pragma: no cover
    retrieval_engine = None


QUERY_PATH = Path("eval_queries.csv")
OUTPUT_PATH = Path("eval_results/paperflow_eval.json")


def load_queries() -> list[dict[str, str]]:
    with QUERY_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def paper_blob(paper: dict[str, Any]) -> str:
    return " ".join(
        str(paper.get(field) or "")
        for field in ("title", "summary", "abstract", "text_for_embedding", "method", "innovation")
    ).lower()


def is_hit(papers: list[dict[str, Any]], query: dict[str, str], top_n: int) -> bool:
    expected_title = (query.get("expected_title") or "").strip().lower()
    expected_keyword = (query.get("expected_keyword") or "").strip().lower()
    for paper in papers[:top_n]:
        if expected_title and expected_title in str(paper.get("title") or "").lower():
            return True
        if expected_keyword and expected_keyword in paper_blob(paper):
            return True
    return False


def evaluate_mode(mode: str, queries: list[dict[str, str]]) -> dict[str, Any]:
    top1 = 0
    top3 = 0
    latencies = []
    details = []
    for query in queries:
        started = time.perf_counter()
        if mode == "hybrid":
            if retrieval_engine is None:
                raise RuntimeError("retrieval_engine is unavailable.")
            papers = retrieval_engine.hybrid_search(query["query"], top_k=3)["papers"]
        else:
            papers = tfidf_search(query["query"], top_k=3)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        top1_hit = is_hit(papers, query, 1)
        top3_hit = is_hit(papers, query, 3)
        top1 += int(top1_hit)
        top3 += int(top3_hit)
        details.append(
            {
                "query": query["query"],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "results": [{"title": paper.get("title", ""), "similarity": paper.get("similarity", 0.0)} for paper in papers],
            }
        )
    total = len(queries)
    return {
        "retrieval_mode": mode,
        "total_queries": total,
        "top1_hit_count": top1,
        "top3_hit_count": top3,
        "top1_hit_rate": round(top1 / total if total else 0.0, 4),
        "top3_hit_rate": round(top3 / total if total else 0.0, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies) if latencies else 0.0, 2),
        "details": details,
    }


def main() -> None:
    load_corpus()
    queries = load_queries()
    results = [evaluate_mode("tfidf", queries)]
    if retrieval_engine is not None:
        try:
            results.append(evaluate_mode("hybrid", queries))
        except Exception as exc:
            print(f"Hybrid evaluation skipped: {exc}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['retrieval_mode']}: top1={result['top1_hit_rate']:.4f}, "
            f"top3={result['top3_hit_rate']:.4f}, avg_latency_ms={result['avg_latency_ms']}"
        )
    print(f"Wrote evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
