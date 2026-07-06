from __future__ import annotations

import json
from pathlib import Path

from research_agent import verify_report
from research_utils import load_corpus, tfidf_search


OUTPUT_PATH = Path("eval_results/ragas_eval.json")


def heuristic_eval() -> dict:
    queries = [
        "LLM agent planning",
        "RAG retrieval",
        "inference serving",
    ]
    rows = []
    for query in queries:
        papers = tfidf_search(query, top_k=3)
        report = "\n".join(["# References", *[f"- {paper['title']}" for paper in papers]])
        verification = verify_report(report, papers)
        rows.append(
            {
                "query": query,
                "context_precision": round(sum(1 for paper in papers if paper.get("similarity", 0) > 0) / max(1, len(papers)), 4),
                "context_recall": 0.0,
                "faithfulness": 1.0 if verification["verification_status"] == "passed" else 0.5,
                "answer_relevance": round(max([paper.get("similarity", 0) for paper in papers] or [0]), 4),
                "verification": verification,
            }
        )
    return {"mode": "heuristic_ragas_fallback", "items": rows}


def main() -> None:
    load_corpus()
    result = heuristic_eval()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RAGAS-style evaluation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
