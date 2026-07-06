#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.citation_verifier import verify_citations  # noqa: E402
from rag_api import ask, AskRequest  # noqa: E402


QUERIES_PATH = ROOT / "eval_queries.csv"
RESULTS_DIR = ROOT / "eval_results"
RAGAS_OUTPUT = RESULTS_DIR / "ragas_results.json"
CITATION_OUTPUT = RESULTS_DIR / "citation_verification.json"


def load_queries() -> list[str]:
    if not QUERIES_PATH.exists():
        return ["retrieval augmented generation hallucination evaluation"]
    with QUERIES_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row.get("question", "").strip() for row in reader if row.get("question", "").strip()]


def heuristic_scores(response: dict[str, Any]) -> dict[str, float]:
    papers = response.get("retrieved_papers", [])
    answer = response.get("answer", "")
    has_context = bool(papers)
    mentions_title = any(str(paper.get("title", "")).lower() in answer.lower() for paper in papers)
    return {
        "context_precision": 1.0 if has_context else 0.0,
        "context_recall": min(len(papers) / 5, 1.0),
        "faithfulness": 1.0 if mentions_title or response.get("llm_used") is False else 0.5,
        "answer_relevance": 1.0 if answer.strip() else 0.0,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    results = []
    citation_results = []
    for question in queries:
        try:
            response = ask(AskRequest(question=question, top_k=5, use_rerank=True))
            scores = heuristic_scores(response)
            status = "heuristic"
        except Exception as exc:
            response = {"question": question, "answer": "", "retrieved_papers": [], "error": type(exc).__name__}
            scores = {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
            }
            status = "skipped"
        verification = verify_citations(response.get("answer", ""), response.get("retrieved_papers", []))
        results.append({"question": question, "status": status, "metrics": scores, "ragas_note": "RAGAS unavailable or optional; heuristic metrics saved."})
        citation_results.append({"question": question, "verification": verification})

    RAGAS_OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CITATION_OUTPUT.write_text(json.dumps(citation_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ragas_results: {RAGAS_OUTPUT}")
    print(f"citation_verification: {CITATION_OUTPUT}")
    print("status: completed")


if __name__ == "__main__":
    main()

