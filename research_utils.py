from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


CORPUS_PATH = Path("web/data/rag_corpus.json")


def float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def authors_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(author) for author in value if str(author).strip())
    return str(value or "")


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"RAG corpus file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("items", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    return [record for record in records if isinstance(record, dict)]


def paper_summary(record: dict[str, Any]) -> str:
    structured_summary = record.get("structured_summary") if isinstance(record.get("structured_summary"), dict) else {}
    parts = [
        str(record.get("problem") or structured_summary.get("problem") or "").strip(),
        str(record.get("method") or structured_summary.get("method") or "").strip(),
        str(record.get("innovation") or structured_summary.get("innovation") or "").strip(),
    ]
    summary = " ".join(part for part in parts if part)
    return summary or str(record.get("abstract") or "")


def format_paper(record: dict[str, Any], similarity: float = 0.0, distance: float | None = None) -> dict[str, Any]:
    paper = {
        "title": str(record.get("title") or ""),
        "authors": authors_text(record.get("authors")),
        "final_score": float_value(record.get("final_score")),
        "similarity": round(float_value(similarity), 4),
        "source_url": str(record.get("source_url") or ""),
        "pdf_url": str(record.get("pdf_url") or ""),
        "summary": paper_summary(record),
        "abstract": str(record.get("abstract") or ""),
        "method": str((record.get("structured_summary") or {}).get("method") or record.get("method") or ""),
        "innovation": str((record.get("structured_summary") or {}).get("innovation") or record.get("innovation") or ""),
        "why_relevant": str((record.get("structured_summary") or {}).get("why_relevant") or record.get("why_relevant") or ""),
        "source_url": str(record.get("source_url") or ""),
        "paper_id": str(record.get("paper_id") or ""),
        "matched_topic_names": record.get("matched_topic_names", []),
    }
    if distance is not None:
        paper["distance"] = round(float_value(distance), 4)
    return paper


def tfidf_search(query: str, top_k: int = 3, corpus_path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    records = load_corpus(corpus_path)
    if not records:
        raise ValueError("RAG corpus is empty.")
    texts = [str(record.get("text_for_embedding") or "") for record in records]
    if not any(text.strip() for text in texts):
        raise ValueError("RAG corpus has no text_for_embedding content.")

    if TfidfVectorizer is None or cosine_similarity is None:
        return lexical_search(query, records, top_k)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).ravel()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [format_paper(records[int(index)], similarity=float(similarities[int(index)])) for index in top_indices]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def lexical_search(query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_terms = set(tokenize(query))
    scored = []
    for record in records:
        text = str(record.get("text_for_embedding") or "")
        terms = set(tokenize(text))
        if not query_terms or not terms:
            score = 0.0
        else:
            score = len(query_terms & terms) / math.sqrt(len(query_terms) * len(terms))
        scored.append((score, record))
    scored.sort(key=lambda item: (item[0], float_value(item[1].get("final_score"))), reverse=True)
    return [format_paper(record, similarity=score) for score, record in scored[:top_k]]


def dedupe_papers_by_title(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for paper in papers:
        key = str(paper.get("title") or paper.get("source_url") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique


def build_evidence_pack(papers: list[dict[str, Any]], max_items: int = 10) -> list[dict[str, Any]]:
    evidence = []
    for paper in papers[:max_items]:
        evidence.append(
            {
                "title": paper.get("title", ""),
                "method": paper.get("method", ""),
                "innovation": paper.get("innovation", ""),
                "summary": paper.get("summary", ""),
                "why_relevant": paper.get("why_relevant", ""),
                "source_url": paper.get("source_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "final_score": paper.get("final_score", 0.0),
            }
        )
    return evidence
