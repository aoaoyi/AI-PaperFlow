#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPERS_INPUT_PATH = ROOT / "web" / "data" / "papers.json"
RAG_OUTPUT_PATH = ROOT / "web" / "data" / "rag_corpus.json"
SUMMARY_OUTPUT_PATH = ROOT / "data" / "rag_corpus_summary.json"
MAX_EMBEDDING_TEXT_LENGTH = 8000
SUMMARY_KEYS = ("problem", "method", "innovation", "evidence", "limitation", "why_relevant")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def compact_text(value: Any, max_length: int = MAX_EMBEDDING_TEXT_LENGTH) -> str:
    text = normalize_space(value)
    return text[:max_length].rstrip() if len(text) > max_length else text


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_space(item) for item in value if normalize_space(item)]
    if isinstance(value, str) and value.strip():
        return [normalize_space(value)]
    return []


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def load_papers() -> list[dict[str, Any]]:
    payload = load_json(PAPERS_INPUT_PATH)
    if isinstance(payload, dict):
        papers = payload.get("papers", [])
    elif isinstance(payload, list):
        papers = payload
    else:
        papers = []
    return [paper for paper in papers if isinstance(paper, dict)]


def fallback_structured_summary(paper: dict[str, Any]) -> dict[str, str]:
    title = normalize_space(paper.get("title"))
    abstract = normalize_space(paper.get("abstract") or paper.get("summary"))
    topics = ", ".join(string_list(paper.get("topics") or paper.get("categories")))
    matched_topics = ", ".join(string_list(paper.get("matched_topic_names")))
    context = abstract or title

    # This fallback is intentionally deterministic and local; it does not call any LLM API.
    return {
        "problem": compact_text(context or f"This paper is related to {topics}.", 700),
        "method": "",
        "innovation": "",
        "evidence": "",
        "limitation": "",
        "why_relevant": compact_text(
            f"Matched topics: {matched_topics or topics}. Title: {title}".strip(),
            700,
        ),
    }


def structured_summary_for(paper: dict[str, Any]) -> dict[str, str]:
    existing = paper.get("structured_summary")
    if isinstance(existing, dict):
        summary = {key: normalize_space(existing.get(key) or existing.get("limitations")) for key in SUMMARY_KEYS}
        if any(summary.values()):
            return summary
    return fallback_structured_summary(paper)


def build_text_for_embedding(record: dict[str, Any]) -> str:
    summary = record["structured_summary"]
    parts = [
        ("Title", record["title"]),
        ("Abstract", record["abstract"]),
        ("Topics", ", ".join(record["topics"])),
        ("Matched topics", ", ".join(record["matched_topic_names"])),
        ("Problem", summary.get("problem")),
        ("Method", summary.get("method")),
        ("Innovation", summary.get("innovation")),
        ("Limitation", summary.get("limitation")),
        ("Why relevant", summary.get("why_relevant")),
    ]
    text = "\n".join(f"{label}: {normalize_space(value)}" for label, value in parts if normalize_space(value))
    return compact_text(text)


def paper_to_record(paper: dict[str, Any]) -> dict[str, Any]:
    title = normalize_space(paper.get("title"))
    abstract = normalize_space(paper.get("abstract") or paper.get("summary"))
    topics = string_list(paper.get("topics") or paper.get("categories"))
    matched_topic_names = string_list(paper.get("matched_topic_names"))
    final_score = float_value(paper.get("final_score"))
    structured_summary = structured_summary_for(paper)
    record = {
        "paper_id": normalize_space(paper.get("paper_id")),
        "openalex_id": normalize_space(paper.get("openalex_id")),
        "doi": normalize_space(paper.get("doi")),
        "title": title,
        "authors": string_list(paper.get("authors")),
        "abstract": abstract,
        "publication_year": int_value(paper.get("publication_year")),
        "publication_date": normalize_space(paper.get("publication_date") or paper.get("published")),
        "cited_by_count": int_value(paper.get("cited_by_count")),
        "topics": topics,
        "matched_topic_ids": string_list(paper.get("matched_topic_ids")),
        "matched_topic_names": matched_topic_names,
        "source_url": normalize_space(paper.get("source_url") or paper.get("paper_url")),
        "pdf_url": normalize_space(paper.get("pdf_url")),
        "open_access": bool(paper.get("open_access")),
        "final_score": final_score,
        "structured_summary": structured_summary,
    }
    record["text_for_embedding"] = build_text_for_embedding(record)
    record["metadata"] = {
        "paper_id": record["paper_id"],
        "title": record["title"],
        "publication_year": record["publication_year"],
        "final_score": final_score,
        "source_url": record["source_url"],
        "pdf_url": record["pdf_url"],
        "matched_topic_names": matched_topic_names,
    }
    return record


def export_rag_corpus() -> tuple[dict[str, Any], dict[str, Any]]:
    papers = load_papers()
    items: list[dict[str, Any]] = []
    skipped_no_title = 0
    skipped_empty_text = 0

    for paper in papers:
        record = paper_to_record(paper)
        if not record["title"]:
            skipped_no_title += 1
            continue
        if not record["text_for_embedding"]:
            skipped_empty_text += 1
            continue
        items.append(record)

    items.sort(key=lambda item: item.get("final_score", 0), reverse=True)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    text_lengths = [len(item["text_for_embedding"]) for item in items]
    payload = {
        "generated_at": generated_at,
        "source": "papers.json",
        "count": len(items),
        "items": items,
    }
    summary = {
        "input_paper_count": len(papers),
        "exported_count": len(items),
        "skipped_no_title": skipped_no_title,
        "skipped_empty_text": skipped_empty_text,
        "missing_abstract_count": sum(1 for paper in papers if not normalize_space(paper.get("abstract") or paper.get("summary"))),
        "average_text_length": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0,
        "generated_at": generated_at,
    }
    return payload, summary


def validate_rag_corpus(path: Path) -> None:
    payload = load_json(path)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(payload, dict):
        raise ValueError("rag_corpus.json must be a JSON object.")
    if len(items) != payload.get("count"):
        raise ValueError("rag_corpus.json count does not match items length.")
    for index, item in enumerate(items[:5], start=1):
        missing = [key for key in ("paper_id", "title", "text_for_embedding", "final_score") if key not in item]
        if missing:
            raise ValueError(f"Item {index} is missing required fields: {', '.join(missing)}")
        if not normalize_space(item.get("text_for_embedding")):
            raise ValueError(f"Item {index} has empty text_for_embedding.")


def main() -> None:
    try:
        payload, summary = export_rag_corpus()
        write_json(RAG_OUTPUT_PATH, payload)
        write_json(SUMMARY_OUTPUT_PATH, summary)
        validate_rag_corpus(RAG_OUTPUT_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"RAG corpus export failed: {type(exc).__name__}: {exc}") from exc

    skipped_total = summary["skipped_no_title"] + summary["skipped_empty_text"]
    print(f"input paper count: {summary['input_paper_count']}")
    print(f"exported count: {summary['exported_count']}")
    print(f"skipped count: {skipped_total}")
    print(f"missing abstract count: {summary['missing_abstract_count']}")
    print(f"average text length: {summary['average_text_length']}")
    print(f"output path: {RAG_OUTPUT_PATH}")
    print(f"summary path: {SUMMARY_OUTPUT_PATH}")
    print("validation: passed")


if __name__ == "__main__":
    main()
