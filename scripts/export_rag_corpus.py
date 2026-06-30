#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("web/data")
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "rag_corpus.json"
PAPER_DATA_FILES = ("papers.json", "conference_papers.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: Any) -> str:
    text = normalize_space(str(value or "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def load_papers(data_dir: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for name in PAPER_DATA_FILES:
        path = data_dir / name
        if not path.exists():
            continue
        payload = load_json(path)
        for item in payload.get("papers", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict):
                papers.append(item)
    return papers


def text_for_embedding(record: dict[str, Any]) -> str:
    categories = ", ".join(record["categories"])
    parts = [
        ("Title", record["title"]),
        ("Abstract", record["abstract"]),
        ("Problem", record["problem"]),
        ("Method", record["method"]),
        ("Innovation", record["innovation"]),
        ("Evidence", record["evidence"]),
        ("Why relevant", record["why_relevant"]),
        ("Categories", categories),
        ("Matched topic", record["matched_topic"]),
    ]
    return "\n".join(f"{label}: {value}" for label, value in parts if value)


def paper_to_record(paper: dict[str, Any]) -> dict[str, Any]:
    best_match = paper.get("best_match") if isinstance(paper.get("best_match"), dict) else {}
    chinese_summary = paper.get("chinese_summary") if isinstance(paper.get("chinese_summary"), dict) else {}
    record = {
        "paper_id": string_value(paper.get("id")),
        "title": string_value(paper.get("title")),
        "authors": string_list(paper.get("authors")),
        "abstract": string_value(paper.get("summary")),
        "problem": string_value(chinese_summary.get("problem")),
        "method": string_value(chinese_summary.get("method")),
        "innovation": string_value(chinese_summary.get("innovation")),
        "evidence": string_value(chinese_summary.get("evidence")),
        "why_relevant": string_value(chinese_summary.get("why_relevant")),
        "categories": string_list(paper.get("categories")),
        "matched_topic": string_value(best_match.get("topic_name")),
        "source_url": string_value(paper.get("paper_url")),
        "pdf_url": string_value(paper.get("pdf_url")),
        "score": float_value(best_match.get("score")),
        "final_score": float_value(paper.get("final_score")),
        "published_date": string_value(paper.get("published")),
    }
    record["text_for_embedding"] = text_for_embedding(record)
    return record


def is_duplicate(record: dict[str, Any], seen_ids: set[str], seen_titles: set[str], seen_urls: set[str]) -> bool:
    paper_id = normalize_key(record["paper_id"])
    title = normalize_key(record["title"])
    source_url = normalize_space(record["source_url"]).lower()
    return bool(
        (paper_id and paper_id in seen_ids)
        or (title and title in seen_titles)
        or (source_url and source_url in seen_urls)
    )


def remember_record(record: dict[str, Any], seen_ids: set[str], seen_titles: set[str], seen_urls: set[str]) -> None:
    paper_id = normalize_key(record["paper_id"])
    title = normalize_key(record["title"])
    source_url = normalize_space(record["source_url"]).lower()
    if paper_id:
        seen_ids.add(paper_id)
    if title:
        seen_titles.add(title)
    if source_url:
        seen_urls.add(source_url)


def export_rag_corpus(data_dir: Path, output_path: Path) -> tuple[int, int, int]:
    papers = load_papers(data_dir)
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    records: list[dict[str, Any]] = []

    for paper in papers:
        record = paper_to_record(paper)
        if is_duplicate(record, seen_ids, seen_titles, seen_urls):
            continue
        remember_record(record, seen_ids, seen_titles, seen_urls)
        records.append(record)

    write_json(output_path, records)
    return len(papers), len(records), len(papers) - len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-daily data as a RAG corpus JSON file.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    loaded_count, exported_count, duplicate_count = export_rag_corpus(args.data_dir, args.output)
    print(f"loaded paper count: {loaded_count}")
    print(f"exported paper count: {exported_count}")
    print(f"duplicate removed count: {duplicate_count}")
    print(f"output path: {args.output}")


if __name__ == "__main__":
    main()
