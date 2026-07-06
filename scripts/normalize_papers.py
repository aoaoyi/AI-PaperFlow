#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_PATH = ROOT / "data" / "raw_openalex_works.json"
PAPERS_OUTPUT_PATH = ROOT / "web" / "data" / "papers.json"
SUMMARY_OUTPUT_PATH = ROOT / "data" / "openalex_fetch_summary.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_hash(title: str) -> str:
    normalized = normalize_title(title)
    return "title:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def abstract_from_inverted_index(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, str(word)))
    return " ".join(word for _, word in sorted(words))


def authors_from_work(work: dict[str, Any]) -> list[str]:
    authors = []
    for authorship in work.get("authorships", []):
        if not isinstance(authorship, dict):
            continue
        name = str((authorship.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
    return authors


def topics_from_work(work: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for topic in work.get("topics", []) or []:
        if isinstance(topic, dict) and topic.get("display_name"):
            names.append(str(topic["display_name"]))
    for concept in work.get("concepts", []) or []:
        if isinstance(concept, dict) and concept.get("display_name"):
            names.append(str(concept["display_name"]))
    return list(dict.fromkeys(name for name in names if name))


def best_location(work: dict[str, Any]) -> tuple[str, str]:
    primary = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    best_oa = work.get("best_oa_location") if isinstance(work.get("best_oa_location"), dict) else {}
    landing = str(primary.get("landing_page_url") or best_oa.get("landing_page_url") or work.get("doi") or work.get("id") or "")
    pdf = str(primary.get("pdf_url") or best_oa.get("pdf_url") or "")
    return landing, pdf


def keyword_hits(entry: dict[str, Any], title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    hits = []
    for keyword in entry.get("topic_keywords", []):
        keyword_text = str(keyword).lower()
        if keyword_text and keyword_text in text:
            hits.append(str(keyword))
    return list(dict.fromkeys(hits))


def final_score(paper: dict[str, Any]) -> float:
    keyword_score = len(paper["keyword_hits"]) * 2
    year = int(paper.get("publication_year") or 0)
    if year >= 2025:
        recency_score = 5
    elif year == 2024:
        recency_score = 3
    elif year == 2023:
        recency_score = 1
    else:
        recency_score = 0
    citation_score = min(float(paper.get("cited_by_count") or 0) / 20, 5)
    pdf_score = 2 if paper.get("pdf_url") else 0
    open_access_score = 1 if paper.get("open_access") else 0
    topic_score = 3 if paper.get("matched_topic_ids") else 0
    return round(keyword_score + recency_score + citation_score + pdf_score + open_access_score + topic_score, 2)


def paper_id_for(doi: str, openalex_id: str, title: str) -> str:
    if doi:
        return doi.lower()
    if openalex_id:
        return openalex_id
    return title_hash(title)


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    work = entry.get("work") if isinstance(entry.get("work"), dict) else {}
    title = str(work.get("title") or "").strip()
    abstract = abstract_from_inverted_index(work)
    doi = str(work.get("doi") or "").strip()
    openalex_id = str(work.get("id") or "").strip()
    source_url, pdf_url = best_location(work)
    open_access = bool((work.get("open_access") or {}).get("is_oa") or pdf_url)
    hits = keyword_hits(entry, title, abstract)
    paper = {
        "paper_id": paper_id_for(doi, openalex_id, title),
        "openalex_id": openalex_id,
        "doi": doi,
        "id": paper_id_for(doi, openalex_id, title),
        "source": "OpenAlex",
        "title": title,
        "authors": authors_from_work(work),
        "abstract": abstract,
        "summary": abstract,
        "publication_year": int(work.get("publication_year") or 0),
        "publication_date": str(work.get("publication_date") or ""),
        "published": str(work.get("publication_date") or ""),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "topics": topics_from_work(work),
        "categories": topics_from_work(work),
        "matched_topic_ids": [str(entry.get("topic_id") or "")] if entry.get("topic_id") else [],
        "matched_topic_names": [str(entry.get("topic_name") or "")] if entry.get("topic_name") else [],
        "source_url": source_url,
        "paper_url": source_url,
        "pdf_url": pdf_url,
        "open_access": open_access,
        "keyword_hits": hits,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    paper["final_score"] = final_score(paper)
    return paper


def dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = (paper.get("doi") or paper.get("openalex_id") or normalize_title(paper.get("title", ""))).lower()
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None or paper.get("final_score", 0) > existing.get("final_score", 0):
            by_key[key] = paper
        elif existing is not None:
            existing["matched_topic_ids"] = list(dict.fromkeys([*existing.get("matched_topic_ids", []), *paper.get("matched_topic_ids", [])]))
            existing["matched_topic_names"] = list(dict.fromkeys([*existing.get("matched_topic_names", []), *paper.get("matched_topic_names", [])]))
            existing["keyword_hits"] = list(dict.fromkeys([*existing.get("keyword_hits", []), *paper.get("keyword_hits", [])]))
            existing["final_score"] = final_score(existing)
    return sorted(by_key.values(), key=lambda item: item.get("final_score", 0), reverse=True)


def topic_counts(papers: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paper in papers:
        for topic_id in paper.get("matched_topic_ids", []):
            counts[topic_id] = counts.get(topic_id, 0) + 1
    return counts


def normalize_all() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = load_json(RAW_INPUT_PATH)
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    normalized = [normalize_entry(entry) for entry in entries if isinstance(entry, dict)]
    normalized = [paper for paper in normalized if paper.get("title")]
    deduped = dedupe_papers(normalized)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "generated_at_iso": generated_at,
        "source": "OpenAlex",
        "count": len(deduped),
        "topics": [],
        "papers": deduped,
    }
    summary = {
        "total_raw_count": len(entries),
        "total_normalized_count": len(normalized),
        "total_after_dedup": len(deduped),
        "topic_counts": topic_counts(deduped),
        "missing_abstract_count": sum(1 for paper in deduped if not paper.get("abstract")),
        "missing_pdf_count": sum(1 for paper in deduped if not paper.get("pdf_url")),
        "generated_at": generated_at,
    }
    return payload, summary


def main() -> None:
    payload, summary = normalize_all()
    write_json(PAPERS_OUTPUT_PATH, payload)
    write_json(SUMMARY_OUTPUT_PATH, summary)
    print(f"Normalized papers: raw={summary['total_raw_count']} normalized={summary['total_normalized_count']} deduped={summary['total_after_dedup']}")
    print(f"Wrote papers: {PAPERS_OUTPUT_PATH}")
    print(f"Wrote summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
