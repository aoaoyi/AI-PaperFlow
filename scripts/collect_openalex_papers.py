#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/research_topics.json"
OUTPUT_PATH = ROOT / "web/data/papers.json"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def abstract_text(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    words = []
    for word, positions in inverted.items():
        for position in positions if isinstance(positions, list) else []:
            if isinstance(position, int):
                words.append((position, word))
    return " ".join(word for _, word in sorted(words))


def request_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-paperflow-openalex/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_topic(topic: dict[str, Any], per_page: int) -> list[dict[str, Any]]:
    query = " ".join(topic.get("keywords", [])) or topic.get("name", "")
    params = {"search": query, "per-page": str(per_page), "sort": "cited_by_count:desc"}
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.getenv("OPENALEX_EMAIL")
    data = request_json(f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}")
    return [normalize_work(work, topic) for work in data.get("results", [])]


def normalize_work(work: dict[str, Any], topic: dict[str, Any]) -> dict[str, Any]:
    primary = work.get("primary_location") or {}
    best_oa = work.get("best_oa_location") or {}
    authors = [
        str((item.get("author") or {}).get("display_name") or "")
        for item in work.get("authorships", [])
    ]
    concepts = [str(item.get("display_name") or "") for item in work.get("concepts", [])[:8]]
    abstract = abstract_text(work)
    doi = str(work.get("doi") or "")
    source_url = doi or str(work.get("id") or "")
    pdf_url = str(primary.get("pdf_url") or best_oa.get("pdf_url") or "")
    open_access = bool((work.get("open_access") or {}).get("is_oa") or pdf_url)
    paper = {
        "openalex_id": str(work.get("id") or ""),
        "doi": doi,
        "id": str(work.get("id") or doi or work.get("title") or ""),
        "title": str(work.get("title") or ""),
        "authors": [author for author in authors if author],
        "abstract": abstract,
        "summary": abstract,
        "publication_year": int(work.get("publication_year") or 0),
        "published": f"{int(work.get('publication_year') or 0)}-01-01T00:00:00+00:00" if work.get("publication_year") else "",
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "topics": [topic.get("name", ""), *[concept for concept in concepts if concept]],
        "source_url": source_url,
        "paper_url": source_url,
        "pdf_url": pdf_url,
        "open_access": open_access,
        "categories": [topic.get("name", ""), *concepts],
        "source": "OpenAlex",
    }
    paper["final_score"] = final_score(paper, topic)
    paper["structured_summary"] = fallback_summary(paper)
    return paper


def final_score(paper: dict[str, Any], topic: dict[str, Any]) -> float:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    keywords = [str(item).lower() for item in topic.get("keywords", [])]
    keyword_hits = sum(1 for keyword in keywords if keyword and keyword in text)
    keyword_score = min(1.0, keyword_hits / max(1, len(keywords)))
    topic_score = 1.0 if topic.get("name", "").lower() in " ".join(paper.get("topics", [])).lower() else 0.5
    current_year = dt.datetime.now(dt.timezone.utc).year
    freshness = max(0.0, 1.0 - max(0, current_year - int(paper.get("publication_year") or current_year)) / 5)
    citation = min(1.0, float(paper.get("cited_by_count") or 0) / 500)
    access = 1.0 if paper.get("pdf_url") or paper.get("open_access") else 0.0
    return round(0.35 * keyword_score + 0.2 * topic_score + 0.2 * freshness + 0.15 * citation + 0.1 * access, 3)


def fallback_summary(paper: dict[str, Any]) -> dict[str, str]:
    abstract = paper.get("abstract", "")
    return {
        "problem": abstract[:300],
        "method": "",
        "innovation": "",
        "evidence": "",
        "limitation": "",
        "why_relevant": "Matched by OpenAlex topic keywords.",
    }


def dedupe(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for paper in papers:
        key = paper.get("openalex_id") or paper.get("doi") or normalize_title(paper.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return sorted(unique, key=lambda item: item.get("final_score", 0.0), reverse=True)


def main() -> None:
    config = load_json(CONFIG_PATH)
    per_topic = int(os.getenv("OPENALEX_PER_TOPIC", "20"))
    papers: list[dict[str, Any]] = []
    for topic in config.get("topics", []):
        try:
            papers.extend(fetch_topic(topic, per_topic))
        except Exception as exc:
            print(f"OpenAlex fetch failed for {topic.get('name')}: {exc}", file=sys.stderr)
    payload = {
        "generated_at_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_kind": "openalex",
        "papers": dedupe(papers),
        "stats": {"paper_count": len(dedupe(papers)), "source": "OpenAlex"},
    }
    write_json(OUTPUT_PATH, payload)
    print(f"Wrote {len(payload['papers'])} OpenAlex papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
