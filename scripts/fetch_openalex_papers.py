#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # The script still works with process environment variables.
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "research_topics.json"
RAW_OUTPUT_PATH = ROOT / "data" / "raw_openalex_works.json"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_PER_QUERY = 20
DEFAULT_FROM_DATE = "2023-01-01"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def openalex_request(query: str, per_query: int, api_key: str) -> dict[str, Any]:
    params = {
        "search": query,
        "per-page": str(per_query),
        "filter": f"from_publication_date:{DEFAULT_FROM_DATE}",
        "sort": "relevance_score:desc",
    }
    if api_key:
        params["api_key"] = api_key
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.getenv("OPENALEX_EMAIL", "")
    url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ai-paperflow-openalex-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all() -> dict[str, Any]:
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    api_key = os.getenv("OPENALEX_API_KEY", "")
    if not api_key:
        print("Warning: OPENALEX_API_KEY missing; continuing without api_key.", file=sys.stderr)

    topics = load_json(CONFIG_PATH)
    per_query = int(os.getenv("OPENALEX_PER_QUERY", str(DEFAULT_PER_QUERY)))
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for topic in topics:
        for query in topic.get("queries", []):
            try:
                payload = openalex_request(str(query), per_query, api_key)
                for work in payload.get("results", [])[:per_query]:
                    entries.append(
                        {
                            "topic_id": topic.get("topic_id", ""),
                            "topic_name": topic.get("name", ""),
                            "topic_keywords": topic.get("keywords", []),
                            "query": query,
                            "work": work,
                        }
                    )
                print(f"OpenAlex query ok: topic={topic.get('topic_id')} query={query} count={len(payload.get('results', []))}")
            except Exception as exc:
                errors.append({"topic_id": str(topic.get("topic_id", "")), "query": str(query), "error_type": type(exc).__name__})
                print(f"OpenAlex query failed: topic={topic.get('topic_id')} query={query} error={type(exc).__name__}", file=sys.stderr)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "OpenAlex",
        "from_publication_date": DEFAULT_FROM_DATE,
        "per_query": per_query,
        "total_raw_count": len(entries),
        "entries": entries,
        "errors": errors,
    }


def main() -> None:
    payload = fetch_all()
    write_json(RAW_OUTPUT_PATH, payload)
    print(f"Wrote raw OpenAlex works: count={payload['total_raw_count']} path={RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
