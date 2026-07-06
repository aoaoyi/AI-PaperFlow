#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_api import app


def main() -> None:
    client = TestClient(app)
    health = client.get("/health")
    print(f"health_status: {health.status_code}")
    response = client.post(
        "/ask",
        json={
            "question": "What papers discuss hallucination in retrieval augmented generation?",
            "top_k": 5,
            "use_rerank": True,
        },
    )
    print(f"ask_status: {response.status_code}")
    data = response.json()
    print(f"retrieved_count: {len(data.get('retrieved_papers', []))}")
    print(f"rerank_used: {data.get('rerank_used')}")
    print(f"llm_used: {data.get('llm_used')}")
    print(f"fallback_reason: {data.get('fallback_reason')}")
    for index, paper in enumerate(data.get("retrieved_papers", [])[:5], start=1):
        print(f"{index}. {paper.get('title', '')}")
    if response.status_code != 200 or not data.get("retrieved_papers"):
        raise SystemExit("ask endpoint test failed")


if __name__ == "__main__":
    main()
