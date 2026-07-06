#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.chroma_retriever import COLLECTION_NAME, DEFAULT_CHROMA_PATH  # noqa: E402
from retrieval.embedding_provider import QwenEmbeddingProvider  # noqa: E402


CORPUS_PATH = ROOT / "web" / "data" / "rag_corpus.json"


def load_items() -> list[dict[str, Any]]:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    return [item for item in items if isinstance(item, dict)]


def clean_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "paper_id": str(item.get("paper_id") or metadata.get("paper_id") or ""),
        "title": str(item.get("title") or metadata.get("title") or ""),
        "final_score": float(item.get("final_score") or metadata.get("final_score") or 0),
        "publication_year": int(item.get("publication_year") or metadata.get("publication_year") or 0),
        "source_url": str(item.get("source_url") or metadata.get("source_url") or ""),
        "pdf_url": str(item.get("pdf_url") or metadata.get("pdf_url") or ""),
    }


def main() -> None:
    try:
        import chromadb
    except Exception as exc:
        print(f"ChromaDB index build skipped: chromadb unavailable ({type(exc).__name__}).")
        print("indexed_count: 0")
        print("skipped_count: 0")
        print("embedding_dimension: 0")
        return

    items = load_items()
    provider = QwenEmbeddingProvider()
    rows: list[tuple[str, list[float], str, dict[str, Any]]] = []
    skipped_count = 0
    embedding_dimension = 0

    for item in items:
        paper_id = str(item.get("paper_id") or "").strip()
        document = str(item.get("text_for_embedding") or "").strip()
        if not paper_id or not document:
            skipped_count += 1
            continue

        embedding = provider.embed(document)
        if embedding is None:
            skipped_count += 1
            continue
        if not embedding_dimension:
            embedding_dimension = len(embedding)
        rows.append((paper_id, embedding, document, clean_metadata(item)))

    if not rows:
        print("ChromaDB index build skipped: Qwen embedding unavailable or no valid corpus items.")
        print("indexed_count: 0")
        print(f"skipped_count: {skipped_count}")
        print("embedding_dimension: 0")
        return

    DEFAULT_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    batch_size = 64
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.add(
            ids=[row[0] for row in batch],
            embeddings=[row[1] for row in batch],
            documents=[row[2] for row in batch],
            metadatas=[row[3] for row in batch],
        )

    print(f"indexed_count: {len(rows)}")
    print(f"skipped_count: {skipped_count}")
    print(f"embedding_dimension: {embedding_dimension}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"chroma_path: {DEFAULT_CHROMA_PATH}")


if __name__ == "__main__":
    main()
