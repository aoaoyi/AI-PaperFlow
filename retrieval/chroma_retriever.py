from __future__ import annotations

from pathlib import Path
from typing import Any

from .embedding_provider import QwenEmbeddingProvider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_PATH = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "paperflow_corpus"


class ChromaRetriever:
    def __init__(self, chroma_path: Path = DEFAULT_CHROMA_PATH, collection_name: str = COLLECTION_NAME) -> None:
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.provider = QwenEmbeddingProvider()
        self.collection = None

        try:
            import chromadb

            if chroma_path.exists():
                client = chromadb.PersistentClient(path=str(chroma_path))
                self.collection = client.get_collection(collection_name)
        except Exception as exc:
            print(f"Warning: ChromaDB collection unavailable: {type(exc).__name__}")
            self.collection = None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.collection or not query.strip():
            return []

        query_embedding = self.provider.embed(query)
        if query_embedding is None:
            return []

        try:
            response = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        except Exception as exc:
            print(f"Warning: ChromaDB query failed: {type(exc).__name__}")
            return []

        ids = response.get("ids", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        results: list[dict[str, Any]] = []
        for paper_id, metadata, distance in zip(ids, metadatas, distances):
            distance_value = float(distance or 0)
            results.append(
                {
                    "paper_id": paper_id,
                    "title": metadata.get("title", "") if isinstance(metadata, dict) else "",
                    "final_score": float(metadata.get("final_score") or 0) if isinstance(metadata, dict) else 0.0,
                    "dense_score": round(1 / (1 + distance_value), 4),
                    "distance": round(distance_value, 4),
                    "source_url": metadata.get("source_url", "") if isinstance(metadata, dict) else "",
                    "pdf_url": metadata.get("pdf_url", "") if isinstance(metadata, dict) else "",
                }
            )
        return results

