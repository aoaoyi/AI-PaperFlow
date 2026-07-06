from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
except ImportError:  # pragma: no cover
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None
    connections = None
    utility = None


if load_dotenv:
    load_dotenv()


CORPUS_PATH = Path("web/data/rag_corpus.json")
COLLECTION_NAME = "paperflow_corpus"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _clip(value: Any, length: int) -> str:
    return str(value or "")[:length]


def _authors_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _summary(record: dict[str, Any]) -> str:
    parts = [record.get("problem"), record.get("method"), record.get("innovation")]
    summary = " ".join(str(part).strip() for part in parts if str(part or "").strip())
    return summary or str(record.get("abstract") or "")


def embedding_configured() -> bool:
    return bool(OpenAI is not None and _env("EMBEDDING_API_KEY") and _env("EMBEDDING_MODEL"))


def is_milvus_available() -> bool:
    if connections is None or utility is None:
        return False
    try:
        connections.connect(alias="default", host=_env("MILVUS_HOST", "127.0.0.1"), port=_env("MILVUS_PORT", "19530"))
        utility.list_collections()
        return True
    except Exception:
        return False


def _embedding_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = _env("EMBEDDING_API_KEY")
    model = _env("EMBEDDING_MODEL")
    if not api_key or not model:
        raise RuntimeError("EMBEDDING_API_KEY and EMBEDDING_MODEL must be configured.")
    kwargs: dict[str, str] = {"api_key": api_key}
    if _env("EMBEDDING_BASE_URL"):
        kwargs["base_url"] = _env("EMBEDDING_BASE_URL")
    return OpenAI(**kwargs)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    client = _embedding_client()
    response = client.embeddings.create(model=_env("EMBEDDING_MODEL"), input=texts)
    return [item.embedding for item in response.data]


def _connect_milvus() -> None:
    if connections is None:
        raise RuntimeError("pymilvus package is not installed.")
    connections.connect(alias="default", host=_env("MILVUS_HOST", "127.0.0.1"), port=_env("MILVUS_PORT", "19530"))


def _load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"RAG corpus file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError("RAG corpus is empty or invalid.")
    return [record for record in data if isinstance(record, dict)]


def _create_collection(dim: int) -> Collection:
    if Collection is None or CollectionSchema is None or DataType is None or FieldSchema is None or utility is None:
        raise RuntimeError("pymilvus package is not installed.")
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="final_score", dtype=DataType.FLOAT),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="pdf_url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="text_for_embedding", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    collection = Collection(COLLECTION_NAME, CollectionSchema(fields, description="AI-PaperFlow RAG corpus"))
    collection.create_index("vector", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}})
    return collection


def build_index(corpus_path: Path = CORPUS_PATH) -> dict[str, Any]:
    records = _load_corpus(corpus_path)
    texts = [str(record.get("text_for_embedding") or "") for record in records]
    if not any(text.strip() for text in texts):
        raise RuntimeError("No text_for_embedding content found in corpus.")
    vectors = _embed_texts(texts)
    if not vectors:
        raise RuntimeError("Embedding API returned no vectors.")
    _connect_milvus()
    collection = _create_collection(len(vectors[0]))
    rows = [
        {
            "paper_id": _clip(record.get("paper_id"), 256),
            "title": _clip(record.get("title"), 1024),
            "authors": _clip(_authors_text(record.get("authors")), 2048),
            "final_score": float(record.get("final_score") or 0.0),
            "source_url": _clip(record.get("source_url"), 2048),
            "pdf_url": _clip(record.get("pdf_url"), 2048),
            "summary": _clip(_summary(record), 8192),
            "text_for_embedding": _clip(record.get("text_for_embedding"), 16384),
            "vector": vector,
        }
        for record, vector in zip(records, vectors)
    ]
    collection.insert(rows)
    collection.flush()
    collection.load()
    return {"collection": COLLECTION_NAME, "inserted": len(rows), "dimension": len(vectors[0])}


def search_papers(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not is_milvus_available():
        raise RuntimeError("Milvus is not available.")
    query_vector = _embed_texts([query])[0]
    if utility is None or Collection is None or not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"Milvus collection not found: {COLLECTION_NAME}")
    collection = Collection(COLLECTION_NAME)
    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["paper_id", "title", "authors", "final_score", "source_url", "pdf_url", "summary", "text_for_embedding"],
    )
    papers = []
    for hit in results[0]:
        entity = hit.entity
        similarity = float(hit.score)
        papers.append(
            {
                "paper_id": entity.get("paper_id"),
                "title": entity.get("title"),
                "authors": entity.get("authors"),
                "final_score": float(entity.get("final_score") or 0.0),
                "source_url": entity.get("source_url"),
                "pdf_url": entity.get("pdf_url"),
                "summary": entity.get("summary"),
                "text_for_embedding": entity.get("text_for_embedding"),
                "similarity": round(similarity, 4),
                "distance": round(1.0 - similarity, 4),
            }
        )
    return papers
