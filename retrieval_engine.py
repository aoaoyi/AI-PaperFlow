from __future__ import annotations

import math
import os
import re
from typing import Any

from research_utils import CORPUS_PATH, float_value, format_paper, load_corpus

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import cohere
except ImportError:  # pragma: no cover
    cohere = None


CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "paperflow_corpus"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def bm25_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    records = load_corpus(CORPUS_PATH)
    texts = [str(record.get("text_for_embedding") or "") for record in records]
    if BM25Okapi is None:
        return lexical_search(query, records, top_k)
    tokenized = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(records)), key=lambda index: scores[index], reverse=True)[:top_k]
    max_score = max(scores) if len(scores) else 0.0
    return [format_paper(records[index], similarity=float(scores[index] / max_score) if max_score else 0.0) for index in order]


def lexical_search(query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_terms = set(tokenize(query))
    scored = []
    for record in records:
        terms = set(tokenize(str(record.get("text_for_embedding") or "")))
        score = len(query_terms & terms) / math.sqrt(len(query_terms) * len(terms)) if query_terms and terms else 0.0
        scored.append((score, record))
    scored.sort(key=lambda item: (item[0], float_value(item[1].get("final_score"))), reverse=True)
    return [format_paper(record, similarity=score) for score, record in scored[:top_k]]


def embedding_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("EMBEDDING_API_KEY and EMBEDDING_MODEL are required.")
    kwargs: dict[str, str] = {"api_key": api_key}
    if os.getenv("EMBEDDING_BASE_URL", "").strip():
        kwargs["base_url"] = os.getenv("EMBEDDING_BASE_URL", "").strip()
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = embedding_client()
    response = client.embeddings.create(model=os.getenv("EMBEDDING_MODEL", "").strip(), input=texts)
    return [item.embedding for item in response.data]


def chroma_collection():
    if chromadb is None:
        raise RuntimeError("chromadb package is not installed.")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(CHROMA_COLLECTION)


def ensure_chroma_index() -> int:
    records = load_corpus(CORPUS_PATH)
    collection = chroma_collection()
    if collection.count() >= len(records):
        return collection.count()
    ids = [str(record.get("paper_id") or index) for index, record in enumerate(records)]
    docs = [str(record.get("text_for_embedding") or "") for record in records]
    vectors = embed_texts(docs)
    metadatas = [
        {
            "paper_id": str(record.get("paper_id") or ""),
            "title": str(record.get("title") or ""),
            "authors": ", ".join(record.get("authors") or []) if isinstance(record.get("authors"), list) else str(record.get("authors") or ""),
            "final_score": float_value(record.get("final_score")),
            "source_url": str(record.get("source_url") or ""),
            "pdf_url": str(record.get("pdf_url") or ""),
            "summary": str(record.get("problem") or record.get("abstract") or ""),
        }
        for record in records
    ]
    collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas)
    return collection.count()


def chroma_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    ensure_chroma_index()
    collection = chroma_collection()
    query_vector = embed_texts([query])[0]
    result = collection.query(query_embeddings=[query_vector], n_results=top_k)
    papers = []
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for metadata, distance in zip(metadatas, distances):
        papers.append(
            {
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", ""),
                "final_score": float_value(metadata.get("final_score")),
                "source_url": metadata.get("source_url", ""),
                "pdf_url": metadata.get("pdf_url", ""),
                "summary": metadata.get("summary", ""),
                "distance": round(float_value(distance), 4),
                "similarity": round(1.0 / (1.0 + float_value(distance)), 4),
            }
        )
    return papers


def merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for paper in group:
            key = str(paper.get("source_url") or paper.get("title") or "").strip().lower()
            if not key:
                continue
            existing = merged.get(key)
            if not existing or float_value(paper.get("similarity")) > float_value(existing.get("similarity")):
                merged[key] = paper
    return list(merged.values())


def cohere_rerank(query: str, papers: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], bool, str]:
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if cohere is None or not api_key:
        return simple_fusion(papers, top_k), False, "Cohere unavailable."
    try:
        client = cohere.Client(api_key)
        docs = [f"{paper.get('title', '')}\n{paper.get('summary', '')}" for paper in papers]
        response = client.rerank(query=query, documents=docs, top_n=min(top_k, len(docs)))
        reranked = []
        for item in response.results:
            paper = dict(papers[item.index])
            paper["rerank_score"] = round(float(item.relevance_score), 4)
            reranked.append(paper)
        return reranked, True, ""
    except Exception as exc:
        return simple_fusion(papers, top_k), False, f"Cohere rerank failed: {exc}"


def simple_fusion(papers: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(
        papers,
        key=lambda paper: (float_value(paper.get("similarity")), float_value(paper.get("final_score"))),
        reverse=True,
    )[:top_k]


def hybrid_search(query: str, top_k: int = 5) -> dict[str, Any]:
    fallback_reasons = []
    bm25 = bm25_search(query, top_k=top_k)
    dense: list[dict[str, Any]] = []
    dense_used = False
    try:
        dense = chroma_search(query, top_k=top_k)
        dense_used = True
    except Exception as exc:
        fallback_reasons.append(f"Dense retrieval unavailable: {exc}")
    candidates = merge_candidates(bm25, dense) if dense_used else bm25
    reranked, rerank_used, rerank_reason = cohere_rerank(query, candidates, top_k)
    if rerank_reason:
        fallback_reasons.append(rerank_reason)
    return {
        "papers": reranked,
        "retrieval_mode": "hybrid" if dense_used else "bm25_fallback",
        "rerank_used": rerank_used,
        "fallback_reason": "; ".join(fallback_reasons),
    }
