"""
AI-PaperFlow FastAPI backend.

Start with:
    uvicorn rag_api:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm.gemini_provider import GeminiProvider
from research_agent import generate_research_report
from retrieval.hybrid_retriever import HybridRetriever


app = FastAPI(title="AI-PaperFlow API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_rerank: bool = True


class ResearchRequest(BaseModel):
    topic: str
    top_k: int = Field(default=5, ge=1, le=10)


def ranked_papers(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    papers = []
    for rank, paper in enumerate(results, start=1):
        papers.append(
            {
                "rank": rank,
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "rerank_score": paper.get("rerank_score"),
                "hybrid_score": paper.get("hybrid_score", 0),
                "final_score": paper.get("final_score", 0),
                "retrieval_sources": paper.get("retrieval_sources", []),
                "source_url": paper.get("source_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "matched_topic_names": paper.get("matched_topic_names", []),
            }
        )
    return papers


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        retriever = HybridRetriever()
        corpus_count = len(retriever.corpus_items)
        retrieval_ready = corpus_count > 0
    except Exception as exc:
        corpus_count = 0
        retrieval_ready = False
        return {"status": "error", "retrieval_ready": retrieval_ready, "paper_count": corpus_count, "error": type(exc).__name__}
    return {
        "status": "ok",
        "retrieval_ready": retrieval_ready,
        "paper_count": corpus_count,
        "api": "AI-PaperFlow",
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    try:
        retrieval = HybridRetriever().search(
            question,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            candidate_k=30,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {type(exc).__name__}") from exc

    retrieved_papers = ranked_papers(retrieval["results"])
    answer, llm_used, model, llm_fallback = GeminiProvider().generate_answer(question, retrieved_papers)
    fallback_parts = [retrieval.get("fallback_reason"), llm_fallback]
    fallback_reason = "; ".join(str(part) for part in fallback_parts if part)

    return {
        "question": question,
        "answer": answer,
        "retrieved_papers": retrieved_papers,
        "retrieval_mode": "hybrid_rerank" if retrieval.get("rerank_used") else "hybrid",
        "rerank_used": retrieval.get("rerank_used", False),
        "llm_used": llm_used,
        "model": model,
        "candidate_count": retrieval.get("candidate_count", 0),
        "fallback_reason": fallback_reason or None,
    }


@app.post("/research")
def research(request: ResearchRequest) -> dict[str, Any]:
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty.")
    try:
        return generate_research_report(topic, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Research generation failed: {type(exc).__name__}") from exc
