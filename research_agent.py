from __future__ import annotations

import json
import re
from typing import Any

from evaluation.citation_verifier import verify_citations
from llm.gemini_provider import GeminiProvider
from research_prompts import DEFAULT_SUB_QUESTIONS, planner_prompt, writer_prompt
from research_utils import build_evidence_pack, dedupe_papers_by_title
from retrieval.hybrid_retriever import HybridRetriever


def parse_json_list(text: str) -> list[str]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    lines = [re.sub(r"^[\-\d\.\)、\s]+", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line][:5]


def plan_sub_questions(topic: str, provider: GeminiProvider) -> tuple[list[str], bool, str | None]:
    text, used, error = provider.generate_text(planner_prompt(topic))
    if used:
        questions = parse_json_list(text)
        if questions:
            return questions[:3], True, None
    fallback = [f"{topic}: {question}" for question in DEFAULT_SUB_QUESTIONS[:5]]
    return fallback[:3], False, error or "Planner fallback used"


def retrieve_for_question(retriever: HybridRetriever, question: str, top_k: int) -> dict[str, Any]:
    response = retriever.search(question, top_k=top_k, use_rerank=True, candidate_k=30)
    return response


def collect_retrieved_papers(sub_questions: list[str], top_k: int) -> tuple[list[dict[str, Any]], list[str], list[str | None], int]:
    papers: list[dict[str, Any]] = []
    modes: list[str] = []
    fallback_reasons: list[str | None] = []
    candidate_count = 0
    retriever = HybridRetriever()
    for question in sub_questions:
        response = retrieve_for_question(retriever, question, top_k=top_k)
        papers.extend(response["results"])
        fallback_reasons.append(response.get("fallback_reason"))
        candidate_count += int(response.get("candidate_count") or 0)
        mode = "hybrid_rerank" if response.get("rerank_used") else "hybrid"
        modes.append(mode)
    return dedupe_papers_by_title(papers), modes, fallback_reasons, candidate_count


def fallback_report(topic: str, sub_questions: list[str], evidence_pack: list[dict[str, Any]]) -> str:
    lines = [
        "# Research Background",
        f"本报告围绕“{topic}”生成，当前使用检索结果 fallback，不调用 Gemini 写作。",
        "",
        "# Key Findings",
    ]
    for item in evidence_pack[:5]:
        lines.append(f"- {item.get('title', '')}: {item.get('summary', '')}")
    lines.extend(["", "# Limitations", "根据当前语料，证据可能不完整，需要继续阅读原文确认。", "", "# References"])
    lines.extend(f"- {item.get('title', '')}" for item in evidence_pack if item.get("title"))
    return "\n".join(lines)


def write_report(
    topic: str,
    sub_questions: list[str],
    evidence_pack: list[dict[str, Any]],
    provider: GeminiProvider,
) -> tuple[str, bool, str | None]:
    text, used, error = provider.generate_text(writer_prompt(topic, sub_questions, evidence_pack))
    if used and text:
        return text, True, None
    return fallback_report(topic, sub_questions, evidence_pack), False, error or "Writer fallback used"


def generate_research_report(topic: str, top_k: int = 5, language: str = "zh") -> dict[str, Any]:
    provider = GeminiProvider()
    sub_questions, planner_used, planner_error = plan_sub_questions(topic, provider)
    retrieved_papers, modes, retrieval_errors, candidate_count = collect_retrieved_papers(sub_questions, top_k=top_k)
    evidence_pack = build_evidence_pack(retrieved_papers, max_items=12)
    report, writer_used, writer_error = write_report(topic, sub_questions, evidence_pack, provider)
    verification = verify_citations(report, retrieved_papers)
    references = [paper.get("title", "") for paper in retrieved_papers if paper.get("title")]
    fallback_reason = "; ".join(str(reason) for reason in [planner_error, writer_error, *retrieval_errors] if reason)
    return {
        "topic": topic,
        "sub_questions": sub_questions,
        "retrieved_papers": retrieved_papers,
        "evidence_pack": evidence_pack,
        "report": report,
        "references": references,
        "verification": verification,
        "llm_used": bool(planner_used or writer_used),
        "fallback_reason": fallback_reason or None,
        "retrieval_mode": "hybrid_rerank" if "hybrid_rerank" in modes else "hybrid",
        "candidate_count": candidate_count,
        "language": language,
    }
