from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from google import genai
except ImportError:
    genai = None


load_dotenv()


def retrieval_only_answer(question: str, retrieved_papers: list[dict[str, Any]]) -> str:
    if not retrieved_papers:
        return "当前语料中没有检索到足够相关的论文。"
    lines = [
        "当前回答基于检索结果生成，未调用 Gemini。",
        f"问题：{question}",
        "",
        "最相关论文：",
    ]
    for index, paper in enumerate(retrieved_papers[:5], start=1):
        title = paper.get("title", "")
        score = paper.get("rerank_score")
        score_text = f"，rerank_score={score}" if score is not None else ""
        lines.append(f"{index}. {title}{score_text}")
    lines.append("")
    lines.append("局限性：这是 retrieval-only summary，需要结合论文全文进一步确认。")
    return "\n".join(lines)


def build_answer_prompt(question: str, retrieved_papers: list[dict[str, Any]]) -> str:
    blocks = []
    for index, paper in enumerate(retrieved_papers, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Paper {index}",
                    f"Title: {paper.get('title', '')}",
                    f"Final score: {paper.get('final_score', 0)}",
                    f"Rerank score: {paper.get('rerank_score')}",
                    f"Hybrid score: {paper.get('hybrid_score', 0)}",
                    f"Topics: {', '.join(paper.get('matched_topic_names', []) or [])}",
                    f"Source URL: {paper.get('source_url', '')}",
                    f"PDF URL: {paper.get('pdf_url', '')}",
                ]
            )
        )
    return f"""
你是 AI-PaperFlow 的论文问答助手。请只基于 retrieved_papers 回答，不能编造论文标题、结论或引用。

回答必须使用中文，并包含三个部分：
1. 简要回答
2. 相关论文依据
3. 局限性

如果证据不足，请明确说明“根据当前语料，证据不足”。

用户问题：
{question}

retrieved_papers:
{chr(10).join(blocks)}
""".strip()


class GeminiProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
        self.client = genai.Client(api_key=self.api_key) if genai and self.api_key else None

    def generate_answer(self, question: str, retrieved_papers: list[dict[str, Any]]) -> tuple[str, bool, str, str | None]:
        if not self.api_key:
            return retrieval_only_answer(question, retrieved_papers), False, self.model, "GEMINI_API_KEY missing"
        if genai is None or self.client is None:
            return retrieval_only_answer(question, retrieved_papers), False, self.model, "google-genai package missing"
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=build_answer_prompt(question, retrieved_papers),
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return retrieval_only_answer(question, retrieved_papers), False, self.model, "Gemini returned empty response"
            return text, True, self.model, None
        except Exception as exc:
            return retrieval_only_answer(question, retrieved_papers), False, self.model, type(exc).__name__

    def generate_text(self, prompt: str) -> tuple[str, bool, str | None]:
        if not self.api_key:
            return "", False, "GEMINI_API_KEY missing"
        if genai is None or self.client is None:
            return "", False, "google-genai package missing"
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return "", False, "Gemini returned empty response"
            return text, True, None
        except Exception as exc:
            return "", False, type(exc).__name__
