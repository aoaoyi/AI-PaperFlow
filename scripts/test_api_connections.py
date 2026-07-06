#!/usr/bin/env python3
from __future__ import annotations

import os
import urllib.parse

import requests
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


def error_type(exc: Exception) -> str:
    return type(exc).__name__


def test_openalex() -> None:
    try:
        params = {
            "search": "retrieval augmented generation",
            "per-page": "3",
        }
        api_key = os.getenv("OPENALEX_API_KEY")
        if api_key:
            params["api_key"] = api_key
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        response = requests.get(url, timeout=30, headers={"User-Agent": "ai-paperflow-env-check/1.0"})
        response.raise_for_status()
        data = response.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        first_title = results[0].get("title", "") if results else ""
        print("OpenAlex: success")
        print(f"returned_count: {len(results)}")
        print(f"first_title: {first_title}")
    except Exception as exc:
        print("OpenAlex: failed")
        print(f"error type: {error_type(exc)}")


def test_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
    if not api_key:
        print("Gemini: skipped")
        return
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="用一句中文解释什么是 RAG。",
        )
        text = getattr(response, "text", "") or ""
        print("Gemini: success")
        print(f"response: {text[:80]}")
    except Exception as exc:
        print("Gemini: failed")
        print(f"error type: {error_type(exc)}")


def test_qwen_embedding() -> None:
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_EMBEDDING_MODEL")
    if not api_key:
        print("Qwen Embedding: skipped")
        return
    try:
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.embeddings.create(
            model=model,
            input="retrieval augmented generation for research papers",
        )
        embedding = response.data[0].embedding
        print("Qwen Embedding: success")
        print(f"embedding dimension: {len(embedding)}")
    except Exception as exc:
        print("Qwen Embedding: failed")
        print(f"error type: {error_type(exc)}")


def test_cohere_rerank() -> None:
    api_key = os.getenv("COHERE_API_KEY")
    model = os.getenv("COHERE_RERANK_MODEL") or "rerank-english-v3.0"
    if not api_key:
        print("Cohere Rerank: skipped")
        return
    try:
        import cohere

        client = cohere.Client(api_key)
        response = client.rerank(
            model=model,
            query="What paper is about RAG?",
            documents=[
                "This paper studies retrieval augmented generation.",
                "This paper studies image classification.",
            ],
            top_n=1,
        )
        top_index = response.results[0].index
        print("Cohere Rerank: success")
        print(f"top document index: {top_index}")
    except Exception as exc:
        print("Cohere Rerank: failed")
        print(f"error type: {error_type(exc)}")


def main() -> None:
    load_dotenv()
    test_openalex()
    test_gemini()
    test_qwen_embedding()
    test_cohere_rerank()


if __name__ == "__main__":
    main()
