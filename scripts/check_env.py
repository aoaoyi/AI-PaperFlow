#!/usr/bin/env python3
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


ENV_VARS = [
    "OPENALEX_API_KEY",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_EMBEDDING_MODEL",
    "COHERE_API_KEY",
    "COHERE_RERANK_MODEL",
]


def main() -> None:
    load_dotenv()
    for name in ENV_VARS:
        status = "loaded" if os.getenv(name) else "missing"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
