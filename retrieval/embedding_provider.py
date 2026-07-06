from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


load_dotenv()


class QwenEmbeddingProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("QWEN_API_KEY")
        self.base_url = os.getenv("QWEN_BASE_URL")
        self.model = os.getenv("QWEN_EMBEDDING_MODEL")
        self._warned = False
        self.disabled = False
        self.client: OpenAI | None = None

        if self.api_key and self.base_url and self.model and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)

    def _warn(self, message: str) -> None:
        if not self._warned:
            print(f"Warning: {message}")
            self._warned = True

    def embed(self, text: str) -> Optional[list[float]]:
        if self.disabled:
            return None
        if not self.api_key:
            self._warn("QWEN_API_KEY is missing; dense embedding is disabled.")
            self.disabled = True
            return None
        if not self.base_url or not self.model:
            self._warn("QWEN_BASE_URL or QWEN_EMBEDDING_MODEL is missing; dense embedding is disabled.")
            self.disabled = True
            return None
        if OpenAI is None:
            self._warn("openai package is missing; dense embedding is disabled.")
            self.disabled = True
            return None
        if not self.client:
            self._warn("Qwen embedding client is not initialized.")
            self.disabled = True
            return None

        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            embedding = response.data[0].embedding
            return list(embedding)
        except Exception as exc:  # Keep retrieval usable when the remote API is unavailable.
            self._warn(f"Qwen embedding request failed: {type(exc).__name__}")
            self.disabled = True
            return None
