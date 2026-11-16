"""OpenAI embedding model (multilingual, handles Vietnamese).

The Day 6 Python-docs run used all-MiniLM-L6-v2, which is English-only; this
corpus is Vietnamese, so a multilingual model is required. Queries and
documents are embedded the same way (symmetric, no prefix); vectors from the
API are already L2-normalized, so cosine distance in Chroma is appropriate.
"""
from __future__ import annotations

import os
import time

PRICE_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 128) -> None:
        from openai import OpenAI  # fail loudly if missing

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (put it in starter/.env)")
        self.client = OpenAI()
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimensions = 1536 if model_name.endswith("small") else 3072
        self.max_seq_length = 8191  # tokens
        self.tokens_used = 0

    @property
    def cost_usd(self) -> float:
        return round(self.tokens_used / 1e6 * PRICE_PER_1M.get(self.model_name, 0.0), 6)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(4):
            try:
                resp = self.client.embeddings.create(model=self.model_name, input=texts)
                self.tokens_used += resp.usage.total_tokens
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except Exception:  # transient API errors: retry, then re-raise
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]
