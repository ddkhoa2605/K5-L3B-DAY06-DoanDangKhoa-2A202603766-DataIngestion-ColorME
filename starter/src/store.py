"""ChromaDB persistent store: one collection per strategy, cosine HNSW index."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

ALLOWED_FILTERS = {"chapter", "content_type", "page", "document_id", "visual_kind"}
COLLECTION_PREFIX = "colorme3_"


def build_where(filters: dict[str, Any] | None):
    if not filters:
        return None
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"filter fields not allowed: {bad}")
    items = []
    for k, v in filters.items():
        items.append({k: {"$in": list(v)}} if isinstance(v, (list, tuple)) else {k: {"$eq": v}})
    return items[0] if len(items) == 1 else {"$and": items}


class ChromaStore:
    def __init__(self, path: Path, dimensions: int, model_name: str) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(self.path))

    def collection(self, strategy: str):
        name = f"{COLLECTION_PREFIX}{strategy}"
        col = self.client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.model_name, "dimensions": self.dimensions})
        meta = col.metadata or {}
        if meta.get("embedding_model") not in (None, self.model_name):
            raise RuntimeError(f"{name} was built with {meta.get('embedding_model')}; "
                               f"refusing to mix with {self.model_name}")
        return col

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        """Idempotent write: stable IDs are upserted; IDs of the same documents that
        no longer exist (source changed) are deleted. Returns before/after counts."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension mismatch")
        col = self.collection(strategy)
        before = col.count()
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids inside one batch")
        new_ids = set(ids)
        stale: list[str] = []
        for doc_id in {c.document_id for c in chunks}:
            existing = col.get(where={"document_id": doc_id}, include=[])["ids"]
            stale.extend(i for i in existing if i not in new_ids)
        if stale:
            col.delete(ids=stale)
        metas = []
        for c in chunks:
            m = {"document_id": c.document_id, "source": c.source, "page": c.page, "strategy": strategy}
            m.update({k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))})
            metas.append(m)
        for i in range(0, len(ids), 1000):
            col.upsert(ids=ids[i:i + 1000], embeddings=vectors[i:i + 1000],
                       documents=[c.text for c in chunks[i:i + 1000]], metadatas=metas[i:i + 1000])
        return {"before": before, "after": col.count(), "stale_deleted": len(stale)}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        res = self.collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=build_where(filters),
            include=["documents", "metadatas", "distances"])
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            out.append({"rank": rank, "chunk_id": cid, "score": round(1 - dist, 4),
                        "text": doc, "metadata": meta})
        return out
