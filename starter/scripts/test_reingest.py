"""Check re-ingest behaviour on a throw-away Chroma directory (not the real index).

1. ingest chapter 7 (CV, PDF p153-162) with strategy B -> N chunks
2. ingest the same version again                        -> still N (no duplicates)
3. simulate a changed source (last page text cut, its figure description removed)
                                                        -> stale chunk IDs deleted, count = new N
Uses the real embedding model. Run from starter/: python scripts/test_reingest.py
"""
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src import vision  # noqa: E402
from src.embeddings import OpenAIEmbedder  # noqa: E402
from src.run import CAPTIONS, MODEL, build_chunks, load_env, load_pages  # noqa: E402
from src.store import ChromaStore  # noqa: E402

load_env()
tmp = Path(tempfile.mkdtemp(prefix="chroma_reingest_"))
try:
    emb = OpenAIEmbedder(MODEL)
    store = ChromaStore(tmp, emb.dimensions, emb.model_name)
    caps = vision.load_cache(CAPTIONS)
    pages = [p for p in load_pages() if p.metadata["chapter"] == "ch7"]
    chunks = build_chunks("B", pages, caps)
    vecs = emb.embed_documents([c.text for c in chunks])
    r1 = store.upsert("B", chunks, vecs)
    r2 = store.upsert("B", chunks, vecs)
    last = pages[-1]
    changed = pages[:-1] + [replace(last, text=last.text[: len(last.text) // 3])]
    caps2 = {k: v for k, v in caps.items() if k != last.page}
    chunks2 = build_chunks("B", changed, caps2)
    r3 = store.upsert("B", chunks2, emb.embed_documents([c.text for c in chunks2]))
    print(f"first ingest : {r1}  (chunks={len(chunks)})")
    print(f"same version : {r2}  duplicates={'NO' if r2['after'] == len(chunks) else 'YES'}")
    print(f"changed doc  : {r3}  (new chunks={len(chunks2)}) "
          f"stale removed={'OK' if r3['after'] == len(chunks2) else 'FAIL'}")
    print(f"embedding tokens used: {emb.tokens_used} (${emb.cost_usd})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
