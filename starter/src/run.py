"""Day 06 pipeline: colorME ebook PDF -> pages (+ vision descriptions) -> A/B/C chunks
-> OpenAI text-embedding-3-small -> Chroma.

Run from ``starter/``::

    python -m src.run prepare    # read PDF, normalize, render pages, manifest
    python -m src.run caption    # vision model describes each page (cached, needs OPENAI_API_KEY)
    python -m src.run validate   # locate every evidence quote -> gold pages (pre-run)
    python -m src.run ingest     # chunk + embed + upsert all strategies (twice: reload check)
    python -m src.run evaluate   # top-3 traces, Hit@1 / Evidence Hit@3
    python -m src.run all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import chunking, ingestion, vision
from .evaluation import evaluate_query, norm, summarize
from .models import DocumentPage

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "raw" / ingestion.PDF_NAME
RENDER_DIR = ROOT / "data" / "page_images"
CAPTIONS = ROOT / "data" / "visual" / "captions.jsonl"
PAGES_FILE = ROOT / "data" / "pages.jsonl"
MANIFEST = ROOT / "data" / "manifest.json"
BENCH = ROOT / "benchmark" / "questions.json"
RESULTS = ROOT / "results"
CHROMA_DIR = ROOT / "starter" / "chroma_db"
MODEL = "text-embedding-3-small"
SOURCE_URL = "https://thietbigas.com.vn/upload/product/EBOOK%20COLORME%203.pdf"
ORIGIN_NOTE = (
    "Bản sao do bên thứ ba lưu trữ (thietbigas.com.vn), KHÔNG phải trang của colorME; file tải về có SHA-256 trùng raw_sha256 (kiểm tra 2026-11-16, 14 641 014 byte). Việc được phép phân phối lại vẫn chưa được xác minh.")
OFFICIAL_SITES = ['https://colorme.vn', 'http://facebook.com/colorME.hanoi', 'mailto:colorME.idea@gmail.com']  # các địa chỉ colorME in trong chính cuốn sách (p183)
LICENSE_NOTE = (
    "Bản quyền thuộc colorME/tác giả. Giấy phép phân phối lại CHƯA được xác minh nên repo không chứa PDF, ảnh render, văn bản toàn sách hay mô tả hình đầy đủ (xem DATA_NOT_INCLUDED.md). Chỉ dùng cho mục đích học tập.")

STRATEGIES = {
    "A": {"content": "text only", "chunking": "page paragraphs <= 800 chars", "filter": False},
    "B": {"content": "text + vision descriptions", "chunking": "page paragraphs <= 800 chars + 1 chunk/visual",
          "filter": False},
    "C": {"content": "text + vision descriptions", "chunking": "same as B", "filter": True},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / "starter" / ".env")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_counted(p: DocumentPage) -> bool:
    """Content page for the >=30 page rule: not cover/TOC, has text or a content image."""
    return p.page not in (1, 4) and (p.metadata["text_chars"] >= 50 or p.metadata["content_images"] > 0)


# ---------------------------------------------------------------- prepare
def prepare() -> list[DocumentPage]:
    if not PDF.exists():
        sys.exit(f"missing {PDF} – copy the ebook there (see README)")
    t0 = time.perf_counter()
    pages, errors, info = ingestion.load_pdf(PDF, RENDER_DIR)
    with PAGES_FILE.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps({"document_id": p.document_id, "source": p.source, "page": p.page,
                                 "metadata": p.metadata, "text": p.text}, ensure_ascii=False) + "\n")
    counted = [p.page for p in pages if is_counted(p)]
    low_text = [p.page for p in pages if p.metadata["text_chars"] < 50]
    chapters: dict[str, list[int]] = {}
    for p in pages:
        chapters.setdefault(p.metadata["chapter"], []).append(p.page)
    write_json(MANIFEST, {
        "corpus": "EBOOK COLORME 3 – sách hướng dẫn thiết kế đồ họa/Photoshop (tiếng Việt), colorME",
        "documents": [{
            "source_id": ingestion.DOCUMENT_ID, "title": "EBOOK COLORME 3", "language": "vi",
            "source_path_or_url": f"data/raw/{ingestion.PDF_NAME}", "origin": SOURCE_URL,
            "origin_note": ORIGIN_NOTE, "official_sites": OFFICIAL_SITES,
            "license_or_permission": LICENSE_NOTE,
            "version": f"PDF 1.5, tạo bằng Microsoft Word 2016, creationDate {info['pdf_metadata'].get('creationDate')}",
            "accessed_at": datetime.now(timezone.utc).date().isoformat(),
            "raw_sha256": info["sha256"], "raw_bytes": info["bytes"],
            "physical_pages": info["page_count"], "page_size_pt": info["page_size_pt"]}],
        "page_rule": "1 đơn vị = 1 trang PDF vật lý (page = chỉ số PDF 1-based; số in ở chân trang = page-1). "
                     "Trang được tính là nội dung: không phải bìa (p1) / mục lục (p4) và có >=50 ký tự text "
                     "hoặc >=1 ảnh nội dung (ảnh không thuộc template lặp >=10 trang).",
        "normalization": "PyMuPDF get_text(sort=True); Unicode NFC; CRLF->LF; NBSP/tab->space; gộp khoảng trắng "
                         "trong dòng; gộp >=3 dòng trống; bỏ dòng số trang ở chân trang.",
        "visual_processing": f"Render mỗi trang PNG {ingestion.RENDER_DPI} dpi (data/page_images/, không commit); "
                             f"model vision {vision.DEFAULT_MODEL} mô tả ảnh/bảng/biểu đồ -> data/visual/captions.jsonl",
        "total_pages": len(pages), "counted_content_pages": len(counted),
        "not_counted_pages": [p.page for p in pages if not is_counted(p)],
        "pages_with_little_text_lt50": low_text,
        "total_text_chars": sum(len(p.text) for p in pages),
        "chapters": {k: {"title": ingestion.CHAPTER_TITLES[k], "pages": [v[0], v[-1]], "count": len(v)}
                     for k, v in chapters.items()},
        "template_image_xrefs": info["template_xrefs"], "errors": errors,
        "pages": [{"page": p.page, "printed_page": p.metadata["printed_page"], "page_id": p.metadata["page_id"],
                   "chapter": p.metadata["chapter"], "page_title": p.metadata["page_title"],
                   "chars": p.metadata["text_chars"], "content_images": p.metadata["content_images"],
                   "counted": is_counted(p)} for p in pages],
    })
    write_json(RESULTS / "ingest_errors.json", {"errors": errors, "pages_with_little_text_lt50": low_text})
    log(f"prepare: {len(pages)} pages ({len(counted)} counted), {sum(len(p.text) for p in pages)} chars, "
        f"{len(low_text)} pages <50 chars text, errors={len(errors)}, {time.perf_counter() - t0:.1f}s")
    return pages


def load_pages() -> list[DocumentPage]:
    pages = []
    with PAGES_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            pages.append(DocumentPage(d["document_id"], d["source"], d["page"], d["text"], d["metadata"]))
    return pages


def caption() -> dict:
    pages = load_pages()
    t0 = time.perf_counter()
    stats = vision.caption_pages(pages, RENDER_DIR, CAPTIONS, log=log)
    stats["time_s"] = round(time.perf_counter() - t0, 1)
    stats["model"] = vision.DEFAULT_MODEL
    stats["prompt_version"] = vision.PROMPT_VERSION
    write_json(RESULTS / "vision_stats.json", stats)
    log(f"caption: {stats}")
    if stats["errors"]:
        sys.exit("some pages failed; re-run `caption` to retry them")
    return stats


# ---------------------------------------------------------------- chunks
def build_chunks(strategy: str, pages: list[DocumentPage], captions: dict | None = None):
    if captions is None:
        captions = vision.load_cache(CAPTIONS)
    chunks = []
    for p in pages:
        chunks.extend(chunking.text_chunks(p, strategy))
        if strategy in ("B", "C", "B2"):
            chunks.extend(chunking.visual_chunks(p, captions.get(p.page), strategy))
    return chunks


# ---------------------------------------------------------------- B2 (Day 7 improvement loop)
CAPTIONS_V3 = ROOT / "data" / "visual" / "captions_v3_gpt41.jsonl"
V3_MODEL = "gpt-4.1"


def caption_v3() -> dict:
    """Second vision pass (prompt v3 = + counting/exact-reading rules, gpt-4.1) on every page
    with >= 1 content image. Chosen on the dev vision audit (scripts/vision_audit.py:
    v2-mini 9/10, v3-mini 9/10, v3-gpt-4.1 10/10). Pages without content images keep v2."""
    pages = [p for p in load_pages() if p.metadata["content_images"] > 0]
    t0 = time.perf_counter()
    stats = vision.caption_pages(pages, RENDER_DIR, CAPTIONS_V3, model=V3_MODEL, prompt="v3", log=log)
    stats.update({"time_s": round(time.perf_counter() - t0, 1), "model": V3_MODEL, "prompt_version": "v3",
                  "pages_targeted": len(pages)})
    write_json(RESULTS / "vision_v3_stats.json", stats)
    log(f"caption_v3: {stats}")
    if stats["errors"]:
        sys.exit("some pages failed; re-run `caption_v3` to retry them")
    return stats


CAPTIONS_LUNA = ROOT / "data" / "visual" / "captions_v2_luna.jsonl"


def caption_luna() -> dict:
    """Round 3: whole-book vision pass with gpt-5.6-luna, prompt v2 (same prompt as index B,
    only the model changes). Dev vision audit (13 facts): v2-mini 12/13, v2-luna 12/13, v3-gpt-4.1 10/13."""
    pages = load_pages()
    t0 = time.perf_counter()
    stats = vision.caption_pages(pages, RENDER_DIR, CAPTIONS_LUNA, model="gpt-5.6-luna", prompt="v2", log=log)
    stats.update({"time_s": round(time.perf_counter() - t0, 1), "model": "gpt-5.6-luna", "prompt_version": "v2"})
    write_json(RESULTS / "vision_luna_stats.json", stats)
    log(f"caption_luna: {stats}")
    if stats["errors"]:
        sys.exit("some pages failed; re-run `caption_luna` to retry them")
    return stats


def merged_captions() -> dict:
    caps = vision.load_cache(CAPTIONS)
    caps.update(vision.load_cache(CAPTIONS_V3))  # v3 replaces v2 where available
    return caps


def build_b3(pages):
    """B3 = text + v2 visual chunks (id ...:vNN) + v3 visual chunks (id ...:wNN) – union of both passes."""
    v2, v3 = vision.load_cache(CAPTIONS), vision.load_cache(CAPTIONS_V3)
    chunks = []
    for p in pages:
        chunks.extend(chunking.text_chunks(p, "B3"))
        chunks.extend(chunking.visual_chunks(p, v2.get(p.page), "B3", "v"))
        chunks.extend(chunking.visual_chunks(p, v3.get(p.page), "B3", "w"))
    return chunks


def ingest_b2(embedder, store, variant: str = "B2") -> dict:
    """Collection colorme3_B2 = B's text chunks + visual chunks from merged v2/v3 captions.
    A/B/C (Day 6 results) are not touched."""
    pages = load_pages()
    if variant == "B2":
        chunks = build_chunks("B2", pages, merged_captions())
    elif variant == "B4":  # text + luna v2 descriptions for every page
        chunks = build_chunks("B2", pages, vision.load_cache(CAPTIONS_LUNA))
        for c in chunks:
            c.chunk_id = c.chunk_id.replace("B2:", "B4:", 1)
    else:
        chunks = build_b3(pages)
    t0 = time.perf_counter()
    tok0 = embedder.tokens_used
    vectors = embedder.embed_documents([c.text for c in chunks])
    first = store.upsert(variant, chunks, vectors)
    second = store.upsert(variant, chunks, vectors)
    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c.metadata["content_type"]] = kinds.get(c.metadata["content_type"], 0) + 1
    out = {"collection": f"colorme3_{variant}", "chunks": len(chunks), "by_content_type": kinds,
           "embed_tokens": embedder.tokens_used - tok0, "time_s": round(time.perf_counter() - t0, 1),
           "upsert_first": first, "upsert_reload": second,
           "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(RESULTS / f"ingest_{variant.lower()}_stats.json", out)
    log(f"ingest {variant}: {out}")
    return out


# ---------------------------------------------------------------- validate
def benchmark_hash(bench: dict) -> str:
    core = [{k: v for k, v in q.items() if k != "gold"} |
            {"gold": [{"source_id": g["source_id"], "evidence_quote": g["evidence_quote"],
                       "evidence_type": g.get("evidence_type", "text"),
                       "pages": g["pages"] if g.get("evidence_type") == "visual" else None}
                      for g in q["gold"]]} for q in bench["questions"]]
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate() -> dict:
    """Text evidence: locate the quote in the normalized page text (sets gold pages).
    Visual evidence: pages are given by the human annotator (checked on the render);
    we only report whether the vision description of that page contains the key phrase.
    Runs before any retrieval; fails loudly if a text quote is missing."""
    pages = load_pages()
    by_page = {p.page: p for p in pages}
    captions = vision.load_cache(CAPTIONS)
    vis_text = {}
    for c in build_chunks("B", pages, captions):
        if c.metadata["content_type"] == "visual":
            vis_text[c.page] = vis_text.get(c.page, "") + "\n" + c.text
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    problems, warnings = [], []
    chapters = {p.metadata["chapter"] for p in pages}
    for q in bench["questions"]:
        for g in q["gold"]:
            et = g.setdefault("evidence_type", "text")
            if et == "text":
                found = [p.page for p in pages if norm(g["evidence_quote"]) in norm(p.text)]
                if not found:
                    problems.append(f"{q['id']}: text evidence not found: {g['evidence_quote'][:60]}")
                g["pages"] = found
            else:
                if not g.get("pages") or any(n not in by_page for n in g["pages"]):
                    problems.append(f"{q['id']}: visual evidence needs valid human-set pages")
                    continue
                in_caption = any(norm(g["evidence_quote"]) in norm(vis_text.get(n, "")) for n in g["pages"])
                g["phrase_in_vision_description"] = in_caption
                if not in_caption:
                    warnings.append(f"{q['id']}: key phrase '{g['evidence_quote']}' not in vision description "
                                    f"of p{g['pages']} (B/C cannot hit; counted as a miss, not removed)")
        if q.get("filter") and q["filter"].get("chapter") not in chapters:
            problems.append(f"{q['id']}: filter value not present in metadata")
    splits = [q["split"] for q in bench["questions"]]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bench.setdefault("first_validated_at", now)  # never overwritten
    bench["validated_at"] = now
    bench["benchmark_sha256"] = benchmark_hash(bench)
    bench["validation_warnings"] = warnings
    bench["counts"] = {"dev": splits.count("dev"), "test": splits.count("test"),
                       "with_filter": sum(1 for q in bench["questions"] if q.get("filter")),
                       "test_with_filter": sum(1 for q in bench["questions"] if q.get("filter") and q["split"] == "test"),
                       "visual_evidence": sum(1 for q in bench["questions"]
                                              if any(g.get("evidence_type") == "visual" for g in q["gold"]))}
    for w in warnings:
        log("WARNING " + w)
    if problems:
        for p in problems:
            log("VALIDATION ERROR " + p)
        sys.exit(1)
    write_json(BENCH, bench)
    log(f"validate: OK {bench['counts']}")
    return bench


# ---------------------------------------------------------------- ingest
def ingest(embedder, store) -> dict:
    pages = load_pages()
    captions = vision.load_cache(CAPTIONS)
    stats = {}
    cache: dict[str, list[float]] = {}
    for strategy in STRATEGIES:
        t0 = time.perf_counter()
        chunks = build_chunks(strategy, pages, captions)
        t_chunk = time.perf_counter() - t0
        texts = [c.text for c in chunks]
        t1 = time.perf_counter()
        tok0 = embedder.tokens_used
        key = "|".join(f"{c.page}{c.metadata['content_hash']}" for c in chunks)
        if key in cache:  # C has exactly B's chunks -> same vectors, no second API call
            vectors, reused = cache[key], True
        else:
            vectors, reused = embedder.embed_documents(texts), False
            cache[key] = vectors
        t_embed = time.perf_counter() - t1
        t2 = time.perf_counter()
        first = store.upsert(strategy, chunks, vectors)
        t_store = time.perf_counter() - t2
        second = store.upsert(strategy, chunks, vectors)  # reload same version: must not grow
        lengths = [len(t) for t in texts]
        kinds = {}
        for c in chunks:
            kinds[c.metadata["content_type"]] = kinds.get(c.metadata["content_type"], 0) + 1
        tokens = embedder.tokens_used - tok0
        stats[strategy] = {
            **STRATEGIES[strategy], "chunk_count": len(chunks), "by_content_type": kinds,
            "pages_covered": len({c.page for c in chunks}), "total_chars": sum(lengths),
            "embed_tokens": tokens, "len_min": min(lengths), "len_median": int(statistics.median(lengths)),
            "len_mean": round(statistics.mean(lengths), 1), "len_max": max(lengths),
            "len_p90": sorted(lengths)[int(0.9 * len(lengths))],
            "time_chunk_s": round(t_chunk, 2), "time_embed_s": round(t_embed, 2),
            "time_store_s": round(t_store, 2), "vectors_reused": reused,
            "upsert_first": first, "upsert_reload": second,
            "reload_no_duplicates": second["after"] == len(chunks) and first["after"] == len(chunks),
            "embed_cost_usd": round(tokens / 1e6 * 0.02, 6),
        }
        log(f"ingest {strategy}: {len(chunks)} chunks {kinds}, embed {t_embed:.1f}s ({tokens} tok), "
            f"reload {first['after']}->{second['after']}")
    vstats = json.loads((RESULTS / "vision_stats.json").read_text(encoding="utf-8")) \
        if (RESULTS / "vision_stats.json").exists() else {}
    write_json(RESULTS / "ingest_stats.json", {
        "embedding_model": embedder.model_name, "dimensions": embedder.dimensions,
        "vector_store": f"chromadb {__import__('chromadb').__version__} PersistentClient",
        "store_path": str(CHROMA_DIR.relative_to(ROOT)).replace("\\", "/"),
        "collections": [f"colorme3_{s}" for s in STRATEGIES],
        "metric": "cosine (HNSW, approximate)", "vision": vstats,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategies": stats})
    return stats


# ---------------------------------------------------------------- evaluate
def evaluate(embedder, store) -> list[dict]:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    sha = benchmark_hash(bench)
    if sha != bench.get("benchmark_sha256"):
        sys.exit("benchmark changed since last validate: run `python -m src.run validate` first")
    traces = []
    qvecs = {}
    for strategy, cfg in STRATEGIES.items():
        for q in bench["questions"]:
            filters = q.get("filter") if cfg["filter"] else None
            t0 = time.perf_counter()
            if q["id"] not in qvecs:  # same query vector for every strategy
                qvecs[q["id"]] = embedder.embed_query(q["question"])
            hits = store.search(strategy, qvecs[q["id"]], top_k=3, filters=filters)
            latency = time.perf_counter() - t0
            ev = evaluate_query(q, hits)
            traces.append({
                "strategy": strategy, "id": q["id"], "split": q["split"], "type": q["type"],
                "evidence_type": q["gold"][0].get("evidence_type", "text"),
                "question": q["question"], "filter_applied": filters, "question_filter": q.get("filter"),
                "expected_answer": q["expected_answer"], "gold": q["gold"], "latency_s": round(latency, 4),
                "eval": ev, "top3": [{**h, "is_hit": f} for h, f in zip(hits, ev["hit_flags"])],
            })
    with (RESULTS / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = summarize(traces)
    for row in summary:
        sub = [t for t in traces if t["strategy"] == row["strategy"] and t["split"] == row["split"]
               and t["question_filter"]]
        ok = sum(all(h["metadata"]["chapter"] == t["question_filter"]["chapter"] for h in t["top3"]) for t in sub)
        row["filter_questions"] = len(sub)
        row["filter_compliant_top3"] = ok
    write_json(RESULTS / "summary.json", {
        "benchmark_sha256": sha, "benchmark_first_validated_at": bench.get("first_validated_at"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_embedding_tokens": embedder.tokens_used, "rows": summary})
    with (RESULTS / "per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "id", "split", "type", "evidence_type", "filter", "hit1", "hit3",
                    "gold_page_in_top3", "top1_chunk", "top1_score", "top_pages"])
        for t in traces:
            w.writerow([t["strategy"], t["id"], t["split"], t["type"], t["evidence_type"],
                        json.dumps(t["question_filter"], ensure_ascii=False) if t["question_filter"] else "",
                        int(t["eval"]["hit1"]), int(t["eval"]["hit3"]), int(t["eval"]["gold_page_in_top3"]),
                        t["top3"][0]["chunk_id"] if t["top3"] else "", t["top3"][0]["score"] if t["top3"] else "",
                        " | ".join(f"p{h['metadata']['page']}:{h['metadata']['content_type']}" for h in t["top3"])])
    for row in summary:
        log(f"{row['strategy']} {row['split']}: Hit@1 {row['hit1']}/{row['N']}  Hit@3 {row['hit3']}/{row['N']} "
            f"(text {row['text_hit3']}/{row['text_N']}, visual {row['visual_hit3']}/{row['visual_N']})  "
            f"filter {row['filter_compliant_top3']}/{row['filter_questions']}")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "caption", "validate", "ingest", "evaluate", "all",
                                     "caption_v3", "ingest_b2", "ingest_b3", "caption_luna", "ingest_b4"])
    step = ap.parse_args().step
    load_env()
    if step == "caption_v3":
        caption_v3()
        return
    if step == "caption_luna":
        caption_luna()
        return
    if step in ("ingest_b2", "ingest_b3", "ingest_b4"):
        from .embeddings import OpenAIEmbedder
        from .store import ChromaStore
        emb = OpenAIEmbedder(MODEL)
        ingest_b2(emb, ChromaStore(CHROMA_DIR, emb.dimensions, emb.model_name), {"ingest_b2": "B2", "ingest_b3": "B3", "ingest_b4": "B4"}[step])
        return
    if step in ("prepare", "all"):
        prepare()
    if step in ("caption", "all"):
        caption()
    if step in ("validate", "all"):
        validate()
    if step in ("ingest", "evaluate", "all"):
        from .embeddings import OpenAIEmbedder
        from .store import ChromaStore
        embedder = OpenAIEmbedder(MODEL)
        store = ChromaStore(CHROMA_DIR, embedder.dimensions, embedder.model_name)
        if step in ("ingest", "all"):
            ingest(embedder, store)
        if step in ("evaluate", "all"):
            evaluate(embedder, store)


if __name__ == "__main__":
    main()
