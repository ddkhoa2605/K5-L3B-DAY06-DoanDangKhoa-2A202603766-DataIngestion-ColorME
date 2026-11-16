"""Text chunks (all strategies) and visual-description chunks (strategies B/C).

Every chunk belongs to exactly one PDF page. Chunk IDs are deterministic
(strategy, page, kind, ordinal), so re-ingesting the same version overwrites
instead of duplicating.

- Text chunk: paragraphs of the page packed up to ``max_size`` chars, prefixed
  with ``[chapter | page title | trang]`` so a chunk is understandable alone.
- Visual chunk: one per *informative* visual element returned by the vision step
  (description + text inside the visual + table/chart data), same prefix.
"""
from __future__ import annotations

import hashlib
import json

from .models import Chunk, DocumentPage

KIND_VI = {"photo": "ảnh chụp", "illustration": "hình minh họa", "diagram": "sơ đồ", "chart": "biểu đồ",
           "table": "bảng", "infographic": "infographic", "logo": "logo", "screenshot": "ảnh chụp màn hình",
           "color_swatch": "bảng/ô màu", "typography_sample": "mẫu chữ", "poster": "poster/ấn phẩm",
           "other": "hình"}
MIN_TEXT_CHARS = 20  # pages with less text (e.g. "Ví dụ" + image) get no text chunk


def prefix(page: DocumentPage) -> str:
    m = page.metadata
    return f"[{m['chapter_title']} | {m['page_title']} | trang {m['printed_page']} (PDF p{page.page})]"


def _make(strategy: str, page: DocumentPage, kind: str, n: int, text: str, extra: dict) -> Chunk:
    m = page.metadata
    return Chunk(
        chunk_id=f"{strategy}:colorme3:p{page.page:03d}:{kind}{n:02d}",
        document_id=page.document_id, source=page.source, page=page.page, text=text,
        metadata={
            "page_id": m["page_id"], "printed_page": m["printed_page"], "chapter": m["chapter"],
            "chapter_title": m["chapter_title"], "page_title": m["page_title"],
            "render": m["render"] or "", "chars": len(text),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16], **extra,
        })


def _split_long(paragraph: str, max_size: int) -> list[str]:
    parts, buf = [], ""
    for line in paragraph.split("\n"):
        while len(line) > max_size:
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(line[:max_size])
            line = line[max_size:]
        if buf and len(buf) + 1 + len(line) > max_size:
            parts.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        parts.append(buf)
    return parts


def text_chunks(page: DocumentPage, strategy: str, max_size: int = 800) -> list[Chunk]:
    if len(page.text.strip()) < MIN_TEXT_CHARS:
        return []
    head = prefix(page) + "\n"
    budget = max_size - len(head)
    paras: list[str] = []
    for para in page.text.split("\n\n"):
        if para.strip():
            paras.extend(_split_long(para, budget) if len(para) > budget else [para])
    chunks, buf, n = [], [], 0
    for para in paras:
        if buf and len("\n\n".join(buf + [para])) > budget:
            chunks.append(_make(strategy, page, "t", n, head + "\n\n".join(buf), {"content_type": "text"}))
            n += 1
            buf = []
        buf.append(para)
    if buf:
        chunks.append(_make(strategy, page, "t", n, head + "\n\n".join(buf), {"content_type": "text"}))
    return chunks


def visual_chunks(page: DocumentPage, caption: dict | None, strategy: str, kind: str = "v") -> list[Chunk]:
    if not caption:
        return []
    res = caption["result"]
    out = []
    regions = json.dumps(page.metadata.get("image_regions", []))
    for n, v in enumerate(x for x in res["visuals"] if x["informative"]):
        lines = [prefix(page), f"Nội dung trang: {res['page_summary']}",
                 f"Phần tử trực quan ({KIND_VI.get(v['kind'], v['kind'])}, {v['position']}): {v['description']}"]
        if v["text_in_visual"].strip():
            lines.append(f"Chữ trong hình: {v['text_in_visual']}")
        if v["data"].strip():
            lines.append(f"Dữ liệu: {v['data']}")
        out.append(_make(strategy, page, kind, n, "\n".join(lines), {
            "content_type": "visual", "visual_kind": v["kind"], "image_regions": regions,
            "vision_model": caption["model"], "vision_prompt": caption["prompt_version"]}))
    return out
