"""Read and normalize the colorME ebook PDF (EBOOK COLORME 3.pdf, 183 pages).

Every PDF page becomes one ``DocumentPage`` (page = 1-based PDF page index; the
number printed in the red footer is ``page - 1`` and is kept as ``printed_page``).

Per page we keep:
- normalized text (PyMuPDF, reading order ``sort=True``; the footer page number is removed)
- ``page_title`` (the text in the red title bar) and the chapter it belongs to
- bounding boxes of *content* images (template images that repeat on >=10 pages,
  i.e. the colorME header logo and chapter-cover background, are excluded)
- a PNG render of the page (input for the vision step and for checking evidence)
"""
from __future__ import annotations

import collections
import hashlib
import re
import unicodedata
from pathlib import Path

from .models import DocumentPage

DOCUMENT_ID = "colorme-ebook-3"
PDF_NAME = "EBOOK COLORME 3.pdf"
RENDER_DPI = 120
TEMPLATE_MIN_PAGES = 10  # an image xref used on >= 10 pages is page decoration

# Chapter titles as printed on the chapter cover pages (MỤC LỤC uses short forms).
CHAPTER_TITLES = {
    "front": "Mở đầu (bìa, lời giới thiệu, mục lục)",
    "ch1": "Chương 1 – Kiến thức cơ bản về thiết kế kĩ thuật số",
    "ch2": "Chương 2 – Các bố cục thường gặp và ứng dụng",
    "ch3": "Chương 3 – Sử dụng màu sắc",
    "ch4": "Chương 4 – Những điều cần biết về Typography",
    "ch5": "Chương 5 – Một số hiệu ứng cơ bản",
    "ch6": "Chương 6 – Nhận diện thương hiệu",
    "ch7": "Chương 7 – Ứng dụng thiết kế CV",
    "ch8": "Chương 8 – Các xu hướng thiết kế",
    "appendix": "Phụ lục và lời kết",
}

_CHAPTER_RE = re.compile(r"^\s*CHƯƠNG\s+(\d+)", re.M)
_APPENDIX_RE = re.compile(r"^\s*(PHỤ LỤC\s+\d+|LỜI KẾT)", re.M)


def normalize_text(text: str) -> str:
    """NFC, unify newlines/spaces, collapse runs of spaces inside a line and blank lines."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("\t", " ")
    text = text.replace("​", "").replace("", "-")
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_footer_number(text: str, printed: int) -> str:
    """Remove the footer page number (last non-empty line equal to the printed number)."""
    lines = text.rstrip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == str(printed):
        lines.pop()
    return "\n".join(lines).strip()


def template_xrefs(doc) -> set[int]:
    count = collections.Counter()
    for page in doc:
        for xref in {img[0] for img in page.get_images(full=True)}:
            count[xref] += 1
    return {x for x, c in count.items() if c >= TEMPLATE_MIN_PAGES}


def content_image_regions(page, template: set[int]) -> list[list[float]]:
    """Bounding boxes (PDF points, clipped to the page) of non-template images."""
    regions = []
    W, H = page.rect.width, page.rect.height
    for img in page.get_images(full=True):
        if img[0] in template:
            continue
        for r in page.get_image_rects(img[0]):
            box = [max(0, r.x0), max(0, r.y0), min(W, r.x1), min(H, r.y1)]
            if box[2] - box[0] > 5 and box[3] - box[1] > 5:
                regions.append([round(v, 1) for v in box])
    return regions


def load_pdf(pdf_path: Path, render_dir: Path | None = None) -> tuple[list[DocumentPage], list[str], dict]:
    import pymupdf  # PyMuPDF

    errors: list[str] = []
    raw = pdf_path.read_bytes()
    doc = pymupdf.open(stream=raw, filetype="pdf")
    template = template_xrefs(doc)
    chapter = "front"
    pages: list[DocumentPage] = []
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
    for i, pg in enumerate(doc):
        n = i + 1
        printed = n - 1
        try:
            text = normalize_text(pg.get_text("text", sort=True))
        except Exception as exc:  # keep going, record the page as failed
            errors.append(f"page {n}: text extraction failed: {exc}")
            text = ""
        text = _strip_footer_number(text, printed)
        m = _CHAPTER_RE.search(text)
        if m:
            chapter = f"ch{int(m.group(1))}"
        elif _APPENDIX_RE.search(text):
            chapter = "appendix"
        first_line = next((ln for ln in text.split("\n") if ln.strip()), "")
        regions = content_image_regions(pg, template)
        render_rel = None
        if render_dir:
            out = render_dir / f"p{n:03d}.png"
            if not out.exists():
                pg.get_pixmap(dpi=RENDER_DPI).save(out)
            render_rel = out.name
        pages.append(DocumentPage(
            document_id=DOCUMENT_ID, source=PDF_NAME, page=n, text=text,
            metadata={
                "page_id": f"{DOCUMENT_ID}#p{n:03d}", "printed_page": printed,
                "chapter": chapter, "chapter_title": CHAPTER_TITLES[chapter],
                "page_title": first_line[:120], "text_chars": len(text),
                "content_images": len(regions), "image_regions": regions,
                "drawings": len(pg.get_drawings()), "render": render_rel,
            }))
    info = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "page_count": doc.page_count,
            "pdf_metadata": {k: v for k, v in doc.metadata.items() if v}, "template_xrefs": sorted(template),
            "page_size_pt": [round(doc[0].rect.width, 1), round(doc[0].rect.height, 1)]}
    return pages, errors, info
