"""Automatic retrieval metrics (Hit@1, Evidence Hit@3) with full traces.

A retrieved chunk is a hit only if it comes from a gold page AND contains the
whole evidence quote (case/whitespace-insensitive, NFC).
- evidence_type "text": the quote is copied verbatim from the PDF text layer.
- evidence_type "visual": the quote is a short key phrase that a correct
  description of the figure must contain (e.g. a colour name or a label read in
  the image). The fact itself was checked by a human against the page render;
  the phrase only lets code decide whether the retrieved chunk carries it.
Manual coherence / grounding labels are NOT inferred here.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def chunk_is_hit(hit: dict[str, Any], golds: list[dict[str, Any]]) -> bool:
    meta, body = hit["metadata"], norm(hit["text"])
    return any(meta["document_id"] == g["source_id"] and meta["page"] in g["pages"]
               and norm(g["evidence_quote"]) in body for g in golds)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [chunk_is_hit(h, question["gold"]) for h in hits]
    page_only = [any(h["metadata"]["document_id"] == g["source_id"] and h["metadata"]["page"] in g["pages"]
                     for g in question["gold"]) for h in hits]
    return {"hit_flags": flags, "hit1": bool(flags and flags[0]), "hit3": any(flags[:3]),
            "gold_page_in_top3": any(page_only[:3])}


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy in sorted({t["strategy"] for t in traces}):
        for split in ("dev", "test"):
            sub = [t for t in traces if t["strategy"] == strategy and t["split"] == split]
            if not sub:
                continue
            n = len(sub)
            h1 = sum(t["eval"]["hit1"] for t in sub)
            h3 = sum(t["eval"]["hit3"] for t in sub)
            pg = sum(t["eval"]["gold_page_in_top3"] for t in sub)
            row = {"strategy": strategy, "split": split, "N": n, "hit1": h1, "hit3": h3,
                   "gold_page_top3": pg, "hit1_rate": round(h1 / n, 3), "hit3_rate": round(h3 / n, 3)}
            for et in ("text", "visual"):
                s2 = [t for t in sub if t["evidence_type"] == et]
                row[f"{et}_N"] = len(s2)
                row[f"{et}_hit3"] = sum(t["eval"]["hit3"] for t in s2)
            rows.append(row)
    return rows
