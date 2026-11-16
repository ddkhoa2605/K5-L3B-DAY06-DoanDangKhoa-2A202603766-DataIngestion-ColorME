"""Build audit sheets for manual labeling (does NOT assign labels).

Coherence sample rule (fixed before labeling, independent of scores):
  5 TEXT positions   = middle line of the pages at 10/30/50/70/90 % of the book,
                       snapped forward to the next page with >= 150 chars of text;
  5 FIGURE positions = the pages at 20/40/60/80/95 % of the list of pages that have
                       >= 1 content image and < 150 chars of text (information lives in the image).
  For each strategy the chunk covering the position is shown: text position ->
  text chunk containing that line; figure position -> the page's visual chunk
  (B/C) or whatever chunk A has for that page (text only, maybe none).
  C uses exactly the same chunks as B, so its coherence sample is identical.

Test sheet: every test question x strategy x top-3 with metadata + text.
Run from starter/:  python scripts/make_audit_sheets.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src.run import build_chunks, load_pages  # noqa: E402

OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
pages = load_pages()
chunks = {s: build_chunks(s, pages) for s in ("A", "B")}
n = len(pages)
positions = []
for pct in (10, 30, 50, 70, 90):
    i = int(n * pct / 100)
    while len(pages[i].text) < 150:
        i += 1
    p = pages[i]
    lines = [ln for ln in p.text.split("\n") if ln.strip()]
    positions.append(("text", p.page, lines[len(lines) // 2]))
fig_pages = [p for p in pages if p.metadata["content_images"] >= 1 and len(p.text) < 150]
for pct in (20, 40, 60, 80, 95):
    p = fig_pages[min(len(fig_pages) - 1, int(len(fig_pages) * pct / 100))]
    positions.append(("figure", p.page, None))

out = ["# Coherence samples (auto-generated; labels in coherence_labels.csv)\n"]
for k, (kind, page, line) in enumerate(positions, 1):
    head = f": `{line}`" if line else " (main figure)"
    out.append(f"\n## POS{k:02d} {kind} p{page}{head}\n")
    for s in ("A", "B"):
        cand = [c for c in chunks[s] if c.page == page]
        if kind == "text":
            cand = [c for c in cand if line in c.text]
        else:
            vis = [c for c in cand if c.metadata["content_type"] == "visual"]
            cand = vis or cand
        if not cand:
            out.append(f"\n### {s}: NO CHUNK covers this position\n")
        for c in cand[:2]:
            out.append(f"\n### {s}: {c.chunk_id}\n```text\n{c.text}\n```\n")
(OUT / "coherence_samples.md").write_text("\n".join(out), encoding="utf-8")
(OUT / "coherence_positions.json").write_text(json.dumps(
    [{"pos": f"POS{k:02d}", "kind": kd, "page": pg, "line": ln} for k, (kd, pg, ln) in enumerate(positions, 1)],
    ensure_ascii=False, indent=2), encoding="utf-8")

traces = [json.loads(line) for line in (ROOT / "results" / "traces.jsonl").open(encoding="utf-8")]
out = ["# Test top-3 sheet (auto-generated; labels in metadata_audit.csv / grounding_labels.csv)\n"]
for t in traces:
    if t["split"] != "test":
        continue
    out.append(f"\n## {t['strategy']} {t['id']} ({t['evidence_type']}) filter_applied={t['filter_applied']}\n")
    out.append(f"Q: {t['question']}\nExpected: {t['expected_answer']}\n"
               f"Gold: {[(g['pages'], g['evidence_quote']) for g in t['gold']]}\n")
    for h in t["top3"]:
        m = h["metadata"]
        out.append(f"\n### rank {h['rank']} {h['chunk_id']} score={h['score']} hit={h['is_hit']}\n"
                   f"source={m['source']} page={m['page']} printed={m['printed_page']} chapter={m['chapter']} "
                   f"type={m['content_type']}\n```text\n{h['text']}\n```\n")
(OUT / "test_top3_sheet.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", OUT / "coherence_samples.md", OUT / "test_top3_sheet.md")
