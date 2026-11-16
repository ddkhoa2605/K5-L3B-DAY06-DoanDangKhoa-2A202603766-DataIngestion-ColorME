"""Metadata audit of test top-3 + aggregation of manual labels.

Per retrieved record (test questions x top-3 x strategy):
  source == manifest PDF; page exists; printed_page == page-1; chapter == chapter of that page;
  text chunk: every paragraph of its body (prefix removed) is inside the normalized page text;
  visual chunk: its description comes from the vision record of the SAME page.
  -> "required_ok" when all hold.
Run from starter/:  python scripts/audit_summary.py
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src import vision  # noqa: E402

AUDIT = ROOT / "results" / "audit"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
pdf_name = manifest["documents"][0]["source_path_or_url"].split("/")[-1]
pages = {}
for line in (ROOT / "data" / "pages.jsonl").open(encoding="utf-8"):
    p = json.loads(line)
    pages[p["page"]] = p
caps = vision.load_cache(ROOT / "data" / "visual" / "captions.jsonl")
traces = [json.loads(line) for line in (ROOT / "results" / "traces.jsonl").open(encoding="utf-8")]

rows = []
for t in traces:
    if t["split"] != "test":
        continue
    for h in t["top3"]:
        m = h["metadata"]
        page = pages.get(m["page"])
        body = h["text"].split("\n", 1)[1] if h["text"].startswith("[") else h["text"]
        if m["content_type"] == "text":
            located = bool(page) and all(norm(par) in norm(page["text"]) for par in body.split("\n\n") if par.strip())
        else:
            descs = [v["description"] for v in caps.get(m["page"], {}).get("result", {}).get("visuals", [])]
            located = any(norm(d) in norm(h["text"]) for d in descs)
        checks = {"source_ok": m["source"] == pdf_name, "page_ok": page is not None,
                  "printed_ok": m["printed_page"] == m["page"] - 1,
                  "chapter_ok": bool(page) and m["chapter"] == page["metadata"]["chapter"],
                  "located_ok": located}
        flt = t["question_filter"]
        rows.append({"strategy": t["strategy"], "id": t["id"], "rank": h["rank"], "chunk_id": h["chunk_id"],
                     "page": m["page"], "printed_page": m["printed_page"], "chapter": m["chapter"],
                     "content_type": m["content_type"], **{k: int(v) for k, v in checks.items()},
                     "required_ok": int(all(checks.values())),
                     "question_filter": json.dumps(flt, ensure_ascii=False) if flt else "",
                     "in_filter_scope": "" if not flt else int(m["chapter"] == flt["chapter"])})
with (AUDIT / "metadata_audit.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

summary = defaultdict(dict)
for s in ("A", "B", "C"):
    r = [x for x in rows if x["strategy"] == s]
    summary[s]["metadata_required_ok"] = f"{sum(x['required_ok'] for x in r)}/{len(r)}"
    fq = defaultdict(list)
    for x in r:
        if x["question_filter"]:
            fq[x["id"]].append(x["in_filter_scope"])
    summary[s]["test_filter_compliant_questions"] = f"{sum(all(v) for v in fq.values())}/{len(fq)}"
for name in ("coherence_labels.csv", "grounding_labels.csv"):
    path = AUDIT / name
    if not path.exists():
        continue
    with path.open(encoding="utf-8") as fh:
        labels = list(csv.DictReader(fh))
    for s in ("A", "B", "C"):
        sc = [int(x["score_0_2"]) for x in labels if x["strategy"] == s]
        if sc:
            c = Counter(sc)
            summary[s][name.split("_")[0]] = {"n": len(sc), "mean_0_2": round(sum(sc) / len(sc), 2),
                                              "count_0": c[0], "count_1": c[1], "count_2": c[2]}
(ROOT / "results" / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
