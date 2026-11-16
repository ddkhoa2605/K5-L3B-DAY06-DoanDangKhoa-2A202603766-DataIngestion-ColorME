"""Dev audit of vision extraction accuracy (Day 7 improvement loop).

Ground-truth facts below were read by a human on the page renders. Pages are
chosen from DEV / non-test pages only (none of the Day 7 test-v1 or test-v2 gold
pages). Each variant (model + prompt) describes the same pages; a fact passes when
its `must` regexes all match and no `must_not` regex matches the page's description
(all visuals joined). Output: results/vision_audit_dev.json (+ per-variant caches).

Run from starter/:  python scripts/vision_audit.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src import vision  # noqa: E402
from src.run import CAPTIONS, RENDER_DIR, load_env, load_pages  # noqa: E402

FACTS = [
    {"page": 9, "fact": "10 lý do kèm 10 biểu tượng (không phải 11)",
     "must": [r"\b10\b"], "must_not": [r"\b11 (biểu tượng|icon|lý do|ô)"]},
    {"page": 12, "fact": "6 ảnh 50/100/150/200/250/300 pixels per inch",
     "must": [r"50 pixels per inch", r"300 pixels per inch"], "must_not": []},
    {"page": 42, "fact": "simmer 20 to 30 minutes; recipes serve 4",
     "must": [r"20 (to|-|–|đến) 30 minutes|20 to 30", r"serve 4"], "must_not": []},
    {"page": 57, "fact": "vòng 12 ô; 3 ô đậm vàng, đỏ, xanh dương",
     "must": [r"\b12\b", r"vàng", r"đỏ", r"xanh dương"], "must_not": []},
    {"page": 63, "fact": "supercolor 50 COLORS. 50 CHOICES.",
     "must": [r"50 COLORS"], "must_not": []},
    {"page": 68, "fact": "3 vòng tròn màu",
     "must": [r"\b(3|ba) vòng|(Vòng tròn màu[\s\S]*){3}"], "must_not": []},  # 3 vòng, hoặc 3 mục vòng riêng
    {"page": 82, "fact": "7 chai neon: đỏ, cam, vàng, xanh lá, xanh dương, tím, hồng",
     "must": [r"\b(7|bảy) chai"], "must_not": []},  # round 3: accept number in words
    {"page": 149, "fact": "2 logo nền xanh dương (Windows, Apple) + 7 logo nền xanh lá",
     "must": [r"\b(2|hai)\b", r"\b(7|bảy)\b", r"Skype"], "must_not": []},
    {"page": 150, "fact": "8 biểu tượng trắng trên nền tròn đỏ",
     "must": [r"\b(8|tám) (biểu tượng|icon|huy hiệu|hình tròn)"], "must_not": []},  # round 3: synonyms
    # round 3: facts that the v3/gpt-4.1 pass lost in round 2 (dev / retired-test pages only)
    {"page": 49, "fact": "gương mặt Brad Pitt ở điểm giao trên bên trái",
     "must": [r"trên[ -]?(cùng )?(bên )?trái|góc trên bên trái|phía trên bên trái"], "must_not": []},
    {"page": 6, "fact": "bong bóng xanh lá (Thiết kế ấn phẩm truyền thông) lớn nhất",
     "must": [r"lớn nhất"], "must_not": []},
    {"page": 50, "fact": "gương mặt Santa ở điểm giao dưới bên trái",
     "must": [r"dưới[ -]?(cùng )?(bên )?trái|góc dưới bên trái|phía dưới bên trái"], "must_not": []},
    {"page": 10, "fact": "lưới pixel 21 dòng x 19 cột",
     "must": [r"21", r"19"], "must_not": []},
]

VARIANTS = {
    "v2_mini": {"model": "gpt-4.1-mini", "prompt": "v2", "cache": CAPTIONS},
    "v3_mini": {"model": "gpt-4.1-mini", "prompt": "v3", "cache": ROOT / "data" / "visual" / "audit_v3_mini.jsonl"},
    "v3_41": {"model": "gpt-4.1", "prompt": "v3", "cache": ROOT / "data" / "visual" / "audit_v3_41.jsonl"},
    # round 3: GPT-5.6 Luna (reasoning effort low)
    "v2_luna": {"model": "gpt-5.6-luna", "prompt": "v2", "cache": ROOT / "data" / "visual" / "audit_v2_luna.jsonl"},
    "v3_luna": {"model": "gpt-5.6-luna", "prompt": "v3", "cache": ROOT / "data" / "visual" / "audit_v3_luna.jsonl"},
}


def page_text(rec) -> str:
    out = []
    for v in rec["result"]["visuals"]:
        out += [v["description"], v["text_in_visual"], v["data"]]
    return " ".join(out)


def main() -> None:
    load_env()
    pages = {p.page: p for p in load_pages()}
    sel = [pages[f["page"]] for f in FACTS]
    report = {}
    for name, cfg in VARIANTS.items():
        if name != "v2_mini":  # v2 already cached for all pages
            vision.caption_pages(sel, RENDER_DIR, cfg["cache"], model=cfg["model"], prompt=cfg["prompt"],
                                 log=lambda m: None)
        cache = vision.load_cache(cfg["cache"])
        rows, cost = [], 0.0
        for f in FACTS:
            rec = cache[f["page"]]
            cost += rec["usage"]["cost_usd"]
            txt = page_text(rec)
            ok = all(re.search(r, txt, re.I) for r in f["must"]) and not any(re.search(r, txt, re.I) for r in f["must_not"])
            rows.append({"page": f["page"], "fact": f["fact"], "pass": ok})
        report[name] = {"model": cfg["model"], "prompt": cfg["prompt"], "pass": sum(r["pass"] for r in rows),
                        "N": len(rows), "cost_usd_10_pages": round(cost, 4), "rows": rows}
        print(name, f"{report[name]['pass']}/{len(rows)}", f"${cost:.4f}",
              [r["page"] for r in rows if not r["pass"]])
    (ROOT / "results" / "vision_audit_dev.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
