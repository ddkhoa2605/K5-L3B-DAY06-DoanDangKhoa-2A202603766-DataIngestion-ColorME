"""Describe the visual content of each page with an OpenAI vision model.

Input per page: the PNG render + the text already extracted by PyMuPDF (so the
model can focus on what text extraction misses: photos, charts, tables drawn as
images, colour swatches, infographics, text inside images).

Output is cached in ``data/visual/captions.jsonl`` (one line per page) keyed by
model + prompt version + render SHA-256, so re-running costs nothing and graders
can read every description. Descriptions are model output: they must be checked
against the page render before being used as benchmark evidence.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PROMPT_VERSION = "v2"
DEFAULT_MODEL = os.environ.get("VISION_MODEL", "gpt-4.1-mini")
# USD per 1M tokens (input, output) – used only for the cost log.
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.0, 8.0),
          "gpt-4o": (2.5, 10.0), "gpt-5.6-luna": (0.20, 1.20)}  # luna: secondary-source list price

SYSTEM = """Bạn là công cụ trích xuất nội dung trực quan từ một trang ebook tiếng Việt về thiết kế đồ họa (colorME).
Nhiệm vụ: mô tả CHÍNH XÁC những gì nhìn thấy trong ảnh/hình/biểu đồ/bảng của trang, để một hệ thống tìm kiếm có thể trả lời câu hỏi về chúng.
Quy tắc:
- Bỏ qua phần trang trí lặp lại ở mọi trang: logo "color ME" góc trên trái, thanh tiêu đề màu đỏ, thanh đỏ cuối trang chứa số trang.
- Chỉ mô tả điều thấy được; không suy đoán thông tin không có trong ảnh. Nếu không chắc, ghi "không rõ".
- Chép nguyên văn chữ nằm TRONG ảnh (kể cả tiếng Anh) vào text_in_visual. Không chép lại đoạn văn bản thường của trang (đã có sẵn).
- Bảng: ghi lại từng hàng/cột dạng "hàng: giá trị | giá trị". Biểu đồ: loại biểu đồ, nhãn, chú giải, giá trị/tỉ lệ ước lượng và màu tương ứng.
- Màu sắc: gọi tên màu bằng tiếng Việt (đỏ, cam, vàng, xanh lá, xanh dương, tím, hồng, đen, trắng, xám, nâu...) và gắn màu với nhãn/ý nghĩa tương ứng nếu trang thể hiện sự tương ứng đó.
- Logo/sản phẩm/thương hiệu: ghi tên nếu đọc được hoặc thể hiện rõ trong ảnh.
- Mỗi phần tử trực quan riêng biệt (ảnh, biểu đồ, bảng, nhóm ô màu...) là một mục trong visuals. Các thanh/ô/hàng cùng loại xếp thành một nhóm (ví dụ danh sách thanh màu kèm chữ, lưới ảnh ví dụ) PHẢI gộp thành MỘT mục (kind table/color_swatch/...) và liệt kê từng hàng trong data, ví dụ "đen: Trang trọng, ... | xanh dương: Uy quyền, ...".
- KHÔNG tạo mục cho đoạn văn bản thường của trang (tiêu đề, đoạn giải thích, link) – chúng đã được trích xuất riêng.
- informative=false cho phần tử chỉ để trang trí, không chứa thông tin.
- Viết tiếng Việt, ngắn gọn nhưng đầy đủ chi tiết có thể hỏi được."""

# v3 (Day 7 dev error analysis): v2 mis-counted elements (p9: "11 icons", real 10) and gave
# loose estimates. v3 adds explicit counting / enumeration / exact-reading rules.
COUNT_RULES = """
Quy tắc bổ sung (độ chính xác dữ liệu):
- ĐẾM cẩn thận mọi phần tử lặp lại (chấm, ô, icon, thanh, ảnh nhỏ, logo, đường lưới, đối tượng): đếm từng cái theo thứ tự trái->phải, trên->dưới rồi ghi tổng, ví dụ "6 chấm: 5 tô xanh dương + 1 xám".
- Với biểu đồ/thanh tiến trình/vòng tròn: ghi số phần tử, phần được tô / tổng, và tỉ lệ kèm chữ "ước lượng"; không làm tròn thành con số đẹp nếu không chắc.
- Đọc chính xác mọi số, đơn vị, ngày tháng, mã màu, nhãn trục/chú giải trong hình; chép nguyên văn.
- Với poster/bìa/quảng cáo: tách riêng tên thương hiệu/tạp chí (masthead) và tiêu đề/khẩu hiệu chính (headline).
- Với vị trí (lưới, điểm giao, trái/phải): mô tả vị trí tương đối rõ ràng."""

PROMPTS = {"v2": SYSTEM, "v3": SYSTEM + COUNT_RULES}

SCHEMA = {
    "name": "page_visuals",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["page_summary", "visuals"],
        "properties": {
            "page_summary": {"type": "string", "description": "1-2 câu: trang này nói/minh họa điều gì"},
            "visuals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "position", "description", "text_in_visual", "data", "informative"],
                    "properties": {
                        "kind": {"type": "string", "enum": [
                            "photo", "illustration", "diagram", "chart", "table", "infographic",
                            "logo", "screenshot", "color_swatch", "typography_sample", "poster", "other"]},
                        "position": {"type": "string", "description": "vị trí trên trang: trên/giữa/dưới, trái/phải"},
                        "description": {"type": "string"},
                        "text_in_visual": {"type": "string"},
                        "data": {"type": "string", "description": "hàng/cột bảng, giá trị biểu đồ, cặp màu-ý nghĩa; rỗng nếu không có"},
                        "informative": {"type": "boolean"},
                    },
                },
            },
        },
    },
}


def _client():
    from openai import OpenAI  # fail loudly if the SDK or key is missing

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in starter/.env)")
    return OpenAI(timeout=90, max_retries=2)


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["page"]] = rec
    return cache


def describe_page(client, model: str, png: Path, page_text: str, page: int,
                  prompt: str = PROMPT_VERSION) -> dict[str, Any]:
    b64 = base64.b64encode(png.read_bytes()).decode()
    user = [
        {"type": "text", "text": f"Trang PDF số {page}. Văn bản đã trích xuất của trang (để tham khảo, KHÔNG chép lại):\n"
                                 f"<<<\n{page_text[:3000]}\n>>>\nHãy mô tả các phần tử trực quan của trang."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
    ]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            # max_tokens guards against degenerate repetition loops (seen on p106 with the
            # unbounded default); retries add a frequency penalty to break the loop.
            if model.startswith(("gpt-5", "gpt-6")):  # reasoning models: no temperature/penalties
                params = {"max_completion_tokens": 16000, "reasoning_effort": "low"}
            else:
                params = {"temperature": 0, "max_tokens": 3500,
                          "frequency_penalty": 0.0 if attempt == 0 else 0.6}
            resp = client.chat.completions.create(
                model=model, **params,
                messages=[{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": SCHEMA})
            latency = time.perf_counter() - t0
            if resp.choices[0].finish_reason == "length":
                raise ValueError("output truncated at max_tokens (repetition loop)")
            parsed = json.loads(resp.choices[0].message.content)
            pin, pout = PRICES.get(model, (0.0, 0.0))
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
            usage["cost_usd"] = round(usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout, 6)
            return {"result": parsed, "usage": usage, "latency_s": round(latency, 2), "model": resp.model}
        except Exception as exc:  # retry transient API errors, then surface them
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"vision failed on page {page}: {last_exc}")


def caption_pages(pages, render_dir: Path, cache_path: Path, model: str = DEFAULT_MODEL,
                  log=print, prompt: str = PROMPT_VERSION) -> dict[str, Any]:
    """Describe every page not yet in the cache (or whose render/prompt/model changed)."""
    cache = load_cache(cache_path)
    client = None
    done = skipped = 0
    errors: list[str] = []
    for p in pages:
        png = render_dir / p.metadata["render"]
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        old = cache.get(p.page)
        if old and old["render_sha256"] == sha and old["prompt_version"] == prompt \
                and old["requested_model"] == model:
            skipped += 1
            continue
        client = client or _client()
        try:
            out = describe_page(client, model, png, p.text, p.page, prompt)
        except RuntimeError as exc:
            errors.append(str(exc))
            log(f"  ERROR {exc}")
            continue
        cache[p.page] = {"page": p.page, "render": p.metadata["render"], "render_sha256": sha,
                         "requested_model": model, "prompt_version": prompt, **out}
        done += 1
        # rewrite after each page so an interrupted run keeps its progress
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(json.dumps(cache[k], ensure_ascii=False) for k in sorted(cache)) + "\n",
                              encoding="utf-8")
        log(f"  vision p{p.page:03d}: {len(out['result']['visuals'])} visuals, "
            f"{out['usage']['input_tokens']}+{out['usage']['output_tokens']} tok, {out['latency_s']}s")
    total_cost = sum(r["usage"]["cost_usd"] for r in cache.values())
    return {"described_now": done, "cached": skipped, "errors": errors, "pages_in_cache": len(cache),
            "total_cost_usd": round(total_cost, 4)}
