# Báo cáo cá nhân — Day 6 (EBOOK COLORME 3, PDF có chữ + hình)

> Nhãn coherence/grounding và các ghi chú đối chiếu hình do agent (AI) gán kèm lý do từng dòng; **chưa được học viên rà lại** (`results/audit/*.csv`, cột `reviewed_by_student` còn trống).

## Dữ liệu và pipeline
- **Chủ đề, nguồn/quyền, số trang:** `EBOOK COLORME 3.pdf` — sách tiếng Việt "Thiết kế cho người mới bắt đầu" của lớp học colorME (Hà Nội), 8 chương: kiến thức số (pixel, độ phân giải, CMYK/RGB, layer, công cụ Photoshop), bố cục, màu sắc, typography, hiệu ứng, logo & nhận diện thương hiệu, CV, xu hướng flat design + 5 phụ lục (bảng phím tắt…). PDF 1.5 tạo bằng Word 2016, **183 trang vật lý**, 14,6 MB, SHA-256 trong `data/manifest.json`. Bản quyền thuộc colorME; **giấy phép phân phối lại chưa được xác minh**, nên PDF, ảnh render, văn bản toàn sách (`data/pages.jsonl`) và mô tả hình (`data/visual/*.jsonl`) **không nằm trong git** (xem `DATA_NOT_INCLUDED.md`). Nguồn tải: <https://thietbigas.com.vn/upload/product/EBOOK%20COLORME%203.pdf> (bản sao do bên thứ ba lưu trữ, SHA-256 trùng file đã dùng; không phải trang của colorME); người chấm đặt PDF vào `data/raw/` rồi chạy lại pipeline.
- **Quy tắc tính trang:** 1 đơn vị = 1 trang PDF (`page` = chỉ số PDF 1-based, số in ở chân trang = `page − 1` → `printed_page`). Trang được tính là nội dung khi không phải bìa (p1)/mục lục (p4) và có ≥50 ký tự text **hoặc** ≥1 ảnh nội dung → **181/183 trang tính**. 16 trang gần như chỉ có hình (<50 ký tự text, vd p41 infographic, p73-74, p80-85 ví dụ màu, p178-179 ảnh lớp học) — thông tin của chúng chỉ lấy được qua vision.
- **Chuẩn hóa, lỗi, giới hạn:** PyMuPDF 1.28.2 `get_text(sort=True)`; NFC, CRLF→LF, NBSP/tab→space, gộp khoảng trắng trong dòng, gộp ≥3 dòng trống, bỏ dòng số trang chân trang. Ảnh "template" (logo colorME + nền bìa chương, xref lặp ≥10 trang) bị loại khỏi danh sách ảnh nội dung; bbox ảnh nội dung lưu trong metadata trang. 0 lỗi trích xuất (`results/ingest_errors.json`). Giới hạn: lỗi chính tả gốc của sách giữ nguyên ("Lighrtoom", "fiel"…); 6 trang Transform Tool (p23-28) và 7 trang Selection Tool (p29-35) có text gần như trùng nhau.
- **Xử lý hình (vision):** render mỗi trang PNG 120 dpi → `gpt-4.1-mini` (structured output, temperature 0, prompt v2 trong `starter/src/vision.py`) mô tả từng phần tử trực quan: loại (ảnh/biểu đồ/bảng/ô màu/logo…), vị trí, mô tả, chữ trong hình, dữ liệu (hàng bảng, giá trị biểu đồ, cặp màu–ý nghĩa). Cache `data/visual/captions.jsonl` (cache cục bộ, không nằm trong git; mô tả đọc được từng trang khi tái tạo) theo model + prompt + SHA-256 ảnh. 183/183 trang, **$0.31**, 0 lỗi cuối cùng. Sự cố: lần đầu treo ở p106 (model lặp JSON vô hạn) → thêm timeout 90 s, `max_tokens=3500`, retry có frequency penalty.
  - **Đối chiếu bản gốc (bắt buộc):** đã so mô tả với ảnh render cho p6, p14, p15, p19, p40, p45, p49, p50, p52, p77, p78, p86, p136, p160 — đúng, **trừ p160**: biểu đồ chấm tròn thực tế 5 chấm xanh + 1 xám, mô tả ghi 4 + 1; tỉ lệ % donut/pie là ước lượng. Ghi nhận là giới hạn, không sửa tay.
- **Embedding / store / chi phí:** OpenAI `text-embedding-3-small`, **1536 chiều**, query và document cùng cách (không prefix) — thay `all-MiniLM-L6-v2` của bản Python-docs vì model đó chỉ hỗ trợ tiếng Anh. ChromaDB **1.5.9** `PersistentClient` (`starter/chroma_db/`), collection `colorme3_{A,B,C}`, cosine HNSW (approximate), score = 1 − distance. Lệnh: `cd starter && python -m src.run all`. Chi phí: embed A 37.807 tok ($0.0008), B 115.959 tok ($0.0023), C dùng lại vector B ($0); query 750 tok; vision $0.31 → **tổng ≈ $0.32**.
- **Metadata:** bắt buộc `document_id` (`colorme-ebook-3`), `source` (tên PDF), `page` (+ `page_id`, `printed_page`). Bổ sung: **`chapter`** (`front`, `ch1`…`ch8`, `appendix` — trường lọc; lấy từ trang bìa chương "CHƯƠNG n"/"PHỤ LỤC" trong text, khớp mục lục p4), `chapter_title`, `page_title` (dòng tiêu đề thanh đỏ), `content_type` (`text`|`visual`), `visual_kind`, `image_regions` (bbox điểm PDF), `render`, `vision_model`, `content_hash`. Chọn `chapter` vì người đọc sách thường biết mình đang ở chương nào, và nhiều chủ đề lặp giữa chương (màu sắc ở ch3, ch6, ch7; layer ở ch1, ch5, phụ lục).

## Cấu hình và benchmark
- **A** — chỉ text PDF; chunk theo trang, gói đoạn ≤800 ký tự, prefix `[chương | tiêu đề trang | trang in (PDF p)]`; không lọc. 181 chunk.
- **B** — chunk text giống hệt A **+ 1 chunk cho mỗi phần tử trực quan "informative"** từ vision (cùng prefix + tóm tắt trang); không lọc. 476 chunk (181 text + 295 visual). **A–B chỉ khác việc thêm nội dung hình.**
- **C** — cùng chunk và vector với B; **khác biệt duy nhất B–C:** `where={"chapter": …}` khi câu hỏi có filter (6/24 câu).
- **Benchmark** `benchmark/questions.json`: 24 câu, chia trước khi chạy **17 dev / 7 test**; loại: direct, paraphrase, condition, exception, metadata_filter, visual_table, visual_chart. **6 câu filter** (dev d13-d15, test t02, t06, t07) — filter là chương người dùng nói trước ("đang đọc chương …"). **4 câu visual** (d16, d17, t05, t06): trang gold do người chấm chọn sau khi xem ảnh render; `evidence_quote` là cụm từ khóa mô tả đúng phải chứa. Evidence text được `validate` tìm nguyên văn (24/24 OK, 0 cảnh báo visual); `benchmark_sha256` lưu trong `questions.json` và `results/summary.json`.
- **Tiêu chí chọn (chốt trước):** dev Evidence Hit@3 → dev Hit@1 → ít chunk/chi phí.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Text / Visual Hit@3 | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|---|
| A | dev | 15/17 | 15/17 | 15/15 · 0/2 | 181 | embed 5,3 s | $0.0008 |
| A | test | 3/7 | 5/7 | 5/5 · 0/2 | 181 | | |
| B | dev | 14/17 | **16/17** | 14/15 · 2/2 | 476 | embed 2,1 s (+ vision 183 trang) | $0.0023 + vision $0.31 |
| B | test | 4/7 | **6/7** | 4/5 · 2/2 | 476 | | |
| C | dev | 14/17 | 16/17 | 14/15 · 2/2 | 476 | 0 s (vector B) | $0 |
| C | test | 4/7 | 6/7 | 4/5 · 2/2 | 476 | | |

| Strategy | Coherence /2 (n=10) | Metadata đủ và đúng (test top-3) | Filter tuân thủ (test) | Grounding 0/1/2 và TB /2 (7 câu test) |
|---|---|---|---|---|
| A | 1,10 (0:4 · 1:1 · 2:5) | 21/21 | 2/3 | 1/1/5 → 1,57 |
| B | 2,00 (2:10) | 21/21 | 3/3 | 1/0/6 → 1,71 |
| C | 2,00 (2:10) | 21/21 | 3/3 | 1/0/6 → 1,71 |

- Trace: `results/traces.jsonl`, `results/per_question.csv`; số liệu `results/summary.json`, `results/ingest_stats.json`; nhãn `results/audit/{coherence,grounding}_labels.csv`, `metadata_audit.csv`, tổng hợp `results/audit_summary.json`.
- Mẫu coherence (quy tắc trong `starter/scripts/make_audit_sheets.py`, chốt trước khi chấm): 5 vị trí text = dòng giữa trang ở 10/30/50/70/90 % sách; 5 vị trí hình = trang chỉ-có-hình ở 20/40/60/80/95 % danh sách. Metadata kiểm tra tự động bằng `audit_summary.py` (source, page, printed_page, chapter, chunk nằm đúng trang/caption đúng trang).
- Nạp lại (`starter/scripts/test_reingest.py`, Chroma tạm, chương 7): cùng phiên bản 29→29 (không trùng); trang cuối đổi nội dung → xóa 1 chunk cũ, 29→28 (`results/reingest_check.txt`).

## Phân tích và kết luận
- **A–B (thêm nội dung hình):** A không thể trả lời câu về hình: d16 (iPhone 401 ppi), d17 (bong bóng lớn nhất), t05 (từ khóa màu orange), t06 (màu của thanh "Uy quyền…") đều 0/4, vì thông tin chỉ nằm trong ảnh; B 4/4. Coherence vị trí hình: A 0 (không có chunk nào cho trang chỉ-có-hình) → B 2. Đổi lại B **mất 1 câu text mỗi split**: d01 ("Pixel là gì?") — top-3 của B là 3 chunk visual p10-p11 (ảnh pixel Captain America / Flappy Bird) đẩy chunk text p10 ra ngoài; t03 (FedEx) — 3 chunk của p147 (text + 2 visual logo ngựa/1234) chiếm hết top-3, p146 bị đẩy ra. Tức là chunk visual cùng chủ đề "chiếm chỗ" trong top-k nhỏ.
- **B–C (filter chương):** cùng kết quả Hit@3; filter đảm bảo top-3 nằm đúng chương (test 3/3) nhưng với corpus 1 tài liệu, chương đã gần như được dense tự nhận ra — tác dụng nhỏ. Hạn chế: filter cắt mất câu trả lời nếu người dùng nói sai chương.
- **Chọn C** theo tiêu chí: dev Hit@3 B = C (16) > A (15); dev Hit@1 hòa (14); chi phí bằng nhau (C dùng lại vector) → hòa; chọn C vì đảm bảo phạm vi khi người dùng cung cấp chương (tie-break này nằm ngoài tiêu chí chính, ghi rõ). Day 7 dùng collection `colorme3_B` (= C không filter) và để agent tự quyết định filter.
- **Hạn chế:** test nhỏ (7 câu) → chênh 1 câu = 14 %; mô tả hình là output model, có lỗi (p160) và không bảo đảm đầy đủ; gold visual dùng cụm từ khóa ngắn nên chỉ là proxy, đúng/sai phải đối chiếu ảnh; nhãn thủ công do AI gán, cùng người viết benchmark → rủi ro thiên lệch; thời gian vision bị ảnh hưởng bởi lần treo p106.
- **Nếu làm lại:** tách top-k theo loại (vd 2 text + 2 visual) hoặc gộp chunk visual cùng trang để giảm "chiếm chỗ"; pass vision thứ hai (model mạnh hơn / prompt chuyên cho biểu đồ đếm phần tử) cho các trang có biểu đồ; thêm OCR đối chứng cho chữ trong hình.
