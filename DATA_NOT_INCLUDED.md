# Dữ liệu KHÔNG nằm trong git (Day 6)

**Lý do:** cuốn *EBOOK COLORME 3* thuộc bản quyền colorME/tác giả; giấy phép phân phối lại **chưa được xác minh**. Vì vậy repo không chứa PDF, ảnh render trang, văn bản toàn sách, và các mô tả hình đầy đủ (các mô tả này chép lại cả chữ trong hình). Nguồn lấy sách ghi ở `data/manifest.json` (`documents[0].origin`).

## Không có trong git

| File / thư mục | Kích thước | SHA-256 của bản dùng để báo cáo |
|---|---|---|
| `data/raw/EBOOK COLORME 3.pdf` | 14 297 KB | `7a739dd454715148d155783a9ea2e664a415b643ed83d06e9145cfdd9061de65` |
| `data/pages.jsonl` | 162 KB | `67293df612ce2993da3e769262a5bbee149c2c5081c6811f75680241b5f2c91b` |
| `data/visual/audit_v2_luna.jsonl` | 22 KB | `e85d0d8eca72f503182d24aa6a3b26994c203b576771b3cd6fb10eb746037619` |
| `data/visual/audit_v3_41.jsonl` | 18 KB | `629961471c70f7d4d325c666689d7f1a8e52e2283e9648b98a88aff2fa59bd59` |
| `data/visual/audit_v3_luna.jsonl` | 22 KB | `f07d84e984ea29725fc50fd8b02cad149fcef967f247acf8e19c1614d4c4d916` |
| `data/visual/audit_v3_mini.jsonl` | 20 KB | `68fac0cfe886d5eba4187669f61896ff9eafeb560beec92967d3a8702f0794ac` |
| `data/visual/captions.jsonl` | 248 KB | `a3d7f7d9f0edd4ab833e2fc76477957e48ec97f686742fe6864acf26a80ed0ad` |
| `data/visual/captions_v2_luna.jsonl` | 284 KB | `dbde6621ac55c031279f5279049e3497746684f1d4e6a634f71d9e046a4d0279` |
| `data/visual/captions_v3_gpt41.jsonl` | 198 KB | `2d29289af8ab7933ca2dc27cf3074d7380d6530b07cd58b4b8a81f956b3e0866` |
| `data/page_images/` (183 ảnh PNG, 120 dpi) | ≈ 93 MB | (tạo lại bằng `prepare`) |
| `starter/chroma_db/` (ChromaDB, 6 collection) | ≈ 60 MB | (tạo lại bằng `ingest*`) |
| `starter/.env` | | khóa API, không bao giờ commit |

## Vẫn có trong git (và lý do)

- `data/manifest.json` (nguồn, quy tắc trang/chương, SHA-256 của PDF, số ký tự mỗi trang), `benchmark/questions.json` (câu hỏi + trích đoạn evidence ngắn), `results/` (trace top-3, chỉ số, nhãn audit). Trace và sheet audit **có chứa các trích đoạn ngắn của sách** làm bằng chứng chấm; nếu cần loại hoàn toàn thì xóa `results/traces.jsonl` và `results/audit/*.md`.

## Tái tạo

1. Tải PDF (nguồn ghi ở `data/manifest.json`; **bản sao do bên thứ ba lưu trữ, không phải trang của colorME**):
   ```bash
   curl -L -o "data/raw/EBOOK COLORME 3.pdf" "https://thietbigas.com.vn/upload/product/EBOOK%20COLORME%203.pdf"
   ``` Kiểm tra: `certutil -hashfile "data\raw\EBOOK COLORME 3.pdf" SHA256` phải khớp SHA-256 ở trên.
2. `starter/.env` có `OPENAI_API_KEY=...` (mẫu: `starter/.env.example`).
3. Chạy trong `starter/` theo thứ tự:
   ```bash
   python -m pip install -r requirements.txt
   python -m src.run prepare       # đọc PDF, render ảnh, pages.jsonl, manifest
   python -m src.run caption       # mô tả hình bằng gpt-4.1-mini (index A/B/C)          ≈ $0.31
   python -m src.run caption_luna  # mô tả hình bằng GPT-5.6 Luna (index B4 của Day 7)   ≈ $0.18
   python -m src.run validate && python -m src.run ingest && python -m src.run ingest_b4 && python -m src.run evaluate
   # tùy chọn (chỉ để tái hiện thí nghiệm vòng 2): caption_v3, ingest_b2, ingest_b3
   ```
4. **Lưu ý tái lập:** mô tả hình do model sinh ra **không đảm bảo giống từng chữ** giữa các lần chạy, nên chunk, thứ hạng và điểm có thể lệch nhẹ so với số liệu báo cáo. Bảng SHA-256 ở trên cho phép so với đúng các file đã dùng; nếu cần đúng bản đó, liên hệ tác giả bài làm.
