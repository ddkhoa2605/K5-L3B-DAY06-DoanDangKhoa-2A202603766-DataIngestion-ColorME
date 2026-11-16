# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

---

## Bài làm: EBOOK COLORME 3 (PDF chữ + hình) + ChromaDB

Corpus: `EBOOK COLORME 3.pdf` (sách thiết kế đồ họa tiếng Việt, 183 trang, 181 trang nội dung, có ảnh/bảng/biểu đồ). Text lấy bằng PyMuPDF; nội dung hình lấy bằng vision model `gpt-4.1-mini` (mô tả từng ảnh/bảng/biểu đồ, cache trong `data/visual/captions.jsonl`). Embedding `text-embedding-3-small` (1536 chiều), ChromaDB 1.5.9 persistent (cosine, HNSW). Ba chiến lược: **A** chỉ text, **B** text + mô tả hình, **C** = B + lọc `chapter`. Bản Python-docs cũ nằm ở repo `K5-L3-Day06-DataIngestionEngineering`.

```bash
# 1. Tải PDF vào data/raw/ (PDF không nằm trong git vì bản quyền colorME):
#    curl -L -o "data/raw/EBOOK COLORME 3.pdf" "https://thietbigas.com.vn/upload/product/EBOOK%20COLORME%203.pdf"   # bản sao của bên thứ ba, SHA-256 trong data/manifest.json
#    data/raw/EBOOK COLORME 3.pdf   (SHA-256 trong data/manifest.json)
# 2. Tạo starter/.env từ starter/.env.example, điền OPENAI_API_KEY
cd starter
python -m pip install -r requirements.txt
python -m src.run all                 # prepare -> caption (vision, cache) -> validate -> ingest -> evaluate
python -m src.run caption_luna        # (dùng cho Day 7) mô tả hình bằng GPT-5.6 Luna, cache
python -m src.run ingest_b4           # (dùng cho Day 7) collection colorme3_B4
python scripts/make_audit_sheets.py   # sheet đọc để chấm thủ công
python scripts/audit_summary.py       # kiểm tra metadata + tổng hợp nhãn
python scripts/test_reingest.py       # kiểm tra nạp lại / tài liệu đổi (Chroma tạm)
```

`caption` dùng cache cục bộ `data/visual/captions.jsonl` (không nằm trong git): lần đầu gọi API (≈ $0.31), các lần sau không tốn thêm. Chi phí cho A/B/C ≈ $0.32; thêm ≈ $0.18 cho `caption_luna`. **Dữ liệu sách không có trong git:** xem [`DATA_NOT_INCLUDED.md`](DATA_NOT_INCLUDED.md) (danh sách file, SHA-256, cách tái tạo).

| Đường dẫn | Nội dung |
|---|---|
| `starter/src/` | ingestion (PDF), vision, chunking, embeddings (OpenAI), store (Chroma), evaluation, run |
| `data/manifest.json` | manifest nguồn/quyền/quy tắc trang/chương, SHA-256 của PDF |
| `data/pages.jsonl`, `data/visual/*.jsonl` | **không nằm trong git** (văn bản toàn sách, mô tả hình); tạo lại bằng pipeline, xem `DATA_NOT_INCLUDED.md` |
| `benchmark/questions.json` | 24 câu (17 dev / 7 test), 6 câu filter, 4 câu evidence hình |
| `artifacts/experiment_log.csv` | cấu hình và giả thuyết A/B/C |
| `results/` | trace top-3, chỉ số, thống kê ingest/vision, nhãn audit, log |
| `report/individual/2A202603766_DoanDangKhoa.md` | báo cáo cá nhân |

Kết quả chính (Evidence Hit@3): A dev 15/17 · test 5/7 (0/4 câu hình); B và C dev 16/17 · test 6/7 (4/4 câu hình). Chọn C. Chi tiết và hạn chế (lỗi vision ở p160) trong báo cáo.
