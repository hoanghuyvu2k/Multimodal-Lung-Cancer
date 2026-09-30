# Kết quả kiểm chứng số liệu bài báo

> **Bài báo giữ nguyên trạng.** Mọi sai khác được gom vào
> [sai-khac-can-quyet-dinh.md](sai-khac-can-quyet-dinh.md) để quyết định sau.

> Bắt đầu 2026-09-30 · Kế hoạch: [ke-hoach-kiem-chung.md](ke-hoach-kiem-chung.md)
> Bản gốc để đối chiếu: `experiments/results/_ref_2026-09-29/` (31 file JSON)
> Runner chạy nối tiếp tất cả phase: `experiments/verify_all.py`

## Tình trạng

| Phase | Nội dung | Tình trạng |
|---|---|---|
| 0 | Tiền kiểm không train | ✅ Đạt, trừ SK-1 |
| 1a | 21 tổ hợp uniform + DyAM | ✅ 64,8 phút — 13/21 khớp tuyệt đối, 8 tổ hợp radiomics lệch (SK-2) |
| 1b | single-run uniform + DyAM | ✅ 13,1 phút |
| 1c | OvO 5-seed + single-run | ✅ 38,2 phút |
| 2 | OvO + NLP | ✅ 71,6 phút — **khớp tuyệt đối 21/21** |
| 3 | Bảng 5 & 6 (7 biến thể) | ✅ 6,4 phút — **không tái lập được (SK-3)** |
| 4 | Survival (Bảng 7, 8) | ✅ 5,0 phút — **không tái lập được (SK-4, SK-5)**; tdAUC/IBS lỗi numpy (SK-6) |
| 5 | Hình + compile | ⏸ hoãn — chờ quyết định sửa hay không |

---

## Phase 0 — Tiền kiểm (xong)

| Hạng mục | Bài báo ghi | Chạy lại | Kết luận |
|---|---|---|---|
| Bảng 1 toàn bộ | `table1_patients.tex` | `make_table1.py` sinh ra **file y hệt**, không khác 1 ký tự | ✅ |
| n cohort | 247 (62 đáp ứng / 185 không) | 247 (62/185) | ✅ |
| df_labs | 13 biến | 13 cột: age, pack_years, ecog, albumin, dnlr, brain_mets, liver_mets, tumor_burden, therapy_line, recieves_combo_therapy, site_lung, recieves_pdl1_therapy, hist_adeno | ✅ |
| Câu ví dụ Note S1 | "The patient is a 69-year-old… pack-years is 48.0…" | Trùng từng ký tự với câu sinh từ code | ✅ |
| Độ dài câu | 65–68 từ | 65–68 | ✅ |
| Số token | 129–135 (giới hạn 256) | 129–135 | ✅ |
| PCA16 giữ phương sai | ≈94% | **94,4%** | ✅ |
| Note S3: nửa độ rộng CI | ±0,07 | ±0,069 đến ±0,073 | ✅ |
| Note S3: SE(Δ) | 0,014–0,016 | **0,0158–0,0166** | ⚠️ **SK-1** |
| Note S3: n cần (Δ=0,02) | 950–1.250 | **1.211–1.343** | ⚠️ **SK-1** |
| Note S3: n cần (Δ=0,03) | 420–550 | **538–597** | ⚠️ **SK-1** |
| Bố cục Hình 1, 2, 3 | — | `check_figures.py` exit 0 | ✅ |

### Sai lệch #1 — Note S3 khai SE(Δ) thấp hơn chính giả định của nó

Note S3 nêu giả định: SE(AUC) theo Hanley–McNeil tại AUC 0,78–0,81 với 62/185 bệnh nhân, tương quan
giữa hai model ρ≈0,9, rồi viết SE(Δ) ≈ 0,014–0,016. Tính lại từ đúng các giả định đó:
SE(Δ) = √(2(1−ρ))·SE(AUC) = 0,4472 × (0,0353…0,0372) = **0,0158–0,0166**.

Kéo theo cỡ mẫu cần thiết lớn hơn con số đang ghi. **Chưa sửa vào bài báo** — ghi ở
[sai-khac-can-quyet-dinh.md](sai-khac-can-quyet-dinh.md) mục SK-1 chờ quyết định:

| | Trước | Sau |
|---|---|---|
| SE(Δ) | 0,014–0,016 | **≈0,016** |
| n cần, Δ=0,02 | 950–1.250 | **1.200–1.350** |
| n cần, Δ=0,03 | 420–550 | **540–600** |
| Thiếu công suất | ~2–5× | ~2–5× (giữ nguyên: 4,9–5,4× ở Δ=0,02; 2,2–2,4× ở Δ=0,03) |

### Phát hiện #2 — Bảng 5 và 6 không có nguồn tái lập được

`result/training-results.md` §2.0b ghi nguồn hai bảng này là notebook `Figures-Finalized.ipynb`.
Quét cả 66 cell của notebook: **không có cell nào** gọi `clinical_nlp_embedding`, `MiniLM`, hay
`sentence-transformers`; duy nhất cell 63 nhắc chữ "nlp" và đó là đoạn kiểm định bootstrap. Vậy
notebook **không** sinh ra các số của Bảng 5/6. Nguồn duy nhất còn lại là ghi chép
`document/nlp-model/nlp-ket-qua.md`.

Đã viết `experiments/nlp_verify/verify_nlp_variants.py` để dựng lại (Phase 3), dùng đúng config lấy
từ cell 13 của notebook: epochs 125, lr 0,01, alpha 0,001, beta 0, rad filter `l1_strength=0.1`,
folds 10, model DyAM.

### Phát hiện #3 — Hai nguồn ghi hai con số khác nhau cho cùng một model

| Model | Bảng 5 (bài báo) | §2.1 từ `survival_analysis_nlp.py` |
|---|---|---|
| No Clinical (IHC-A) | 0,764 | 0,776 |
| + NLP-PCA16 (IHC-A) | 0,784 | 0,773 |

Không chỉ lệch ~0,01–0,02 mà còn **đảo chiều hiệu ứng NLP** (bài báo: NLP tăng +0,020; survival
script: NLP giảm −0,003). Phase 3 và Phase 4 sẽ xác định số nào đúng.

### Phát hiện #4 — Môi trường đã đổi: scipy 1.14 làm hỏng pipeline radiomics

`run_paper_combos.py` chết ở tổ hợp #5 (tổ hợp radiomics đầu tiên) với
`scipy.optimize._optimize.BracketError` từ `PowerTransformer` (Yeo-Johnson). Các lần chạy gốc dùng
scipy < 1.14 — bản đó không raise. Đây là lý do `nlp_verify/nlp_run.py` đã có sẵn bản vá từ trước.

Đã đưa cùng bản vá vào `experiments/verify_all.py` (cột nào brent không bracket được thì λ=1, gần
identity), **không sửa `lung_helpers.py`**.

> **Hệ quả cho việc kiểm chứng:** các tổ hợp có radiomics không thể tái lập *tuyệt đối* con số cũ,
> vì thư viện bên dưới đã đổi hành vi. Chênh lệch nhỏ ở các tổ hợp này là do môi trường, không phải
> do bài báo ghi sai. Các tổ hợp không có radiomics thì vẫn phải khớp tuyệt đối.

**Xác nhận đã chạy được trước khi crash:** 4 tổ hợp k=1 (TMB 0,6161; PDL1 0,7198; IHC-A 0,6352;
Gen 0,6763) khớp bản gốc, lệch lớn nhất 0,00053. Phép kiểm k=1 (uniform phải bằng DyAM) đạt cả 4.
