# Plan: Kiểm chứng NLP-clinical trên 21 tổ hợp bài báo (single-run + 5-seed)

> Ngày tạo: 2026-07-22
> Mục tiêu: kiểm chứng đóng góp **NLP-clinical** (mã hoá 13 biến lâm sàng → câu chữ → embedding 384-dim MiniLM)
> trên **đúng 21 tổ hợp** của `bang-so-sanh-paper-combos.md`, đo **cả single-run và 5-seed**, cập nhật vào bảng đó.

## 0. Bối cảnh & thiết kế

- `cnl_dem_labs` = **13 biến lâm sàng THÔ** (tuổi, ECOG, albumin, dNLR…). NLP mã hoá đúng 13 biến này thành câu chữ
  rồi embedding → modality `cnl_nlp` (384-dim, **no_scale** theo CLAUDE.md vì đã chuẩn hoá cosine).
- **So sánh 3 cách dùng lâm sàng** (đúng như phần survival của bài báo), cho MỖI tổ hợp:
  - **base** = tổ hợp KHÔNG có lâm sàng (bỏ `cnl_dem_labs` nếu combo #20/#21 có).
  - **+Labs** = base + 13 biến thô.
  - **+NLP** = base + embedding NLP của 13 biến.
- Fusion model: **uniform_avg** (backbone; neural≈LR nên đại diện tốt). Cả **5-seed** (robust) và **single-run** (kiểu bài báo).
- **Câu hỏi:** +NLP có > +Labs (và > base) không, và có nhất quán qua các tổ hợp không?

## 1. Nguyên tắc
- Mã `experiments/nlp_verify/`. KHÔNG sửa lung_helpers. Tái dùng harness (train_uniform_avg, get_training_data, build_l1).
- Embedding NLP build **một lần** → cache parquet. Kết quả `results/nlp_combos.json` + scores (lưu tăng dần).
- Mỗi Step một lượt loop; job nặng chạy background; utf-8; **chạy python từ code root** (cwd hay trôi).

## 2. Các bước

### Step 0 — Build NLP modality + smoke
- [x] `nlp_modality.py` + cache `nlp_clinical_embed.parquet` (247×384). `add_nlp_modality(ctx)` thêm `cnl_nlp` (no_scale)
- [x] **Sự cố môi trường (đã fix):** các lần cài NLP nâng numpy/scipy → scipy≥1.14 (bản duy nhất cho py3.13) raise
  BracketError trên cột radiomics suy biến (PowerTransformer yeo-johnson). Fix: **monkeypatch** `_yeo_johnson_optimize`
  trả lambda=1 khi bracket lỗi (khôi phục hành vi scipy cũ, không sửa lung_helpers). BM1 reproduce 0.7917≈0.7915 ✓
- [x] **Smoke BM1 (~160s/variant): base 0.7720 / +Labs 0.7661 / +NLP 0.7832 (5-seed); single-run 0.7917/0.7879/0.8091.**
  → NLP vượt cả base lẫn Labs thô. Full Step 1 ≈ 21×3×160s ≈ ~2.5h

### Step 1 — Chạy 21 tổ hợp × {base, +Labs, +NLP}
- [x] uniform_avg cho mỗi tổ hợp × 3 biến thể, **5-seed** (mean±sd) + **single-run**; no_scale đúng vị trí NLP; l1 filter rad đúng vị trí
- [x] Lưu `results/nlp_combos.json` (đủ 21/21) + single_run. base khớp cột `uni` cũ trong ~0.003 (môi trường monkeypatch)

### Step 2 — Cập nhật bảng + kết luận
- [x] Thêm bảng NLP vào `bang-so-sanh-paper-combos.md`: mỗi combo có base / +Labs / +NLP (5-seed) và (single-run)
- [x] Nhận xét: +NLP > +Labs 19/21 (5-seed), 18/21 (single-run); fusion đa nguồn thắng gần tuyệt đối. Viết `ket-qua.md` → dừng loop

## 3. Nhật ký
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-22 | 0 | NLP modality build + cache (247×384); smoke BM1: base 0.7720 / +Labs 0.7661 / +NLP 0.7832 (5-seed) | fix môi trường (monkeypatch PowerTransformer); chạy full |
| 2026-07-22 | 1 | Đủ 21/21 combo × {base,+Labs,+NLP} × {5-seed, single-run} (2 lần resume do job bị kill, lưu tăng dần) | OK |
| 2026-07-22 | 2 | **+NLP > +Labs 19/21 (5s), 18/21 (1r)**; fusion đa nguồn +0.015→0.053; BM1 +NLP 0.7832/0.8091 (best); đỉnh 1run 0.8105 (#19) | Đóng góp dương DUY NHẤT → đưa vào paper; dừng loop |

## 4. Ghi chú
- no_scale: khi `cnl_nlp` ở vị trí p trong list → truyền `no_scale=[p]` vào model_params (train_uniform_avg → MultiModalDynamicModelUncertainty).
- base cho 19 combo không-labs = chính tổ hợp đó → AUC phải khớp cột `uni`/`uni¹` cũ (dùng làm reproduce check).
- Nếu thời gian/run lớn (Step 0 đo), tách Step 1 thành 5-seed và single-run 2 lượt.
- Tiêu chí GIỮ (cho paper): +NLP ≥ +Labs nhất quán (nhất là combo có PDL1/Gen mạnh); nếu chỉ ngang → vẫn báo cáo trung thực.
