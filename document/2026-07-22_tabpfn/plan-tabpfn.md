# Plan: TabPFN (tabular foundation model) cho fusion đa nguồn n-nhỏ

> Ngày tạo: 2026-07-22
> Ý tưởng: bài toán = tabular đa nguồn, **n nhỏ** — đúng sân nhà TabPFN (in-context Bayesian, mạnh dữ liệu nhỏ).
> Kiểm chứng: TabPFN có vượt LR/uniform_avg (per-modality, external, và trong fusion) không?
> Kỳ vọng trung thực: literature cho thấy TabPFN *thường ngang* ML mạnh (±0.01) — nhưng là comparator hiện đại
> reviewer mong đợi, và có thể nhỉnh trên modality nhỏ.

## 0. Bối cảnh

- Đã khoá: neural≈LR, attention=uniform_avg, embedding không giúp. Trần AUC ~0.78 nội bộ.
- TabPFN v2 (Nature 2025): pretrained trên synthetic, fit(X,y) không train thật, mạnh n nhỏ, cap ~500 feature.
- Dữ liệu: n 150–250, modality dim khác nhau (pdl1=1, gen nhỏ, labs=13, glcm~150, radiomics~30 sau lọc).

**Câu hỏi:** (a) per-modality TabPFN > LR? (b) external TabPFN generalize tốt hơn? (c) TabPFN late-fusion > uniform_avg?

## 1. Nguyên tắc
- Mã `experiments/tabpfn/`. KHÔNG sửa lung_helpers. Kết quả `results/tabpfn_*.json`. Mỗi Step một lượt loop.
- Cap feature cao chiều (glcm, radiomics) bằng SelectKBest~50 IN-FOLD (chống rò rỉ) cho công bằng.
- TabPFN chạy CPU được (n nhỏ) → không cần GPU nặng. utf-8, path tuyệt đối.

## 2. Các bước

### Step 0 — Setup + smoke
- [x] tabpfn v8.1.0 **bắt buộc login (TABPFN_TOKEN)** → downgrade **tabpfn 2.2.1** (bản local, không login, tải weights HF)
- [x] Package đổi tên `tabpfn_exp/` (tránh đụng tên `tabpfn`). Smoke PDL1: **TabPFN 0.7101 vs LR 0.7186** (~43s/modality)

### Step 1 — Per-modality: TabPFN vs LR (discovery-CV)
- [x] 5 modality xong. **TabPFN thắng LR ở 0/5**: PDL1 0.710 vs 0.719, Gen 0.656 vs 0.676, GLCM 0.547 vs 0.576,
  Labs 0.560 vs 0.594, radiomics 0.579 vs 0.576 (chỉ ngang). Ghi `results/tabpfn_single.json`

### Step 2 — External: TabPFN vs LR
- [x] pathology: LR **0.767** vs TabPFN 0.748 (thấp hơn); radiomics: LR 0.436 vs TabPFN 0.481 (cả hai ~ngẫu nhiên, CI khổng lồ). TabPFN không generalize hơn. Ghi `tabpfn_external.json`

### Step 3 — Fusion: TabPFN late-fusion vs uniform_avg (BM1–4)
- [x] TabPFN-fusion: BM1 0.715 / BM2 0.719 / BM3 0.667 / BM4 0.743. **Thắng uniform_avg chỉ 1/4** (BM4 +0.024,
  combo 2-modality nhỏ); thua rõ BM1/BM2/BM3 (−0.04..−0.06, do radiomics OOF thêm nhiễu). Ghi `tabpfn_fusion.json`

### Step 4 — Tổng hợp
- [x] Hình `fig-tabpfn.svg` (per-modality + fusion) + `ket-qua.md`. **Quyết định: TabPFN ngang/không hơn LR/uniform_avg
  (thắng chỉ BM4 nhỏ); GIỮ làm comparator hiện đại cho paper, không phải cải tiến.** **PLAN HOÀN TẤT — dừng loop.**

## 3. Nhật ký
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-22 | 1-4 | TabPFN: per-modality 0/5 thắng LR; external không hơn (path 0.748 vs 0.767); fusion 1/4 (chỉ BM4 +0.024). Hình + ket-qua xong | **Không hơn baseline. GIỮ làm comparator hiện đại. Plan xong, dừng loop** |
| 2026-07-22 | 0 | tabpfn v8 bắt buộc login → downgrade v2.2.1. Smoke PDL1 OK | Sang Step 1 |

## 4. Ghi chú
- TabPFN v2 cap ~10k mẫu / ~500 feature — dữ liệu ta thừa sức. Cap feature ~50 cho glcm/radiomics.
- Tiêu chí GIỮ: TabPFN vượt LR/uniform có ý nghĩa ở ≥1 trục (nhất là external/n-nhỏ). Nếu chỉ ngang → vẫn báo cáo làm comparator hiện đại.
