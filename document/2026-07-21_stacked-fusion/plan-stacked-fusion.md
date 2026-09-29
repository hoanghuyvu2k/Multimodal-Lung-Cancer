# Plan: Mô Hình Mới — Stacked Late-Fusion (bỏ attention)

> Ngày tạo: 2026-07-21
> Mục tiêu: đề xuất một **cập nhật mô hình cho paper** thay thế cơ chế attention, dựa trên bằng chứng
> hai dự án trước. Thắng ở đúng trục có đòn bẩy: **diễn giải + generalization + calibration**, không
> chạy đua AUC nội bộ (đã kịch trần ~0.77 trên cohort này).

## 0. Bối cảnh — vì sao Stacked Late-Fusion

Hai dự án trước đã khoá:
- **Fusion attention đã cạn** (`2026-07-20_attention-redesign`): attention trơ, giá trị chỉ là chuẩn hoá
  `1/Σmask`; `uniform_avg` ≈ mọi biến thể; neural ≈ LR.
- **Không có đòn bẩy feature** (`2026-07-21_feature-validation`); và phát hiện chính: **pathology
  generalize (ext 0.767), radiomics KHÔNG (≤0.46)** — mô hình hiện tại cân trọng số theo tín hiệu
  *in-sample* nên overweight radiomics (tốt trên discovery-CV nhưng vô dụng ngoài mẫu).

**Ý tưởng mô hình mới:** thay attention bằng **stacked generalization (late-fusion xếp chồng)**:
```
mỗi modality m -> base learner LR -> dự đoán OUT-OF-FOLD (OOF, nested CV)
meta-LR học trọng số modality TỪ dự đoán OOF (không phải in-fold)
score = meta-LR(base_preds)  [+ seed-ensemble + calibration]
```
Vì meta-LR học trên **OOF** chứ không in-fold, nó **tự động hạ trọng số modality không generalize**
(radiomics OOF không nhất quán) và tin modality ổn định (pathology). Đây là cách "mã hoá" phát hiện
chính vào chính mô hình — và cho **trọng số modality trung thực, diễn giải được** thay cho attention trơ.

## 1. Nguyên tắc thực thi (kế thừa)

- Mỗi mô hình mới = một folder độc lập dưới `experiments/`. **KHÔNG sửa `lung_helpers.py`.**
  Folder mô hình này: `experiments/stacked_fusion/`.
- Tái dùng harness: `common/{data_setup,evaluate,benchmarks}.py`, `baselines/model_lr.py`.
- Kỷ luật token: mỗi Step một lượt, ghi kết quả ra `results/*.json`, in bảng ngắn, log dài vào scratchpad.
  Đặt `PYTHONIOENCODING=utf-8`. **Chạy background job với đường dẫn tuyệt đối / `cd` về code root** (bài
  học: cwd trôi làm sai đường dẫn).

### 1.1. Benchmark cố định + số cần đối chiếu
| BM | Config | uniform_avg (nền, 5-seed) | uniform_avg **ensemble** |
|---|---|---|---|
| BM1 | Rad+IHC-G+Gen+PDL1 | 0.7746 ± 0.0186 | 0.7850 |
| BM2 | +Labs | 0.7665 ± 0.0110 | 0.7744 |
| BM3 | Rad+Gen | 0.7110 ± 0.0148 | 0.7255 |
| BM4 | PDL1+Gen | 0.7191 ± 0.0084 | 0.7242 |

### 1.2. Tiêu chí GIỮ / XÓA (khác dự án feature — đây là mô hình, không phải "cải thiện feature")
Giá trị của mô hình mới nằm ở **diễn giải + robustness + không thua**, không phải +AUC (đã kịch trần).
- **GIỮ** nếu: (a) AUC nội bộ **KHÔNG thua có ý nghĩa** `uniform_avg` trên cả 4 BM (không config nào
  paired-bootstrap p<0.05 theo hướng stacked *thua*) **VÀ** (b) trọng số meta diễn giải được, hợp lý
  (hạ radiomics, giữ pathology) — tức cung cấp câu chuyện mà attention không có.
- **XÓA** nếu: thua `uniform_avg` có ý nghĩa ở ≥2/4 BM, hoặc trọng số meta vô nghĩa/không ổn định.
- Ghi rõ: nếu stacked NGANG uniform_avg nhưng **diễn giải tốt hơn** → vẫn GIỮ (đó là điểm bán cho paper).

## 2. Các bước

### Step 0 — Khung
- [x] Tạo `experiments/stacked_fusion/`; xác nhận harness + `results/backbone_lock.json` đọc được (nền BM1–4 = 0.7746/0.7665/0.7110/0.7191)
- [x] Ghi lại số nền cần vượt/không-thua (bảng 1.1) — đã có trong plan

### Step 1 — Dựng model Stacked Late-Fusion
- [ ] `experiments/stacked_fusion/model_stack.py`:
  - `train_stack(...)` chữ ký chuẩn `(modality_list, mask, outcomes, l1_filter, params, folds, seed)`
  - Outer 10-fold (như harness). Trong mỗi outer-train: **inner 5-fold** sinh dự đoán OOF cho từng
    modality (base = LR balanced; radiomics áp đúng L1 filter theo fold, modality vắng mặt → base pred 0
    sau z-score theo train). Refit base trên full outer-train → dự đoán outer-val.
  - Meta-LR (regularized) học trên OOF base-preds của outer-train → dự đoán trên val base-preds.
  - Trả `(summary_df, coef_df)`; `coef_df` lưu trọng số meta per modality mỗi fold (cho Step 3).
- [x] Smoke test BM1 1-seed: **AUC = 0.7756** (khớp ballpark uniform_avg 0.7746, không cao bất thường → không rò rỉ), chạy sạch không lỗi

### Step 2 — Đánh giá nội bộ đầy đủ
- [x] Chạy `train_stack` BM1–BM4 × 5 seeds × 10-fold → `results/stacked.json` + scores CSV (`run_stack.py`)
- [x] Paired bootstrap vs `uniform_avg`, `Original`, `OvO`, `lr_late` (dùng scores đã lưu)
- [x] Bảng mean±sd + Δ vs uniform_avg; **điều kiện (a) ĐẠT**: không BM nào stacked thua uniform_avg có ý nghĩa
  - BM1 0.7524 (Δ−0.019, p=0.42) · BM2 0.7479 (Δ−0.010, p=0.69) · BM3 0.7081 (Δ−0.001, p=0.95) · BM4 0.7422 (Δ+0.021, p=0.26)
  - stacked ≈ `lr_late` (Δ≤0.023, mọi p≥0.20) như dự đoán; BM4 stacked nhỉnh vs original (+0.033)

### Step 3 — Trọng số meta (câu chuyện diễn giải thay attention)
- [x] Trích trọng số meta-LR per modality, trung bình + sd qua 5 seed × 10 fold, cho BM1 (và BM2) (`run_weights.py`)
- [x] Kiểm chứng story — **GIẢ THUYẾT BỊ BÁC BỎ**: meta KHÔNG hạ radiomics. `rad_lesion_ln` coef **1.27 (cao nhất)**,
  radiomics mean|coef|=0.79 > non-rad 0.53 (BM1); non/rad = **0.67x** (BM1) / 0.54x (BM2). Tệ hơn: `path_ihc_glcm`
  — modality DUY NHẤT generalize ngoài mẫu (ext 0.767) — mean|coef|≈**0.17, gần 0** (mean hơi âm).
  Lý do: meta học từ OOF *trong discovery*, nơi radiomics vẫn informative in-sample → tái lập đúng bias in-sample
  của attention cũ. Đòn bẩy "generalization" không tự mã hoá được vào trọng số vì meta không thấy cohort ngoài.
- [x] Ghi `results/stacked_weights.json` + nhận xét. **Điều kiện (b) KHÔNG đạt** → đe doạ luận điểm bán paper

### Step 4 — Seed-ensemble + calibration cho stacked
- [x] Ensemble điểm stacked qua 5 seed → AUC ensemble (`run_ensemble_stack.py`). Lợi ensemble +0.013–0.017
  (BM1 0.7658 / BM2 0.7640 / BM3 0.7248 / BM4 0.7448) — cùng mức với uniform_avg. Vẫn dưới uniform_avg-ens
  ở BM1/BM2, ngang BM3, hơn BM4. **Không thu hẹp được khoảng cách.**
- [x] Brier (Platt) cho stacked vs uniform_avg: **không lợi thế calibration** (BM1/BM2 hơi tệ hơn +0.007/+0.005,
  BM3/BM4 hơi tốt hơn −0.001/−0.010).
- [x] Cập nhật `results/stacked.json` (phần `ensemble`)

### Step 5 — Kiểm chứng ngoài mẫu (trung thực về giới hạn)
- [x] Nêu rõ giới hạn: cohort ngoài **đơn-modality** → stacked rút gọn về base learner → fusion đa-modality
      KHÔNG test được ngoài mẫu (`run_external_note.py`)
- [x] External base-LR: pathology **0.765**, radiomics **0.461**. Lập luận **ĐẢO NGƯỢC** kỳ vọng ban đầu:
      meta đề cao đúng radiomics (fail ngoài mẫu) và hạ pathology (generalize) → trọng số stacked **phản-robust**,
      KHÔNG khiến mô hình bền hơn với domain shift.
- [x] Ghi vào `results/stacked.json` phần `external_note`

### Step 6 — Tổng hợp
- [x] Bảng cuối: uniform_avg | uniform_avg-ens | Original | OvO | lr_late | **stacked** | **stacked-ens**, BM1–BM4 (`run_final.py`)
- [x] Hình: (a) `fig-forest-delta.svg` ΔAUC(stacked − uniform_avg) với CI (mọi CI phủ 0); (b) `fig-meta-weights.svg` bar trọng số meta (radiomics cam cao nhất, pathology xanh ≈0)
- [x] Áp tiêu chí 1.2 → **(a) ĐẠT nhưng (b) TRƯỢT → không GIỮ như phương pháp đề xuất; đóng gói NEGATIVE FINDING.** KHÔNG `rm -rf` (mã+kết quả là bằng chứng negative finding, giá trị cho Discussion)
- [x] Viết `document/2026-07-21_stacked-fusion/ket-qua.md`
- [x] Cập nhật Nhật ký; mọi step tick → **dừng loop**

---

## 3. Nhật ký kết quả
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-21 | 6 | Bảng cuối + 2 hình xong (`run_final.py`). Tiêu chí 1.2: (a) không thua uniform_avg ĐẠT; (b) trọng số meta diễn giải TRƯỢT (đề cao radiomics, bỏ pathology). `ket-qua.md` viết xong | **KẾT: negative finding.** Không đề xuất stacked trong paper; giữ mã làm bằng chứng (không xóa). **PLAN HOÀN TẤT — dừng loop** |
| 2026-07-21 | 5 | External base-LR pathology 0.765 / radiomics 0.461. Meta đề cao radiomics (fail) + hạ pathology (generalize) → trọng số phản-robust. Fusion đa-modality không test được ngoài mẫu (cohort đơn-modality) | Xong. Củng cố negative finding. Sang Step 6 tổng hợp + quyết định |
| 2026-07-21 | 4 | Seed-ensemble stacked +0.013–0.017 (cùng mức uniform_avg), vẫn dưới uniform_avg-ens BM1/BM2. Brier không lợi thế | Xong. Không cứu được từ ensemble/calibration. Sang Step 5 |
| 2026-07-21 | 3 | **Giả thuyết chính BỊ BÁC BỎ.** Meta-LR KHÔNG hạ radiomics: rad_lesion_ln coef 1.27 (cao nhất), radiomics mean\|coef\| 0.79 > non-rad 0.53 (non/rad 0.67x BM1, 0.54x BM2). Pathology (modality generalize duy nhất) coef ≈0. Meta học OOF in-sample nên tái lập bias attention cũ | **Điều kiện (b) trượt.** Story "hạ radiomics" sai. Vẫn chạy Step 4–5 lấy đủ ngữ cảnh, nhưng Step 6 nghiêng về XÓA/viết thành negative finding |
| 2026-07-21 | 2 | **Đánh giá đầy đủ (5 seed) xong** (`run_stack.py` → `results/stacked.json` + scores). AUC: BM1 0.7524 / BM2 0.7479 / BM3 0.7081 / BM4 0.7422. Δ vs uniform_avg đều KHÔNG có ý nghĩa (p 0.42/0.69/0.95/0.26); stacked ≈ lr_late. **Điều kiện (a) ĐẠT** (không thua ở BM nào) | Sang Step 3 (trọng số meta = điều kiện b, điểm bán paper) |
| 2026-07-21 | 1 | **model_stack.py dựng xong** (stacked nested-CV: L1 fit outer-train, OOF qua inner 5-fold, meta-LR trên OOF, trọng số meta lưu vào coef_df). Smoke BM1 1-seed = **0.7756**, khớp ballpark uniform_avg, không rò rỉ | Xong, sang Step 2 (đánh giá đầy đủ) |
| 2026-07-21 | 0 | Khung xong: `experiments/stacked_fusion/` tạo, backbone đọc được (nền BM1–4 = 0.7746/0.7665/0.7110/0.7191). | Xong, sang Step 1 (dựng model_stack.py) |

---

## 4. Ghi chú kỹ thuật
- Base learner = LR (`class_weight='balanced'`, L2, C=1.0) — sạch, diễn giải, và neural≈LR nên không mất gì.
- Chống rò rỉ: OOF base-preds sinh bằng inner CV TRONG outer-train; base refit trên full outer-train mới
  chạm outer-val. L1 filter radiomics fit theo từng (outer/inner) train, không nhìn val.
- Modality vắng mặt: base-pred = 0 sau z-score theo train (tương đương mask triệt tiêu trong uniform_avg);
  meta-LR có intercept nên xử lý được.
- Chuẩn hoá base-preds: z-score mỗi modality theo phân bố OOF trên train trước khi vào meta (như `lr_late`).
- `seed` của lung_helpers vô hiệu → `run_repeated_cv` hoán vị hàng outcomes theo seed (đã có trong harness).
- Đối chiếu số nền: `results/backbone_lock.json`, scores tại `results/scores/{uniform_avg,original,ovo,lr_late}__BM*.csv`.
- External đơn-modality: `results/validation.json` (pathology 0.767/0.765, radiomics 0.425/0.461).
