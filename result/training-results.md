# Kết Quả Training — Tổng Hợp Toàn Bộ Dự Án

> File này gom **tất cả kết quả training/thực nghiệm** đã chạy được từ đầu dự án đến nay (tính đến
> 2026-08-19), kèm **cấu hình training** dùng cho từng lần chạy. Đây là log tra cứu nhanh — chi tiết
> đầy đủ (phương pháp, hình vẽ, mã nguồn) nằm ở các file `document/*/ket-qua.md` được dẫn nguồn trong
> từng mục.
>
> **Quy tắc cập nhật (xem cuối file):** từ nay, sau khi chạy training xong, kết quả (AUC/C-index/config)
> phải được thêm vào file này.

---

## 0. Cấu hình training chuẩn (harness dùng xuyên suốt từ 2026-07-20 trở đi)

Nguồn: `experiments/common/benchmarks.py`

```python
MODEL_PARAMS = {
    "epochs": 125,
    "lr": 0.01,
    "alpha": 0.001,
    "beta": 0.0,
    "cross_modality_enabled": False,
}
SEEDS = [42, 7, 123, 2024, 31337]
FOLDS = 10
```

- Đánh giá chuẩn: **5 seeds × 10-fold CV**, seed-permutation (vì `train()`/`train_ovo()` có bug: tham số
  `seed` không có tác dụng — `KFold` hardcode `random_state=0`, `torch.manual_seed(42)` trong `__init__`;
  harness né bằng cách hoán vị hàng `outcomes` theo seed).
- Baseline không-attention: `uniform_avg` (train_uniform_avg) — trung bình có mask `1/Σmask`, không tham số học.
- 4 benchmark cố định (BM1–BM4), dùng cho **mọi** phương pháp để so sánh nhất quán:

| BM | Tên | Modalities | `use_rad_filters` |
|---|---|---|---|
| **BM1** (primary) | Rad+IHC-G+Gen+PDL1 | `rad_lesion_{pc,pl,ln}` + `path_ihc_glcm` + `gen_driver_mut_amp` + `cnl_pdl1_score` | True |
| **BM2** | +Labs | BM1 + `cnl_dem_labs` | True |
| **BM3** | Rad+Gen | `rad_lesion_{pc,pl,ln}` + `gen_driver_mut_amp` | True |
| **BM4** | PDL1+Gen | `cnl_pdl1_score` + `gen_driver_mut_amp` | False |

- Dataset: Discovery cohort, `n=247`, label `1=SD/POD` (185, 75%) / `0=PR/CR` (62, 25%), event rate PFS 84.6%.
- Cohort ngoài (external): `rad_valid` (n=50, chỉ radiomics), `path_valid` (n=71, chỉ pathology) — không cohort nào đủ modality để test fusion ngoài mẫu, chỉ per-modality.
- Baseline LR (dùng trong so sánh): `LogisticRegression(C=1.0, class_weight='balanced')` — `lr_concat` (gộp feature) và `lr_late` (trung bình điểm per-modality).

**Backbone khoá (5-seed, dùng làm mốc so sánh cho mọi thực nghiệm sau):**

| BM | uniform_avg mean±sd | uniform_avg + seed-ensemble |
|---|---|---|
| BM1 | 0.7746 ± 0.019 | **0.7850** |
| BM2 | 0.7665 ± 0.011 | 0.7744 |
| BM3 | 0.7110 ± 0.015 | 0.7255 |
| BM4 | 0.7191 ± 0.008 | 0.7242 |

Nguồn: `experiments/results/backbone_lock.json`.

---

## 1. Dòng thời gian các dự án

| # | Dự án | Ngày | Folder tài liệu | Kết luận |
|---|---|---|---|---|
| 0 | Base model + OvO attention (gốc, single-seed) | trước 2026-06 | `document/ovo-model/ovo-complete.md`, `document/nlp-model/nlp-ket-qua.md` | Headline gốc AUC 0.7839(Original)/0.8003(OvO) trên BM1 — **về sau xác định là ảo giác chọn seed** |
| 1 | Survival analysis (NLP extension) | 2026-06-08/09 | `document/2026-06-08_pathology-pdl1-glcm/survival-analysis-plan.md` | C-index không significant nhưng Cox HR + tdAUC ủng hộ NLP |
| 2 | Attention redesign | 2026-07-20 | `document/2026-07-20_attention-redesign/ket-qua.md` | Không phương pháp attention mới nào thắng; `uniform_avg` ≈ mọi biến thể; attention gốc **trơ**, giá trị chỉ ở chuẩn hoá `1/Σmask` |
| 3 | Feature quality & external validation | 2026-07-21 | `document/2026-07-21_feature-validation/ket-qua.md` | Feature engineering vô ích; **pathology generalize (ext 0.767), radiomics không (≤0.46)** |
| 4 | Stacked late-fusion | 2026-07-21 | `document/2026-07-21_stacked-fusion/ket-qua.md` | Ngang `uniform_avg` nhưng trọng số meta **sai lệch** (đề cao radiomics, bỏ pathology) → không đề xuất |
| 5 | uniform_avg vs DyAM — 21 tổ hợp bài báo | 2026-07-21 | `document/2026-07-21_uniform-dyam-allcombo/ket-qua.md`, `bang-so-sanh-paper-combos.md` | `uniform_avg` ≥ DyAM ở đa số tổ hợp đa nguồn; DyAM chỉ thắng 2/26 (ít nguồn) |
| 6 | Foundation embedding (pathology + CT) | 2026-07-21/22 | `document/2026-07-21_foundation-embedding/`, `document/2026-07-22_ct-foundation-embedding/` | ÂM toàn bộ — Phikon không vượt GLCM external; BiomedCLIP/MedicalNet/fmcib CT đều ≈ ngẫu nhiên (~0.51) nội bộ |
| 7 | TabPFN | 2026-07-22 | `document/2026-07-22_tabpfn/ket-qua.md` | Không vượt LR/uniform_avg: 0/5 per-modality, 1/4 fusion (chỉ BM4) |
| 8 | **NLP-clinical verify** | 2026-07-22 | `document/2026-07-22_nlp-verify/ket-qua.md` | ★ **DƯƠNG DUY NHẤT** — +NLP > +Labs ở 19/21 tổ hợp; BM1 5-seed 0.7832, single-run 0.8091 |
| 9 | OvO + NLP-clinical, đủ 21 combo | 2026-08-19 | `result/training-results.md` §2.9 | Xác nhận NLP dương với backbone OvO (15/21 combo, khớp pattern §8); OvO vẫn ≈ uniform_avg (BM1 Δ=−0.001), không có lợi thế kiến trúc thật |

---

## 2. Chi tiết kết quả theo dự án

### 2.0. Base model gốc — OvO vs Original (single-seed, KFold random_state=0)

**Script:** `compare_ovo_attention_full.py` · **Nguồn:** `excel/ovo_comparison_full_results.xlsx`, `document/ovo-model/ovo-complete.md`
**Config:** `train()`/`train_ovo()` mặc định thư viện (epochs không cố định 125 ở giai đoạn này), `folds=10`, KFold `random_state=0` (1 phân hoạch cố định, không phải 5-seed).

- 20 test case, so `AttentionMatrix` (Original) vs `AttentionMatrixOvO`.
- OvO tốt hơn: 7/20 (35%) · Original tốt hơn: 9/20 (45%) · bằng nhau: 4/20 (20%, đơn-modality).
- **Tốt nhất:** OvO `Rad+IHC-G+Gen+PDL1` (=BM1) = AUC **0.8003** [0.739–0.862]; Original tốt nhất `Rad+IHC-G+Gen+PDL1+Labs` (=BM2) = 0.7879 [0.723–0.853].
- Top OvO thắng: PDL1+Gen +3.27%, Rad+Gen +2.23%, Rad+IHC-G+Gen+PDL1 +2.10%.
- Top Original thắng: Rad+IHC-A+Gen −4.94% (OvO thua), IHC-G+Gen −4.33%.

> **Lưu ý quan trọng (xác nhận lại ở dự án #2):** con số 0.8003/0.7839 này là kết quả **1 phân hoạch CV cố định** (`KFold(random_state=0)`), KHÔNG phải trung bình nhiều seed. Khi đánh giá lại bằng 5-seed × 10-fold (dự án #2), BM1 Original tụt còn 0.7595±0.022, OvO 0.7728±0.020 — chênh giữa 2 model trong nhiễu, không còn +2.1%.

### 2.0b. NLP-clinical (bản gốc, trước verify) — 7 biến thể trên BM1/BM1'(IHC-A)

**Script:** notebook gốc (`Figures-Finalized*.ipynb`) · **Nguồn:** `document/nlp-model/nlp-ket-qua.md`
**Config:** KFold 10-fold, n=247, `all-MiniLM-L6-v2` cho NLP raw (384d), PCA→16d cho NLP-PCA16, BioClinBERT-PCA16 làm đối chứng.

| Model | Arm | AUC | 95% CI DeLong |
|---|---|---|---|
| DyAM No Clinical | IHC-A | 0.764 | [0.695–0.833] |
| + Labs | IHC-A | 0.768 | [0.700–0.837] |
| + NLP raw | IHC-A | 0.781 | [0.717–0.845] |
| **+ NLP-PCA16** | IHC-A | **0.784** | [0.719–0.848] |
| + Labs+NLP combined | IHC-A | 0.767 | [0.700–0.835] |
| + BioClinBERT-PCA16 | IHC-A | 0.767 | [0.701–0.833] |
| DyAM No Clinical | IHC-G | 0.784 | [0.717–0.850] |
| + Labs | IHC-G | 0.788 | [0.723–0.853] |
| **+ NLP raw** | IHC-G | **0.813** | [0.753–0.874] |
| + NLP-PCA16 | IHC-G | 0.812 | [0.751–0.873] |

- Không có cặp nào có CI DeLong không chồng nhau → không significant ở giai đoạn này (n=247 nhỏ).
- KM log-rank: tất cả model p<0.005 (IHC-G +NLP-PCA16 χ²=28.92 cao nhất).
- Single-modality baseline: LR Clinical 0.570, LR Rad-PC 0.640, **LR PDL1-TPS 0.729** (tốt nhất đơn-modality), LR Gen 0.650, NLP-only 0.539, Labs-only 0.594.

### 2.1. Survival analysis — NLP-clinical (2026-06-08/09)

**Script:** `survival_analysis_nlp.py` · **Output:** `excel/survival_analysis_summary.xlsx`, `excel/external_validation_cindex.xlsx`
**Config:** dùng lại `summary_df` (score) từ DyAM 7 biến thể ở §2.0b; primary endpoint PFS (n=247, event rate 84.6%, median PFS 2.7m, range 0.1–49.1m); C-index Harrell qua `lifelines`, bootstrap 1000 lần; tdAUC (`sksurv`) tại 6/12/18m; Cox đa biến adjust `age, ecog, albumin, dnlr, liver_mets` (penalizer=0.1); IBS (`sksurv`).

| Model | AUC binary | C-index [95% CI boot] | tdAUC 6m/12m/18m | Cox HR [95% CI] | Cox p | IBS |
|---|---|---|---|---|---|---|
| No Clinical (IHC-A) | 0.776 | 0.623 [0.580–0.665] | 0.717/0.718/0.700 | 5.30 [3.10–9.07] | <0.001 | 0.1745 |
| + Labs (IHC-A) | 0.754 | 0.626 [0.583–0.669] | 0.713/0.707/0.662 | 5.31 [2.88–9.79] | <0.001 | 0.1739 |
| + NLP raw (IHC-A) | 0.774 | 0.625 [0.583–0.663] | 0.715/0.739/0.718 | 5.91 [3.40–10.25] | <0.001 | 0.1723 |
| **+ NLP-PCA16 (IHC-A)** | 0.773 | **0.628** [0.584–0.668] | 0.719/**0.742/0.720** | **6.07** [3.50–10.52] | <0.001 | **0.1716** |
| No Clinical (IHC-G) | 0.793 | 0.626 [0.583–0.665] | 0.722/0.691/0.667 | 4.74 [2.82–7.97] | <0.001 | 0.1748 |
| + Labs (IHC-G) | 0.776 | 0.632 [0.587–0.675] | 0.711/0.686/0.649 | 4.49 [2.56–7.89] | <0.001 | 0.1746 |
| **+ NLP raw (IHC-G)** | **0.797** | 0.632 [0.591–0.672] | 0.720/0.699/0.665 | 4.97 [2.98–8.31] | <0.001 | 0.1736 |
| + NLP-PCA16 (IHC-G) | 0.796 | 0.631 [0.590–0.672] | 0.721/0.694/0.656 | 4.69 [2.86–7.68] | <0.001 | 0.1736 |

- **Wilcoxon paired C-index (No Clinical vs NLP-PCA16, per-fold):** IHC-A ΔC=+0.005 p=0.492; IHC-G ΔC=+0.005 p=0.375 — **không significant.**
- **Cox HR là kết quả mạnh nhất:** NLP-PCA16 IHC-A cho HR cao nhất (6.07 vs 5.30 baseline); tất cả 8 model là independent prognostic factor (p<0.001).
- **tdAUC:** lợi thế NLP rõ nhất ở 12m/18m (long-term), không phải 6m.
- **External validation (train discovery → test):**

| Model | Cohort | n | C-index [CI] | AUC binary | ΔC vs imaging-only |
|---|---|---|---|---|---|
| IHC-A only | path_valid | 52 | 0.533 [0.422–0.634] | 0.598 | ref |
| IHC-A + NLP-PCA16 | path_valid | 52 | 0.544 [0.447–0.644] | 0.622 | +0.011 |
| **IHC-G only** | path_valid | 52 | **0.618** [0.514–0.709] | **0.767** | ref |
| IHC-G + NLP-PCA16 | path_valid | 52 | 0.593 [0.480–0.703] | 0.705 | −0.025 |
| Rad only | rad_valid | 46 | 0.537 [0.441–0.634] | 0.536 | ref |
| Rad + NLP-PCA16 | rad_valid | 46 | 0.528 [0.419–0.628] | 0.447 | −0.009 |

- **IHC-G (GLCM) generalize tốt nhất** — gap internal→external chỉ −0.008 (0.626→0.618); AUC external 0.767 là con số mạnh nhất toàn bộ external validation.
- NLP effect ngoài mẫu **không nhất quán** (mixed: +0.011/−0.025/−0.009) — khác với lợi thế nội bộ.

### 2.2. Attention redesign (2026-07-20)

**Mã:** `experiments/{baselines,audit}`, method A/B (đã xoá folder) · **Kết quả:** `experiments/results/{baseline,method_A,method_B,baseline_lr,ablation,diagnostics}.json`
**Config:** MODEL_PARAMS chuẩn (§0), 5 seeds × 10-fold, paired bootstrap 2000 lần trên bệnh nhân.

Bảng kết quả cuối (mean±sd AUC, 5-seed):

| BM | Original | OvO | Gated A | Uncertainty B | Uniform avg | LR concat | LR late |
|---|---|---|---|---|---|---|---|
| BM1 | 0.7595±0.0221 | 0.7728±0.0197 | 0.7330±0.0173 | **0.7792±0.0255** | 0.7746±0.0186 | 0.7276±0.0179 | 0.7615±0.0102 |
| BM2 | 0.7638±0.0179 | 0.7632±0.0097 | 0.7288±0.0260 | 0.7718±0.0073 | 0.7665±0.0110 | 0.7112±0.0195 | 0.7556±0.0054 |
| BM3 | 0.7095±0.0177 | 0.7132±0.0150 | 0.6504±0.0226 | 0.7101±0.0138 | 0.7110±0.0148 | 0.6908±0.0087 | 0.6968±0.0083 |
| BM4 | 0.7072±0.0088 | 0.7183±0.0115 | 0.7317±0.0054 | 0.7262±0.0110 | 0.7191±0.0084 | 0.7472±0.0110 | **0.7515±0.0084** |

- Tiêu chí GIỮ (Δ≥+0.015 trên ≥3/4 BM, p<0.10 BM1) — **không phương pháp mới nào đạt** → Gated A, Uncertainty B đã **XOÁ**.
- OvO vs Original: |Δ| ≤ 0.011, p ≥ 0.17 ở cả 4 BM → không khác biệt hệ thống.
- `uniform_avg` (hằng số `1/Σmask`, không tham số) ngang hoặc hơn Original/OvO ở mọi BM.
- `cross_modality_enabled=True` trơ (≤+0.0034, p≥0.53) → khuyến nghị luôn để `False`.
- Kết quả p<0.05 duy nhất: `lr_late` vs Original trên BM4 (+0.0393, p=0.045) — nghiêng về LR, không phải neural.

### 2.3. Feature quality & external validation (2026-07-21)

**Mã:** `experiments/{audit,validation,ensemble}` · **Kết quả:** `results/{backbone_lock,signal_audit,feat_select,feat_reduce,nlp_modality,validation,ensemble}.json`
**Config:** backbone `uniform_avg` cố định, MODEL_PARAMS chuẩn, 5-seed×10-fold nội bộ + train-discovery/test-external.

**Feature engineering (đều XOÁ, không đạt tiêu chí Δ≥+0.015 & p<0.10):**

| Phương pháp | Config | Kết quả |
|---|---|---|
| Feature selection (elastic-net sweep + MI top-k) | MI top-k=20 | Winner=config mặc định (Δ=0); MI top-20 hại BM1 −0.050 (p=0.01) |
| Dimensionality reduction | PCA-95%, PLS-5 (fit trong fold) | Tệ hơn mọi config có radiomics (Δ −0.03…−0.055) |
| NLP clinical (thay Labs bằng 384-dim MiniLM) | no_scale | BM2 +0.018 (p=0.157, không significant); NLP đơn-modality 0.549 < Labs thô 0.573 |

**External validation (per-modality, LR vs head neural):**

| Modality | disc-CV uniform | disc-CV LR | **ext neural** [CI] | **ext LR** [CI] | n_test |
|---|---|---|---|---|---|
| **pathology** → path_valid | 0.622 | 0.631 | **0.767** [0.62–0.89] | **0.765** [0.61–0.89] | 52 |
| **radiomics** → rad_valid | 0.587 | 0.598 | **0.425** [0.20–0.65] | **0.461** [0.23–0.68] | 46 |

- Pathology generalize mạnh (external > discovery-CV); radiomics sụp về ~0.42–0.46 (dưới ngẫu nhiên).
- **Ensemble qua 5 seed** (trung bình điểm per-patient): BM1 uniform_avg 0.7746 → **0.7850** (+0.010), toàn bộ +0.005…+0.016, Brier ~0.15. Cải thiện rẻ duy nhất được xác nhận.

### 2.4. Stacked late-fusion (2026-07-21)

**Mã:** `experiments/stacked_fusion/{model_stack,run_stack,run_weights,run_ensemble_stack,run_prune_weight}.py`
**Kết quả:** `results/{stacked,stacked_weights,prune_weight}.json`
**Config:** base learner = LR (`class_weight='balanced'`, L2, C=1.0); OOF qua inner 5-fold trong outer-train; meta = LR; MODEL_PARAMS chuẩn cho phần so sánh với Original/OvO/DyAM.

| method | BM1 | BM2 | BM3 | BM4 | ens BM1/2/3/4 |
|---|---|---|---|---|---|
| uniform_avg (nền) | 0.775±0.019 | 0.767±0.011 | 0.711±0.015 | 0.719±0.008 | 0.785/0.774/0.726/0.724 |
| original (attn) | 0.760±0.022 | 0.764±0.018 | 0.709±0.018 | 0.707±0.009 | 0.775/0.776/0.726/0.712 |
| OvO (attn) | 0.773±0.020 | 0.763±0.010 | 0.713±0.015 | 0.718±0.012 | 0.784/0.771/0.726/0.722 |
| lr_late | 0.762±0.010 | 0.756±0.005 | 0.697±0.008 | 0.751±0.008 | 0.768/0.762/0.702/0.751 |
| **stacked** | 0.752±0.015 | 0.748±0.010 | 0.708±0.012 | 0.742±0.007 | 0.766/0.764/0.725/0.745 |

- Paired bootstrap stacked − uniform_avg: mọi BM CI phủ 0 → không khác significant (BM1 Δ−0.019 p=0.42, BM2 Δ−0.010 p=0.69, BM3 Δ−0.001 p=0.95, BM4 Δ+0.021 p=0.26). Điều kiện (a) "không thua" **ĐẠT**.
- **Trọng số meta (BM1, mean|coef|):** `rad_lesion_ln` 1.27 (cao nhất) > `cnl_pdl1_score` 0.77 > `rad_lesion_pc` 0.71 > `gen_driver_mut_amp` 0.64 > `rad_lesion_pl` 0.40 > **`path_ihc_glcm` 0.17 (thấp nhất — modality generalize duy nhất lại bị bỏ)**. Giả thuyết "meta hạ radiomics" **bị bác bỏ** → điều kiện (b) TRƯỢT → **không đề xuất**, giữ làm negative finding.
- Phụ lục prune modality nhiễu: bỏ `rad_lesion_pl` (n=21) −0.006; bỏ hết radiomics (n=3) −0.037 (gần p<0.05, p=0.073) → **không giúp**.
- Phụ lục trọng số theo AUC (`w=AUC−0.5`) cho lr_late: 0.7651→0.7747 (+0.0096) nhưng vẫn dưới ens uniform_avg 0.7850 (caveat: rò rỉ in-sample nhẹ vì trọng số lấy từ `signal_audit` cùng cohort).

### 2.5. uniform_avg vs DyAM — toàn bộ tổ hợp (2026-07-21)

**Mã:** `experiments/allcombo/{run_combo,run_analyze,run_paper_combos,run_paper_singlerun,run_paper_ovo}.py`
**Kết quả:** `results/{allcombo,allcombo_analysis,paper_combos}.json`
**Config:** MODEL_PARAMS chuẩn, 5-seed×10-fold (seed-permutation), paired bootstrap.

**A. 26 tổ hợp ≥2 nguồn (5 domain: Rad/Path/Gen/PDL1/Labs):**

| k (số nguồn) | #combo | mean Δ(uni−DyAM) | uniform thắng (p<.05) | DyAM thắng (p<.05) |
|---|---|---|---|---|
| 2 | 10 | +0.0006 | 1 (RD) | **2 (PG, RL)** |
| 3 | 10 | +0.0067 | 1 (RPD) | 0 |
| 4 | 5 | +0.0110 | 1 (PGDL) | 0 |
| 5 | 1 | +0.0027 | 0 | 0 |

→ Δ tăng theo k: DyAM chỉ có lợi thế khi ít nguồn nhất (k=2); từ k≥3 DyAM không thắng combo nào.

**B. 21 tổ hợp giống bài báo gốc (`Figures-Finalized.ipynb` cell 18)** — 5-seed vs single-run:

| # | Tổ hợp | #mod | uni 5s | DyAM 5s | uni 1run | DyAM 1run | ref bài báo |
|---|---|---|---|---|---|---|---|
| 8 | PDL1+Gen (BM4) | 2 | 0.7191 | 0.7072 | 0.7124 | **0.6932** | 0.6931 |
| 11 | Rad+Gen (BM3) | 4 | 0.7110 | 0.7095 | 0.7472 | **0.7549** | 0.7384 |
| 18 | Rad+IHC-G+Gen+PDL1 (BM1) | 6 | **0.7746** | 0.7595 | 0.7915 | **0.7976** | 0.7839 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | 7 | **0.7665** | 0.7638 | 0.7784 | **0.7908** | 0.7879 |

- Reproduce khớp đúng 4 benchmark khoá (BM1–BM4).
- Đơn modality: uniform_avg = DyAM hệt nhau (attention tầm thường khi 1 nguồn).
- Tổ hợp lớn (5–7 modality, #14–21): uniform_avg nhỉnh mean 7/8, DyAM không thắng cái nào.
- **Thứ hạng ĐẢO giữa single-run và 5-seed** ở combo lớn: single-run thiên vị DyAM (7/8 combo ≥5 modality DyAM>uniform), 5-seed thì ngược lại → **con số 0.79 gốc trong bài báo là 1 phân hoạch thuận lợi cho DyAM, không phải bằng chứng model tốt hơn.**
- **3-model đồng cấu hình (uniform/DyAM/OvO):** OvO ≈ uniform_avg ở 5-seed (bám sát ±0.005), cả hai ≥ DyAM. → OvO về bản chất gần uniform, không phải cải tiến thật.
- DyAM có 2 lợi thế thật (p<0.05): **Rad** (radiomics 3 lesion, p=0.017) và **IHC-G+Gen** (p=0.010) — cả hai đều ít-nguồn, mất lợi thế khi thêm nguồn thứ 3.

### 2.6. Foundation embedding — pathology & CT (2026-07-21/22)

**Mã:** `experiments/foundation_embed/{*, ct_*, ct3d_*, fmcib_eval, fm_fusion_compare}.py`
**Kết quả:** `results/{fm_pathology,fm_ct,fm_ct3d,fmcib_eval,fm_fusion}.json`
**Config:** đánh giá bằng LR sạch (không phải DyAM) cho so sánh embedding vs feature thủ công, discovery-CV 5-seed + external (train discovery→test).

**Single-modality (discovery-CV | external):**

| Model | domain | nội bộ | external | so hand-crafted |
|---|---|---|---|---|
| Phikon (ViT-B, mean-pool, 768d) | Pathology | **0.680±0.018** | 0.697 [0.53,0.85] | thua GLCM 0.765 external |
| GLCM thủ công | Pathology | 0.630±0.017 | **0.765** [0.61,0.89] | — |
| BiomedCLIP 2.5D (512d) | Radiology CT | 0.510±0.019 | 0.519/0.530 [0.35,0.72] | ≈ ngẫu nhiên |
| MedicalNet 3D ResNet50 (2048d, MONAI) | Radiology CT | 0.514±0.027 | 0.627 [0.44,0.81] (nhiễu) | ≈ ngẫu nhiên |
| **fmcib thật** (SimCLR cancer-CT, 4096d, chạy trên Colab) | Radiology CT | **0.503** (ngẫu nhiên) | 0.546 [0.33,0.76] | ≈ ngẫu nhiên |
| radiomics thủ công (SelectKBest 30) | Radiology CT | 0.582±0.024 | 0.461 [0.23,0.68] | — |

**Fusion (uniform_avg, discovery-CV 5-seed, BM1 = A):**

| config | uniform_avg | DyAM |
|---|---|---|
| A — thủ công (BM1) | **0.7746±0.019** | 0.7595±0.022 |
| B — embed cả 2 (Phikon PCA-64 + MedicalNet PCA-64) | 0.6630±0.023 | 0.6394±0.020 |
| C — embed pathology, rad thủ công | 0.7531±0.014 | 0.7468±0.015 |
| D — embed radiology, path thủ công | 0.7192±0.014 | 0.7068±0.016 |
| E — BM1 **+ fmcib** (thêm modality) | 0.7445 (**−0.030**) | — |

- Ba model CT (BiomedCLIP, MedicalNet, fmcib) đều hội tụ ~0.50–0.51 nội bộ = ngẫu nhiên — **không phải lỗi chọn model**, kết luận âm dứt khoát.
- Embedding trong fusion **luôn kéo tụt**, tỉ lệ với độ yếu của embedding (D −0.056 > C −0.022 > B cộng dồn −0.112).
- **Kết luận chung:** giới hạn nằm ở DỮ LIỆU (n nhỏ, trần tín hiệu ~0.78), không phải ở model/feature.

### 2.7. TabPFN (2026-07-22)

**Mã:** `experiments/tabpfn_exp/{tabpfn_eval,step1_single,step2_external,step3_fusion,step4_fig}.py`
**Kết quả:** `results/tabpfn_{single,external,fusion}.json` · **Bản dùng:** tabpfn 2.2.1 (v8+ cần login → downgrade v2 local)
**Config:** per-modality discovery-CV 5-seed; fusion late = TabPFN per-modality OOF → trung bình có-mask, 3-seed; so với `uniform_avg` khoá.

| modality | LR | TabPFN |
|---|---|---|
| PDL1 | 0.7186 | 0.7101 |
| Gen (mut_amp) | 0.6760 | 0.6557 |
| GLCM pathology | 0.5762 | 0.5471 |
| Labs | 0.5942 | 0.5596 |
| radiomics (largest) | 0.5765 | 0.5793 |

External: pathology LR 0.767 vs TabPFN 0.748; radiomics LR 0.436 vs TabPFN 0.481 (cả hai ~ngẫu nhiên).

| BM | TabPFN-fusion | uniform_avg | Δ |
|---|---|---|---|
| BM1 | 0.7149 | 0.7746 | −0.060 |
| BM2 | 0.7192 | 0.7665 | −0.047 |
| BM3 | 0.6672 | 0.7110 | −0.044 |
| BM4 | **0.7430** | 0.7191 | **+0.024** |

→ Per-modality thắng 0/5, fusion thắng 1/4 (chỉ BM4, combo 2-modality sạch nhất). Giữ làm comparator hiện đại cho paper, không phải cải tiến.

### 2.8. ★ NLP-clinical verify — 21 tổ hợp bài báo (2026-07-22)

**Mã:** `experiments/nlp_verify/{nlp_modality,nlp_run,run_nlp}.py` · **Kết quả:** `results/nlp_combos.json`
**Config:** MODEL_PARAMS chuẩn (125 epoch, lr 0.01, alpha 0.001, folds=10, seeds chuẩn), backbone `uniform_avg`, cache embedding `../datasets/nlp_clinical_embed.parquet` (247×384, `all-MiniLM-L6-v2`), `no_scale=[idx cnl_nlp]`.
So 3 cách dùng lâm sàng cho MỖI tổ hợp: **base** (không lâm sàng) / **+Labs** (13 biến thô) / **+NLP** (embedding của đúng 13 biến đó).

**BM chủ lực (5-seed | single-run):**

| # | Tổ hợp | base | +Labs | +NLP | Δ(NLP−Labs) 5-seed |
|---|---|---|---|---|---|
| 18 | **Rad+IHC-G+Gen+PDL1 (BM1)** | 0.7720 \| 0.7917 | 0.7661 \| 0.7879 | **0.7832** \| **0.8091** | +0.0171 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | 0.7720 \| 0.7917 | 0.7661 \| 0.7879 | **0.7832** \| **0.8091** | +0.0171 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 0.7596 \| 0.8015 | 0.7447 \| 0.7811 | **0.7687** \| **0.8105** | +0.0241 |

- **+NLP > +Labs ở 19/21 tổ hợp (5-seed), 18/21 (single-run).** Biên trong fusion đa nguồn: **+0.015 → +0.053** (5-seed), tới **+0.064** (single-run, #9 Rad+IHC-A).
- 2 ngoại lệ (PDL1 đơn-modality, Rad-LU radiomics-only) — bản chất fusion 1-nguồn, không phải NLP kém.
- **Đỉnh single-run toàn đợt = 0.8105** (#19), **0.8091** (BM1) — cao hơn mọi con số DyAM/OvO trước đó.
- Head-to-head sạch nhất (thay raw Labs bằng NLP trên đúng config gốc #20/#21): +0.0156/+0.0171 (5-seed), +0.0194/+0.0212 (single-run).
- Đây là **can thiệp DUY NHẤT** trong toàn bộ chuỗi thực nghiệm cho cải thiện dương, bền vững, nhất quán trên nhiều tổ hợp.

### 2.9. ★ OvO + NLP-clinical — đủ 21 tổ hợp bài báo (2026-08-19)

**Mã:** `experiments/nlp_verify/run_ovo_nlp.py` (mới — tái dùng `run_variant` đã tổng quát hoá thêm
tham số `train_fn` trong `nlp_run.py`, cắm `train_ovo_safe` từ `experiments/allcombo/run_ovo_singlerun.py`)
**Kết quả:** `results/ovo_nlp_combos.json`
**Config:** MODEL_PARAMS chuẩn (125 epoch, lr 0.01, alpha 0.001, folds=10, seeds chuẩn `[42,7,123,2024,31337]`),
backbone **OvO** (`MultiModalDynamicModelOvO` qua `train_ovo_safe`, tự loại `cross_modality_enabled`
khỏi params vì OvO không hỗ trợ), cache embedding NLP giống hệt §2.8
(`../datasets/nlp_clinical_embed.parquet`, `no_scale=[idx cnl_nlp]`). Cohort: Discovery, n=247.
Chỉ so 2 biến thể (không chạy lại +Labs riêng cho OvO — đã đủ dữ kiện Labs thô từ §2.2 BM1→BM2):
**base** (không lâm sàng, PDL1 giữ nguyên nếu vốn có trong combo) / **+NLP** (base + `cnl_nlp`).

**BM chủ lực (5-seed | single-run):**

| # | Tổ hợp | OvO base | OvO+NLP | Δ 5-seed | so uniform_avg+NLP (§2.8) |
|---|---|---|---|---|---|
| 8 | PDL1+Gen (BM4) | 0.7183±0.012 \| 0.7157 | 0.7472±0.007 \| **0.7372** | +0.0289 | uni+NLP 0.7445 — OvO nhỉnh hơn +0.0027 |
| 11 | Rad+Gen (BM3) | 0.7045±0.013 \| 0.7633 | 0.7077±0.014 \| 0.7552 | +0.0032 | uni+NLP 0.7076 — gần như bằng nhau |
| 18 | **Rad+IHC-G+Gen+PDL1 (BM1)** | 0.7670±0.019 \| 0.7909 | **0.7822**±0.017 \| **0.8123** | +0.0152 | uni+NLP **0.7832**±0.016 — Δ=−0.0010 (bằng nhau) |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 0.7548±0.013 \| 0.8039 | 0.7648±0.012 \| **0.8133** ★ | +0.0100 | uni+NLP 0.7687 \| 0.8105 |

★ #19 single-run 0.8133 là **con số single-run cao nhất toàn bộ project** (vượt 0.8105 của §2.8) —
nhưng cùng combo này 5-seed mean chỉ 0.7648, **thấp hơn** BM1 → xác nhận lại nguyên tắc đã rút ra ở
§2.5/§4: single-run là 1 phân hoạch may rủi, không phải bằng chứng model tốt hơn, chỉ dùng để đối
chiếu kiểu bài báo gốc.

**Toàn bộ 21 combo — NLP thắng/thua so base:**

- **+NLP thắng ở 15/21 combo (toàn bộ combo ≥2 nguồn, 15/15), thua ở 6/21** (đúng 6 combo đơn/gần-đơn:
  TMB, PDL1, IHC-A, Gen, Rad, Rad-LU) — **khớp chính xác pattern thắng/thua của uniform_avg+NLP (§2.8)**,
  cùng 6 combo thua, cùng 15 combo thắng. Xác nhận phát hiện NLP-clinical robust qua 2 kiến trúc
  attention khác nhau, không phải đặc thù riêng của `uniform_avg`.
- Delta trung bình 21 combo: OvO +0.0078, uniform_avg +0.0069 — chênh lệch không đáng kể giữa 2 backbone.
- **OvO không cộng thêm giá trị so với `uniform_avg` kể cả khi có NLP**: BM1 OvO+NLP 0.7822±0.017 vs
  uniform_avg+NLP 0.7832±0.016 (Δ=−0.0010, trong nhiễu) — khớp với kết luận đã có ở §2.5
  ("OvO ≈ uniform_avg ở 5-seed").
- 2 cặp combo trùng số (#17≡#20, #18≡#21) — do `base` loại bỏ `cnl_dem_labs` trước khi train nên với
  2 combo vốn có Labs, base/+NLP trùng với combo không-Labs tương ứng; đây là hành vi **kế thừa từ
  §2.8**, không phải lỗi.
- **Verify chéo (theo kế hoạch):** combo không-radiomics (TMB, PDL1+Gen) khớp AUC gần như tuyệt đối với
  `paper_combos.json["ovo"]` (§2.5, sai số <0.001). Combo có radiomics lệch nhẹ và nhất quán
  (BM1: 0.7670 ở đây vs 0.7728 ở §2.5, Δ≈−0.006) — cùng hướng, cùng độ lớn với lệch đã thấy giữa
  `uniform_avg` base ở §2.8 (0.7720) và `uniform_avg` gốc ở §0/§2.5 (0.7746, Δ≈−0.003) — do họ script
  `nlp_verify` áp `clean_rad_filters` (loại cột radiomics suy biến, lỗi `PowerTransformer` trên
  scipy≥1.14) mà họ `allcombo` gốc không áp. Xác nhận không phải bug, chỉ là methodology nhất quán
  nội bộ họ `nlp_verify`, giữ nguyên để so sánh táo-với-táo với §2.8.

**Kết luận: GIỮ.** Đây là bằng chứng thứ 2 (backbone OvO, độc lập với `uniform_avg`) cho phát hiện
dương duy nhất của project — NLP-clinical cải thiện fusion đa nguồn nhất quán, bất kể cơ chế attention.
Đồng thời củng cố thêm: OvO tự thân không mang lại lợi thế kiến trúc thật (kể cả khi ghép với can
thiệp dữ liệu tốt nhất đã biết) → khuyến nghị dùng `uniform_avg` + NLP-clinical làm backbone chính
cho báo cáo, OvO chỉ giữ làm đối chứng.

#### Bảng đầy đủ 21 tổ hợp — 4 MODEL (uniform / DyAM / OvO / OvO+NLP), 5-seed & single-run

Tái tạo từ "Bảng 3 MODEL đồng nhất cấu hình" trong
`document/2026-07-21_uniform-dyam-allcombo/bang-so-sanh-paper-combos.md`, thêm 2 cột **OvO+NLP** (kết
quả §2.9 vừa chạy). Cột `uni/DyAM/OvO` giữ nguyên số gốc từ `results/paper_combos.json` (radiomics
không lọc cột suy biến); cột `OvO+NLP` lấy từ `results/ovo_nlp_combos.json` (họ script `nlp_verify`,
có `clean_rad_filters` — lệch nhẹ ~0.003–0.006 với cột OvO gốc ở combo có radiomics, xem lưu ý "verify
chéo" ở trên). Δ tính trên cặp lệch methodology này nên chỉ mang tính tham khảo xu hướng, không phải
so sánh tuyệt đối chính xác từng phần nghìn.

| # | Tổ hợp | #mod | uni 5s | DyAM 5s | OvO 5s | **OvO+NLP 5s** | Δ(NLP−OvO) | uni 1r | DyAM 1r | OvO 1r | **OvO+NLP 1r** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | 0.6161 | 0.6161 | 0.6161 | 0.6062 | −0.0099 | 0.6150 | 0.6150 | 0.6150 | 0.6097 |
| 2 | PDL1 | 1 | 0.7198 | 0.7198 | 0.7198 | 0.6942 | −0.0256 | 0.7080 | 0.7080 | 0.7080 | 0.6775 |
| 3 | IHC-A | 1 | 0.6347 | 0.6347 | 0.6347 | 0.6248 | −0.0099 | 0.6056 | 0.6056 | 0.6056 | 0.6597 |
| 4 | Gen | 1 | 0.6763 | 0.6763 | 0.6763 | 0.6610 | −0.0153 | 0.6759 | 0.6759 | 0.6759 | 0.6546 |
| 5 | Rad | 3 | 0.6561 | 0.6725 | 0.6591 | 0.6515 | −0.0076 | 0.6822 | 0.6925 | 0.6789 | 0.7058 |
| 6 | Rad-LU | 4 | 0.6450 | 0.6373 | 0.6455 | 0.6290 | −0.0165 | 0.6674 | 0.6783 | 0.6711 | 0.6914 |
| 7 | TMB+PDL1 | 2 | 0.7096 | 0.6642 | 0.7088 | 0.7141 | +0.0053 | 0.6980 | 0.6362 | 0.6990 | 0.6979 |
| 8 | PDL1+Gen (BM4) | 2 | 0.7191 | 0.7072 | 0.7183 | **0.7472** | +0.0289 | 0.7124 | 0.6932 | 0.7157 | **0.7372** |
| 9 | Rad+IHC-A | 4 | 0.6679 | 0.6708 | 0.6676 | **0.7116** | +0.0440 | 0.6868 | 0.6883 | 0.6814 | **0.7459** |
| 10 | Rad+IHC-G | 4 | 0.7022 | 0.7055 | 0.7029 | **0.7360** | +0.0331 | 0.7126 | 0.7096 | 0.7000 | **0.7753** |
| 11 | Rad+Gen (BM3) | 4 | 0.7110 | 0.7095 | 0.7132 | 0.7077 | −0.0055 | 0.7472 | 0.7549 | 0.7511 | 0.7552 |
| 12 | IHC-A+Gen | 2 | 0.6982 | 0.6934 | 0.6972 | 0.7016 | +0.0044 | 0.7138 | 0.7148 | 0.7157 | 0.7232 |
| 13 | IHC-G+Gen | 2 | 0.7184 | 0.7470 | 0.7193 | **0.7516** | +0.0323 | 0.7238 | 0.7558 | 0.7227 | 0.7562 |
| 14 | Rad+IHC-A+Gen | 5 | 0.7333 | 0.7249 | 0.7353 | **0.7439** | +0.0086 | 0.7283 | 0.7723 | 0.7360 | **0.8013** |
| 15 | Rad+IHC-G+Gen | 5 | 0.7500 | 0.7507 | 0.7521 | **0.7595** | +0.0074 | 0.7787 | 0.7828 | 0.7690 | **0.8031** |
| 16 | Rad+IHC-A+MutAmp | 6 | 0.7478 | 0.7338 | 0.7443 | **0.7474** | +0.0031 | 0.7651 | 0.7823 | 0.7667 | 0.7677 |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 0.7548 | 0.7428 | 0.7526 | **0.7569** | +0.0043 | 0.7726 | 0.7961 | 0.7674 | **0.7978** |
| 18 | Rad+IHC-G+Gen+PDL1 (BM1) | 6 | 0.7746 | 0.7595 | 0.7728 | **0.7822** | +0.0094 | 0.7915 | 0.7976 | 0.7975 | **0.8123** |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7704 | 0.7488 | 0.7662 | 0.7648 | −0.0014 | 0.7885 | 0.7892 | 0.7906 | **0.8133** ★ |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 0.7467 | 0.7363 | 0.7450 | **0.7569** | +0.0119 | 0.7628 | 0.7650 | 0.7622 | **0.7978** |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | 7 | 0.7665 | 0.7638 | 0.7632 | **0.7822** | +0.0190 | 0.7784 | 0.7908 | 0.7780 | **0.8123** |

★ = con số single-run cao nhất toàn bộ project (không phải chỉ số robust, xem caveat ở trên).

**Đọc bảng:** ở cột **5-seed** (chỉ số robust), OvO+NLP thắng cả 3 model còn lại (uni/DyAM/OvO
không-NLP) ở 15/21 combo đa nguồn — cùng tập combo mà `uniform_avg+NLP` (§2.8) cũng thắng — và là
**thấp nhất trong 4 model** ở đúng 6 combo đơn/gần-đơn còn lại (#1–6, clinical text không có gì để
bổ trợ khi chỉ có 1 nguồn). Cột **single-run** thì nhiễu hơn nhiều và không theo pattern này ở #1–6
(vd IHC-A #3, Rad #5: OvO+NLP 1r lại cao nhất, không phải thấp nhất) — đúng như cảnh báo đã lặp lại
nhiều lần trong file này: single-run là 1 phân hoạch, không đại diện xu hướng thật, chỉ cột 5-seed
mới nên dùng để kết luận.

---

### 2.10. ★ Kiểm chứng lại toàn bộ số liệu bài báo (2026-09-30)

**Mã:** `experiments/verify_all.py` (runner nối 6 phase) · `experiments/nlp_verify/verify_nlp_variants.py` (mới)
· `experiments/compare_to_paper.py` (mới) · **Kế hoạch & kết quả:** `document/2026-09-29_verify-paper-numbers/`
**Bản gốc đối chiếu:** `experiments/results/_ref_2026-09-29/` (31 JSON sao lưu trước khi chạy)
**Config:** đúng chuẩn §0 (125 epoch, lr 0.01, alpha 0.001, beta 0.0, folds=10, seeds `[42,7,123,2024,31337]`),
Discovery n=247. **Khác biệt môi trường so với các lần chạy gốc:** scipy 1.14.1 (gốc <1.14) —
phải vá `PowerTransformer._yeo_johnson_optimize` (cột lỗi → λ=1), cùng bản vá đã có trong `nlp_run.py`.
Thời gian: 1a 64,8′ · 1b 13,1′ · 1c 38,2′ · 2 71,6′ · 3 6,4′ · 4 5,0′ (tổng ~3h20, CPU).

| Nhóm số | Kết quả |
|---|---|
| Bảng 1, Note S1 (câu/token), PCA 94,4% | ✅ khớp tuyệt đối |
| Bảng 4 — 13/21 tổ hợp không radiomics | ✅ khớp tới 4 chữ số |
| Bảng 4 — **cột OvO+NLP cả 21 tổ hợp** | ✅ **khớp tuyệt đối** (đường chạy này vốn đã có bản vá scipy) |
| Bảng 4 — 23/63 ô còn lại (đều có radiomics) | ⚠️ lệch ≤ 0,016, hai chiều — do scipy |
| Note S3 (SE(Δ), cỡ mẫu) | ⚠️ sai số học: SE(Δ) thực 0,0158–0,0166; n cần 1.211–1.343 (Δ=0,02) |
| **Bảng 5, 6** (AUC theo mã hoá lâm sàng) | ❌ **không tái lập**: IHC-A No clin 0,778 vs 0,764; +NLP-PCA16 0,770 vs 0,784 → hiệu ứng NLP **đảo chiều** (−0,008 thay vì +0,020) |
| **Bảng 7** (C-index, Cox HR) | ❌ **không tái lập**: C-index No clin 0,636 (bài 0,623), +NLP-PCA16 0,621 (bài 0,628); HR 5,84 vs 5,30 và 4,72 vs 6,07 — đảo chiều |
| **Bảng 8** (KM log-rank χ²) | ❌ **không tái lập** với cả 3 cách chia nhóm (median/mean/ngưỡng AUC); IHC-G +NLP-PCA16 16,43 thay vì 28,92 |
| Bảng 7 — tdAUC, IBS | ⛔ không kiểm được: sksurv 0.27 gọi `numpy.trapezoid` (cần numpy ≥ 2, đang 1.26.4) |
| Cox p<0,001 mọi model · Wilcoxon per-fold không ý nghĩa · KM p<0,005 | ✅ khớp |

- **Phát hiện provenance:** Bảng 5/6 KHÔNG do notebook `Figures-Finalized.ipynb` sinh ra (quét cả 66 cell:
  không cell nào gọi `clinical_nlp_embedding`/`MiniLM`/`sentence-transformers`). §2.0b ở trên ghi nguồn sai.
  Nguồn duy nhất còn lại là `document/nlp-model/nlp-ket-qua.md`.
- **Hai script độc lập** (`verify_nlp_variants.py` mới và `survival_analysis_nlp.py` cũ) cho cùng bộ số mới,
  lệch nhau <0,001 → không phải lỗi script.
- Bảng 5–8 đều là **đơn-phân-hoạch**; lệch 0,01–0,02 đúng bằng SE(Δ)≈0,016 mà Note S3 tính ra.
- **VERDICT: GIỮ bằng chứng 5-seed** (Bảng 4, §2.8, §2.9 — tái lập 100%). **Bảng 5–8 cần chạy lại bằng
  5-seed** thì mới đứng được; danh sách sai khác + phương án chờ quyết định ở
  `document/2026-09-29_verify-paper-numbers/sai-khac-can-quyet-dinh.md`. Bài báo giữ nguyên trạng, chưa sửa.

### 2.11. Đo GPU vs CPU (2026-09-30, không phải thí nghiệm mô hình)

Cài `torch 2.13.0+cu126` (index cu124 chỉ có tới 2.6.0 nên không dùng được), RTX 3060 12GB, driver 560.94.
Đo cùng cấu hình (4 modality, 10-fold, 125 epoch), 3 lần liên tiếp trong một tiến trình:

| Thiết bị | Thời gian | AUC (fold-mean) |
|---|---|---|
| CPU (torch 2.13.0+cpu) | **15,4 s** | 0,539 |
| GPU (torch 2.13.0+cu126) | 20,0 / 18,0 / 17,9 s | 0,547 |

**Thí nghiệm tách nguyên nhân** (cùng torch 2.13.0+cu126, cùng seed, chỉ đổi thiết bị bằng cách ép
`torch.cuda.is_available = lambda: False`):

| Thiết bị | Thời gian | AUC pooled |
|---|---|---|
| CPU (ép) | 13,8 / 14,6 s | **0,5390** |
| GPU | 17,9 / 18,7 / 20,0 s | **0,5473** |

→ Khác biệt đến từ **thiết bị**, không phải phiên bản/bản build torch. Đã loại trừ TF32
(`matmul.allow_tf32=False`, `float32_matmul_precision='highest'` vẫn cho 0,5473). Nguyên nhân là
kernel và thứ tự cộng dồn của cuBLAS khác CPU BLAS; sai số ~1e-7 mỗi phép được khuếch đại qua
125 epoch × 10 fold nên quỹ đạo huấn luyện tách ra. Mỗi thiết bị **tự nó tất định** (chạy lại ra
đúng cùng số).

- GPU **chậm hơn ~17%** — mô hình quá nhỏ (batch 247, vài lớp linear), chi phí phóng kernel lấn át; phần
  RobustScaler/PowerTransformer/chọn đặc trưng L1 vẫn chạy CPU.
- GPU cho AUC lệch **+0,008** trên cùng cấu hình tất định → mọi số chạy trên GPU không so trực tiếp được
  với các mục trên.
- **VERDICT: nên quay lại bản CPU** cho dự án này. Nếu vẫn dùng GPU, phải ghi rõ "chạy GPU" trong mọi mục
  kết quả sau đó.

### 2.12. ★ Bảng 5–8 chạy lại bằng 5-seed (2026-09-30) — SK-3/4/5

**Mã:** `experiments/nlp_verify/verify_nlp_5seed.py` (mới) · **Kết quả:** `results/verify_nlp_5seed.json`
**Config:** chuẩn §0 (125 epoch, lr 0.01, alpha 0.001, beta 0.0, folds=10, seeds `[42,7,123,2024,31337]`),
model DyAM (`train`), rad filter `l1_strength=0.1` PC/PL/LN, vá scipy như §2.10, CPU. 11 biến thể
(IHC-A 7, IHC-G 4) × 5 seed = 55 lượt train, ~35 phút. Thay cho số ĐƠN-PHÂN-HOẠCH của Bảng 5–8.

| Nhánh | Biến thể | AUC 5-seed | (bài) | C-index | Cox HR | KM χ² |
|---|---|---|---|---|---|---|
| IHC-A | No clinical | **0.7490±0.020** | 0.764 | 0.611±0.005 | 4.31±0.62 | 13.4±3.2 |
| IHC-A | + Labs | 0.7300±0.024 | 0.768 | 0.616±0.007 | 4.85±0.27 | 16.9±6.3 |
| IHC-A | + NLP raw | 0.7428±0.019 | 0.781 | 0.608±0.008 | **4.97±0.22** | **19.3±7.0** |
| IHC-A | + NLP-PCA16 | 0.7389±0.017 | 0.784 | 0.608±0.010 | 4.83±0.33 | 19.2±8.7 |
| IHC-A | + Labs+NLP | 0.7387±0.010 | 0.767 | 0.610±0.006 | 4.72±0.47 | 15.2±6.0 |
| IHC-A | + Labs+NLP-PCA16 | 0.7381±0.009 | 0.753 | 0.609±0.008 | 4.61±0.48 | 14.6±6.0 |
| IHC-A | + BioClinBERT-PCA16 | 0.7265±0.016 | 0.767 | 0.612±0.010 | 5.19±0.70 | 18.0±6.1 |
| IHC-G | No clinical | 0.7679±0.026 | 0.784 | 0.618±0.007 | 4.16±0.50 | 12.4±3.3 |
| IHC-G | + Labs | 0.7490±0.021 | 0.788 | **0.626±0.011** | 4.77±0.85 | 19.1±7.8 |
| IHC-G | **+ NLP raw** | **0.7688±0.026** | 0.813 | 0.619±0.014 | **4.92±0.50** | 18.2±9.4 |
| IHC-G | + NLP-PCA16 | 0.7666±0.027 | 0.812 | 0.619±0.012 | 4.91±0.46 | 17.5±7.3 |

**Paired bootstrap vs No clinical (2000 lần):** IHC-A +Labs −0.019 p=0.108 · +NLP raw −0.006 p=0.272 ·
+NLP-PCA16 −0.010 p=0.134 · **+BioClinBERT-PCA16 −0.023 p=0.028 (có ý nghĩa)**;
IHC-G +Labs −0.019 p=0.209 · +NLP raw +0.001 p=0.882 · +NLP-PCA16 −0.001 p=0.926.

- **① "NLP > Labs thô": GIỮ.** IHC-A 0.7428 vs 0.7300 (+0.013); IHC-G 0.7688 vs 0.7490 (+0.020).
  Nhất quán với §2.8 (19/21 tổ hợp) và cột OvO+NLP §2.9 (tái lập 21/21).
- **② "NLP cải thiện phân tầng sống còn": KHÔNG ỔN ĐỊNH — XOÁ hoặc nói rất dè dặt.** Chiều lật giữa
  hai lần chạy: đơn-phân-hoạch HR 5.84→4.60 (NLP kém hơn), 5-seed HR 4.31→4.97 (NLP tốt hơn).
  Ngoài ra Cox HR ở đây KHÔNG so được giữa model: `compute_multivariate_cox` min-max scale điểm về
  [0,1] nên HR phụ thuộc biên độ điểm; đổi sang z-score thì HR co về 1.43–1.54 và thứ tự đảo
  (No clinical 1.540 cao nhất). C-index (không phụ thuộc thang đo) cho No clinical cao nhất ở cả hai
  lần chạy. Chỉ giữ được: mọi model phân tầng có ý nghĩa p<0.005; khác biệt C-index không có ý nghĩa.
- **③ "NLP > không-dùng-lâm-sàng" (cột ΔAUC của Bảng 5/6): XOÁ.** Bài ghi +0.020/+0.030; 5-seed cho
  −0.010/+0.001, p=0.13/0.93. Ở cấu hình này **mọi** cách thêm lâm sàng đều không cải thiện
  (Labs thô −0.019 ở cả hai nhánh).
- Mới: **BioClinBERT kém hơn mốc No clinical có ý nghĩa thống kê** (p=0.028) — mạnh hơn cách Note S2
  đang diễn đạt.
- **VERDICT:** dùng bộ 5-seed này thay số đơn-phân-hoạch của Bảng 5–8; đổi trục so sánh của §3.3 sang
  "NLP vs Labs". Đề xuất chi tiết ở `document/2026-09-29_verify-paper-numbers/sai-khac-can-quyet-dinh.md`
  (SK-3/4/5). Bài báo vẫn giữ nguyên trạng, chờ quyết định.

> **Đính chính 2026-10-04 (áp cho §2.13 và §2.15):** hai mục này ghi "phân hoạch giống harness chuẩn".
> Không hoàn toàn đúng: mã cooperative learning và XGBoost hoán vị theo seed rồi dùng `KFold(10)`
> KHÔNG shuffle, còn harness (`train_uniform_avg`) dùng `KFold(10, shuffle=True, random_state=0)` trên
> chỉ số đã hoán vị. Cả hai đều là CV 5-seed × 10-fold hợp lệ nhưng KHÔNG cùng fold. Kết luận không
> đổi (chênh lệch −0.04…−0.10 lớn hơn nhiều lần nhiễu phân hoạch ~0.015), chỉ là mô tả thiếu chính xác.

### 2.13. Cooperative learning (PNAS 2022) — 4 benchmark (2026-09-30) · XOÁ

**Mã:** đã xoá sau khi kết luận âm — thuật toán + số liệu đầy đủ giữ ở
`document/2026-09-30_cooperative-learning/ket-qua.md`
**Config:** lasso-based block coordinate descent, agreement penalty ρ ∈ {0, 0.2, 0.5, 1, 2, 5},
n_iter=4, λ chọn bằng LassoCV(cv=3) vòng đầu; chuẩn hoá RobustScaler theo view, hàng thiếu modality
= 0, điểm = trung bình các view có mask; phân hoạch giống harness chuẩn (hoán vị theo seed +
KFold(10)), seeds chuẩn, 5-seed × 10-fold. Không dùng L1-filter radiomics (đã kiểm riêng, xem dưới).

| BM | coop (ρ tốt nhất) | uniform_avg | DyAM | OvO |
|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7330 | **0.7746** | 0.7595 | 0.7704 |
| BM2 +Labs | 0.7165 | **0.7665** | 0.7638 | 0.7692 |
| BM3 Rad+Gen | 0.6687 | **0.7160** | 0.7095 | 0.7163 |
| BM4 PDL1+Gen | **0.7464** | 0.7191 | 0.7072 | 0.7183 |

- Đạt ≥ uniform_avg ở **1/4** BM → không đủ ngưỡng (≥2/4) để chạy 21 tổ hợp, dừng theo tiêu chí đặt trước.
- **Kiểm tra công bằng:** áp bộ lọc L1 radiomics của dự án trong từng fold train (3 seed) → BM1 0.729,
  BM3 0.677, gần như không đổi. Kết quả âm KHÔNG do cách chọn đặc trưng.
- **ρ gần như không ảnh hưởng** (BM1: 0.719 → 0.733 khi ρ đi từ 0 đến 5; BM2–BM4 phẳng) → vị trí trên
  dải early↔late fusion không phải yếu tố quyết định; ρ tốt nhất luôn về phía late fusion.
- **Trùng mẫu hình với TabPFN (§2.7):** cả hai chỉ thắng ở BM4 (tổ hợp ít chiều, không radiomics) và
  thua 0.04–0.06 ở mọi tổ hợp có radiomics.
- **VERDICT: XOÁ.** Giữ làm comparator hiện đại để nhắc một câu trong bài. Ghi nhận phụ: BM4 = 0.7464
  xấp xỉ mức tốt nhất ở tổ hợp đó (OvO+NLP 0.7472, uniform+NLP 0.7445) nhưng KHÔNG cần dữ liệu NLP.
  Không phải con số cao nhất dự án — cao nhất vẫn là uniform_avg+NLP 0.7832 (BM1).

### 2.14. ★★ Mẫu khuyết modality là TÍN HIỆU — NLP + miss_pattern, 21 tổ hợp (2026-09-30)

**Mã:** `experiments/missingness/run_miss.py` · **Kết quả:** `results/missingness.json`
**Config:** backbone `uniform_avg`, MODEL_PARAMS chuẩn, 5-seed × 10-fold, seeds chuẩn, Discovery n=247,
vá scipy như §2.10, `no_scale` cho cả `cnl_nlp` và `miss_pattern`.
**Modality mới `miss_pattern`:** vector nhị phân `has_<modality>` cho từng nguồn trong tổ hợp +
`n_available` (tổng số nguồn có sẵn). Luôn khả dụng với mọi bệnh nhân.

**Phát hiện dẫn tới thí nghiệm** (đo trên điểm đã lưu, BM1, không cần train):

| Chỉ dùng | AUC |
|---|---|
| "có tiêu bản pathology hay không" (1 biến nhị phân) | **0.668** |
| số modality có sẵn k | 0.665 |
| PD-L1 TPS (chỉ dấu sinh học mạnh nhất) | 0.729 |
| toàn bộ mô hình đa modality | 0.773 |

Tỉ lệ KHÔNG đáp ứng theo k: k=1 100% (n=9) · k=2 85% (52) · k=3 80% (85) · k=4 70% (70) · k=5 50% (30).
AUC trong từng nhóm cùng k: 0.69–0.76 → mô hình vẫn có giá trị riêng, không chỉ ăn theo mẫu khuyết.
Sau khi hồi quy bỏ ảnh hưởng tuyến tính của k: 0.731 → **~0.04 hiệu năng đến từ mẫu khuyết.**
Chuẩn hoá z theo từng nhóm k (tưởng là khử nhiễu) làm AUC GIẢM 0.046–0.055 → đó là tín hiệu, không phải nhiễu.

**Kết quả 21 tổ hợp:**

| | base | +NLP | **+NLP+miss** |
|---|---|---|---|
| Số tổ hợp thắng | 5/21 | 2/21 | **14/21** |
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7730 | 0.7832 | **0.7896 ± 0.011** ← cao nhất dự án |
| Rad+IHC-G (#10) | 0.7089 | 0.7383 | **0.7609** (+0.052 so base) |
| IHC-G+Gen (#13) | 0.7184 | 0.7514 | **0.7697** (+0.051) |
| Rad+IHC-G+Gen (#15) | 0.7510 | 0.7576 | **0.7871** (+0.036) |

- base chỉ thắng ở tổ hợp đơn-nguồn (#1, 2, 4, 5) và #11 Rad+Gen — đúng các tổ hợp không có
  modality nào mang mẫu khuyết đáng kể (genomics có đủ 247/247).
- Lợi ích tập trung ở tổ hợp **có pathology hoặc PD-L1** — hai nguồn thiếu nhiều nhất (142/247 và 46/247).
- **Chưa đạt ý nghĩa thống kê:** BM1 nlp_miss vs base Δ=+0.0133, p=0.330 (paired bootstrap 2000).
- Cột "+miss đơn lẻ" chỉ có 3 tổ hợp (#8, 11, 18) do lỗi tham số `--v=` cắt chuỗi sai; đã sửa.
- **VERDICT: GIỮ — cao nhất dự án (0.7896), nhưng phải khai báo minh bạch.** Phần tăng từ
  `miss_pattern` là thông tin về QUY TRÌNH CHĂM SÓC (bệnh nhân nặng ít được làm đủ xét nghiệm), không
  phải sinh học khối u; gần như chắc chắn không khái quát sang cohort có quy trình thu thập khác.
  Bản thân phát hiện "một biến nhị phân có-pathology đạt AUC 0.668" đủ giá trị để đưa vào Limitations
  dù có dùng miss_pattern trong mô hình hay không.

### 2.15. XGBoost — lỗ hổng comparator, 4 benchmark (2026-10-01) · XOÁ

**Mã:** `experiments/gbdt/run_xgb.py` · **Kết quả:** `results/xgboost.json` · xgboost 3.4.1 (mới cài)
**Lý do chạy:** `lung_helpers.train_XGBoost` tồn tại từ lâu nhưng CHƯA TỪNG được đánh giá qua harness
(và thư viện còn chưa cài — hàm đó sẽ lặng lẽ rơi về RandomForest). Tổng quan 2025-2026 nói GBDT vượt
mạng nơ-ron trên dữ liệu bảng; XGBoost lại xử lý khuyết dữ liệu ngay trong thuật toán nên về lý thuyết
tự khai thác được tín hiệu khuyết của §2.14.
**Config:** 400 cây, depth 3, lr 0.03, subsample 0.8, colsample 0.5, min_child_weight 5, reg_lambda 2,
`scale_pos_weight` theo tỉ lệ lớp; 5-seed × 10-fold, phân hoạch giống harness chuẩn.
Hai biến thể: `concat` (nối mọi đặc trưng, GIỮ NaN) · `late` (XGB từng modality → trung bình có mask).

| Tổ hợp | concat | late | uniform_avg | Δ |
|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | 0.6936 | 0.7115 | **0.7730** | −0.062 |
| BM2 +Labs | 0.6860 | 0.6927 | **0.7686** | −0.076 |
| BM3 Rad+Gen | 0.6173 | 0.6116 | **0.7160** | −0.099 |
| BM4 PDL1+Gen | **0.7366** | 0.7077 | 0.7191 | +0.018 |

- **Mẫu hình lặp lại lần thứ BA.** Ba phương pháp hoàn toàn khác nhau, cùng một hình dạng kết quả:

| Phương pháp | BM4 (không radiomics) | Các tổ hợp có radiomics |
|---|---|---|
| TabPFN (§2.7) | +0.024 | −0.044 … −0.060 |
| Cooperative learning (§2.13) | +0.027 | −0.043 … −0.050 |
| XGBoost (§2.15) | +0.018 | −0.062 … −0.099 |

- Kết luận rút ra: **thứ phân biệt các phương pháp KHÔNG phải cơ chế hợp nhất, mà là cách xử lý
  radiomics.** Bộ mã hoá theo-từng-modality của dự án (tanh(Wx) + lọc L1 theo độ bền) mới là phần
  gánh hiệu năng; mọi bộ học dạng bảng tổng quát đều chết ở radiomics chiều cao, khuyết nhiều
  (PC 163/247, PL 21/247, LN 67/247).
- Ngay cả ở BM4, XGBoost (0.7366) vẫn thua lr_late (0.7515) và cooperative learning (0.7464).
- **VERDICT: XOÁ khỏi hướng cải tiến.** Giữ làm comparator trong bài: cùng với TabPFN, nó cho thấy
  phương pháp dạng bảng hiện đại không thay được kiến trúc theo-modality trên dữ liệu này.

### 2.16. ★★ Bốn hướng cải thiện mô hình (2026-10-01/03) — chốt 0.7970

**Mã:** `experiments/modelimp/` · **Kết quả:** `results/modelimp.json`, `results/combat_external.json`
**Tài liệu:** `document/2026-10-01_model-improvement/ket-qua.md`
**Config:** backbone `uniform_avg`, 5-seed × 10-fold, seeds chuẩn, Discovery n=247, vá scipy như §2.10.

**CON SỐ CHỐT:**

| Cấu hình | AUC 5-seed | + ensemble theo seed |
|---|---|---|
| BM1 nền | 0.7730 ± 0.019 | 0.7838 |
| **BM1 + NLP + miss_pattern** | **0.7896 ± 0.011** | **0.7970** ← cao nhất dự án |
| Rad+IHC-G+Gen nền | 0.7510 ± 0.015 | 0.7631 |
| Rad+IHC-G+Gen + NLP + miss | 0.7871 ± 0.014 | 0.7956 (+0.0446 so nền) |

| Hướng | Kết quả | Verdict |
|---|---|---|
| **Ensemble theo seed** | +0.0091, thắng **19/21** tổ hợp; tính từ điểm đã lưu, không train lại | **GIỮ** |
| Dò siêu tham số | BM1/3-seed +0.008 → BM1/5-seed **−0.0035**. Giúp nhiều ở tổ hợp ít chiều (PDL1+Gen +0.033, IHC-G+Gen +0.023), hại ở tổ hợp giàu radiomics | KHÔNG ĐỨNG VỮNG |
| Pairwise loss (11,470 cặp) | +0.0045 (BM1), +0.0016 (BM4); triệt tiêu khi gộp với tham số mới | QUÁ NHỎ |
| Modality dropout 0.2 | −0.0028 (BM1), +0.0035 (BM4) | XOÁ |
| ComBat radiomics | nội bộ −0.023 (BM1) / −0.029 (BM3); ngoại kiểm rad_valid 0.42 → **0.51** (mức ngẫu nhiên) | XOÁ |

- **Bài học phương pháp:** +0.008 đo trên 3 seed của MỘT tổ hợp rồi chọn cấu hình theo chính tổ hợp
  đó là tự lừa mình — chạy lại 5 seed thì thành −0.0035. Mọi dò tham số sau này phải xác nhận trên
  nhiều tổ hợp với đủ seed.
- **alpha = 0.01 làm AUC sụp 0.044** — mô hình rất nhạy với mức phạt L1 lên trọng số attention.
- **ComBat âm nhưng có giá trị:** loại trừ giả thuyết "radiomics không khái quát vì hiệu ứng máy chụp".
  Nếu đúng vậy thì khử batch phải làm ngoại kiểm nhích lên; thực tế chỉ từ dưới-ngẫu-nhiên lên
  đúng-ngẫu-nhiên. Nguyên nhân nằm chỗ khác (chọn/phân vùng tổn thương, hoặc tín hiệu không chuyển được).
- **SỰ CỐ ĐÃ XỬ LÝ:** `load_all(use_cache=False)` ghi đè cache dùng chung 98 MB → dữ liệu đã ComBat
  ngấm vào cache, làm nhánh "gốc" đọc ra chính dữ liệu đã hài hoà (Δ=0.0000 tuyệt đối). Đã tách cache
  ComBat sang file riêng và dựng lại cache sạch. Hướng ensemble/tham số/loss chạy TRƯỚC khi nhiễm nên
  không bị ảnh hưởng. **Cảnh báo cho session sau: đừng gọi `load_all(use_cache=False)` với dữ liệu
  đã biến đổi mà không đổi hướng `_CACHE_PATH`.**

### 2.17. Mô hình tự thiết kế SHARE-Rad (2026-10-03) · XOÁ cả hai thành phần mới

**Mã:** `experiments/modelimp/{model_shared,run_design}.py` · **Kết quả:** `results/share_rad.json`
**Config:** backbone `uniform_avg`, MODEL_PARAMS chuẩn, 5-seed × 10-fold (riêng thành phần chia sẻ
trọng số chỉ chạy chẩn đoán 1 seed vì mức tụt quá lớn). Mã mới tái lập ĐÚNG baseline (Rad seed 42 =
0.6727) khi tắt mọi thay đổi.

**Thành phần A — chia sẻ trọng số bộ mã hoá radiomics giữa PC/PL/LN** (giả thuyết: mỗi vị trí học
trên quá ít dữ liệu 163/21/67, gộp lại sẽ tốt hơn). Tổ hợp Rad, seed 42:

| Cấu hình | AUC |
|---|---|
| gốc: chọn đặc trưng riêng + trọng số riêng (~61 đặc trưng/vị trí) | **0.6727** |
| chọn đặc trưng trên tổn thương GỘP (~96), trọng số riêng | 0.6289 (−0.044) |
| chọn đặc trưng gộp + BUỘC trọng số | 0.5636 (−0.109) |

→ **Bác bỏ.** Tín hiệu radiomics ĐẶC THÙ theo vị trí tổn thương; không gộp được cả ở bước chọn đặc
trưng lẫn ở trọng số. Giải thích thêm vì sao XGBoost/TabPFN/coop sụp ở radiomics: không cái nào có
bước chọn đặc trưng riêng theo vị trí.

**Thành phần B — chế độ huấn luyện theo modality** (kênh không-radiomics: lr 0.005, alpha 0.0001;
radiomics giữ 0.01/0.001; huấn luyện chung), 5 seed:

| Tổ hợp | uniform_avg | B | Δ | Δ ensemble |
|---|---|---|---|---|
| PDL1+Gen | 0.7191 | 0.7369 | **+0.0178** | +0.0167 |
| IHC-G+Gen | 0.7184 | 0.7295 | **+0.0111** | +0.0074 |
| Rad+IHC-G+Gen | 0.7510 | 0.7520 | +0.0010 | −0.0003 |
| Rad+Gen | 0.7160 | 0.7119 | −0.0041 | −0.0027 |
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7730 | 0.7690 | −0.0040 | −0.0019 |
| Rad+IHC-A+Gen+PDL1 | 0.7547 | 0.7409 | −0.0138 | −0.0100 |

→ Giúp ở tổ hợp KHÔNG có radiomics (+0.011, +0.018), trung tính hoặc hại ở tổ hợp có radiomics — kể
cả khi chế độ nhẹ chỉ áp cho kênh không-radiomics. Cùng mẫu hình với §2.16 (cấu hình nhẹ toàn cục).

- **VERDICT: XOÁ cả hai.** Mô hình tốt nhất không đổi: `uniform_avg` + đầu riêng theo vị trí + NLP +
  miss_pattern + ensemble seed = **0.7970** (§2.16).
- **Tổng kết chuỗi tìm kiếm mô hình (§2.2–§2.17):** 7 cơ chế hợp nhất, 3 bộ học dạng bảng, 5 thay đổi
  huấn luyện, 1 hài hoà hoá dữ liệu, 2 thay đổi kiến trúc bộ mã hoá — không cái nào vượt
  `uniform_avg` ở tổ hợp chủ lực ngoài ensemble theo seed. Trần hiệu năng do dữ liệu (n=247) quy định.

### 2.18. RAEF — hợp nhất theo độ tin cậy + tiên nghiệm sẵn-có (2026-10-03)

**Mã:** `experiments/modelimp/{model_raef,run_raef}.py` · **Kết quả:** `results/raef.json`
**Mô hình:** `s = Σ m_i ρ_i tanh(W_i x_i) / Σ m_i ρ_i + v·m`. Khởi tạo ρ bằng nhau, v=0 → trùng khít
`uniform_avg` (đã kiểm: tắt cả hai khối cho đúng 0.7927 ở seed 42). Thêm 2N tham số.
**Config:** MODEL_PARAMS chuẩn, 5-seed × 10-fold, 6 tổ hợp, mốc `uniform_avg` cùng harness.

| Tổ hợp | uniform | R học gradient | R ước lượng CV | A sẵn-có | R+A |
|---|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7730 | 0.7400 | 0.7653 | 0.7670 | 0.7658 |
| Rad+IHC-A+Gen+PDL1 | 0.7547 | 0.7090 | 0.7503 | 0.7563 | 0.7542 |
| Rad+IHC-G+Gen | 0.7510 | 0.7296 | 0.7345 | **0.7726** | 0.7670 |
| Rad+Gen | 0.7160 | 0.7008 | 0.7163 | 0.7178 | 0.7175 |
| PDL1+Gen | 0.7191 | 0.7209 | 0.7169 | 0.7230 | 0.7154 |
| IHC-G+Gen | 0.7184 | 0.7330 | 0.7025 | **0.7412** | 0.7394 |
| **Δ trung bình / thắng** | — | **−0.0165** / 2 | **−0.0077** / 1 | **+0.0076** / 5 | +0.0045 / 3 |

- **(R) trọng số độ tin cậy theo modality: XOÁ cả hai dạng.** Học bằng gradient −0.0165 (tệ nhất ở tổ
  hợp radiomics: −0.033, −0.046); ước lượng bằng AUC 3-fold bên trong fold train −0.0077. Đây là dạng
  trọng số hợp nhất thứ 3 và thứ 4 thất bại sau DyAM và OvO → **ở n=247, hợp nhất không tham số là
  lựa chọn được số liệu ủng hộ, không phải thiếu sót.**
- **(A) tiên nghiệm sẵn-có dạng cộng: dương vừa phải** (+0.0076, 5/6; giảm sd còn 0.005–0.011), mạnh ở
  tổ hợp có pathology (+0.022, +0.023) nhưng −0.006 ở BM1 — kém hơn dạng "kênh chuyên gia" của §2.14
  (BM1 +0.009). → dùng dạng kênh.
- **VERDICT:** không thay mô hình chốt. Kiến trúc trình bày = tập chuyên gia theo modality (radiomics
  riêng theo vị trí · lâm sàng mã hoá ngôn ngữ · chuyên gia sẵn-có) + hợp nhất chuẩn hoá theo mask
  không tham số + ensemble = **0.7970** (§2.16). Mỗi thành phần có một dòng ablation: gộp vị trí −0.11
  (§2.17) · NLP +0.017 (§2.8) · sẵn-có +0.009 (§2.14) · trọng số học được −0.017 (mục này) ·
  ensemble +0.009 (§2.16).

### 2.19. Sàng lọc họ ĐỒ THỊ BỆNH NHÂN (patient similarity network) (2026-10-04) · XOÁ

**Mã:** `experiments/graphfuse/run_psn.py` · **Kết quả:** `results/psn_screen.json`
**Lý do:** họ duy nhất chưa thử, khác bản chất với mọi mô hình trước (dự đoán từ bệnh nhân tương tự thay
vì hàm tham số trên từng bệnh nhân); có bằng chứng ở cỡ mẫu gần (MAGNET 2025: TCGA n=360/433/956).
**Config:** tương đồng RBF theo từng modality (RobustScaler trên fold train, radiomics qua đúng bộ lọc L1
theo vị trí), hợp nhất = trung bình trên các modality hai bệnh nhân cùng có, dự đoán Nadaraya–Watson.
Phân hoạch ĐÚNG harness (`KFold(10, shuffle=True, random_state=0)` trên chỉ số hoán vị theo seed),
5 seed. Cột "uni+λ" = z(điểm uniform_avg đã lưu) + λ·z(điểm đồ thị), λ cố định trước, không dò.

| Tổ hợp | uniform_avg | đồ thị | uni + 0.5·đồ thị | uni + 1.0·đồ thị |
|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | **0.7730** | 0.6877 | 0.7634 (−0.010) | 0.7519 (−0.021) |
| Rad+IHC-G+Gen | **0.7510** | 0.6110 | 0.7271 (−0.024) | 0.7079 (−0.043) |
| Rad+Gen | **0.7160** | 0.6355 | 0.7067 (−0.009) | 0.6976 (−0.018) |
| IHC-G+Gen | **0.7184** | 0.5827 | 0.7009 (−0.018) | 0.6833 (−0.035) |
| PDL1+Gen | **0.7191** | 0.6750 | 0.7157 (−0.004) | 0.7104 (−0.009) |

- Riêng đồ thị: 0.58–0.69, thấp hơn uniform_avg 0.04–0.14. **Cộng vào điểm uniform_avg làm AUC GIẢM ở
  cả 5/5 tổ hợp** → thông tin láng giềng không bổ sung, chỉ là bản nhiễu hơn của cùng tín hiệu.
- Chưa dựng GNN đầy đủ kiểu MAGNET (embedding học end-to-end có thể cho độ tương đồng tốt hơn RBF trên
  đặc trưng thô) — nhưng thiên kiến quy nạp nền tảng của họ này không có tín hiệu bổ sung ở đây.
- **VERDICT: XOÁ.**

**Ghi chú liên quan — OvO gần như phẳng** (`results/diagnostics.json`, chẩn đoán có từ 2026-07): tại BM1,
đầu ra sigmoid của cổng OvO là 0.58–0.60 với sd chỉ 0.034–0.037 ở MỌI modality (logit thô sau khi chia cho
số đặc trưng có sd 3e-5…1.5e-3). Tức sau chuẩn hoá L1, trọng số OvO ≈ chia đều trên các modality có sẵn →
**OvO hoà với uniform_avg (Δ=+0.0004/21 tổ hợp) vì về mặt chức năng nó gần như LÀ uniform_avg.**
Số đặc trưng thực tế một fold BM1: rad PC/PL/LN = 31/11/26, GLCM 150, Gen 11, PDL1 1 (tổng 230) →
tham số: uniform ≈ 236, OvO ≈ 472, DyAM ≈ 1.650 (đính chính ước lượng 350/700/2.450 đã nêu trước đó).

### 2.20. GNN trên đồ thị bệnh nhân — kiểu MAGNET (2026-10-04) · XOÁ

**Mã:** `experiments/graphfuse/run_gnn.py` · **Kết quả:** `results/gnn_patient_graph.json`
**Config:** đồ thị kNN k=10 trên tương đồng hợp nhất (trung bình RBF trên modality chung), dựng trên đặc
trưng của mọi nút trong fold; huấn luyện transductive (mất mát chỉ trên nút train); GraphSAGE gộp-trung-bình
2 lớp, hidden 8, dropout 0.3, Adam lr 0.01, weight_decay 1e-3, 125 epoch; giữ bộ lọc L1 radiomics theo vị
trí + RobustScaler theo modality; phân hoạch đúng harness, 5 seed × 10 fold.

| Tổ hợp | harness uniform | V0 không đồ thị (đối chứng) | V1 expert+graph | V2 gnn-mean | V3 gnn-attn (MAGNET) |
|---|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7730 | **0.7619** | 0.6991 | 0.7029 | 0.6840 |
| Rad+IHC-G+Gen | 0.7510 | **0.7368** | 0.6913 | 0.6808 | 0.6682 |
| Rad+Gen | 0.7160 | **0.7100** | 0.6848 | 0.6824 | 0.6832 |
| IHC-G+Gen | 0.7184 | **0.6938** | 0.6608 | 0.6524 | 0.6484 |
| PDL1+Gen | 0.7191 | 0.7197 | 0.7246 | **0.7440** | 0.7363 |
| **Δ so với V0 / thắng** | — | — | −0.0323 / 1 | −0.0319 / 1 | −0.0404 / 1 |

- V0 (cùng vòng lặp, không đồ thị) thấp hơn harness trung bình 0.011 do khác chi tiết huấn luyện → mọi so
  sánh đồ thị lấy V0 làm mốc để tách riêng tác động của đồ thị.
- **Cả ba biến thể đồ thị thua 4/5 tổ hợp, 0.025–0.078.** V1 khởi đầu ĐÚNG BẰNG V0 (lớp cuối = 0) mà vẫn
  tụt 0.063 ở BM1 → nhánh đồ thị học ra nhiễu. Bản sát MAGNET nhất (V3, có attention) tệ nhất.
- **Chỉ thắng ở PDL1+Gen** — lần thứ TƯ cùng mẫu hình (TabPFN +0.024, coop +0.027, XGBoost +0.018,
  GNN +0.024). Kết hợp với cấu hình nhẹ (+0.033) và lr_late (0.7515): ở tổ hợp này chính baseline đang
  yếu, chứ không phải các phương pháp kia giỏi.
- **VERDICT: XOÁ.** Khép câu hỏi về họ đồ thị bệnh nhân.

### 2.21. Học đa nhiệm đáp ứng + sống còn (Cox / ordinal BOR) (2026-10-04) · XOÁ

**Mã:** `experiments/modelimp/{model_mt,run_mt}.py`
**Ý tưởng:** thêm TÍN HIỆU GIÁM SÁT thay vì tham số. Nhãn nhị phân gộp 185 ca "không đáp ứng" làm một
dù rất không đồng nhất (SD n=48 PFS trung vị 7.05 th; PD n=92 1.75 th; mã BOR 5–8 n=45 ~1–2.5 th).
`L = BCE + λ_cox·Cox(PFS) + λ_ord·xếp-hạng-cặp(CR/PR < SD < PD)`, chỉ dùng PFS/BOR của bệnh nhân train.
**Config:** backbone `uniform_avg`, MODEL_PARAMS chuẩn, BM1, seed 42 (sàng lọc liều–đáp ứng).
λ=0 tái lập ĐÚNG baseline (AUC 0.7927, C-index 0.6306).

| | AUC đáp ứng | Δ | C-index PFS | Δ |
|---|---|---|---|---|
| nền (λ=0) | 0.7927 | — | 0.6306 | — |
| + Cox λ=0.1 | 0.7750 | −0.018 | 0.6241 | −0.007 |
| + Cox λ=0.3 | 0.7527 | −0.040 | 0.6225 | −0.008 |
| + Cox λ=1.0 | 0.7210 | −0.072 | 0.6008 | −0.030 |
| + Ord λ=0.1 | 0.7853 | −0.007 | 0.6258 | −0.005 |
| + Ord λ=0.3 | 0.7679 | −0.025 | 0.6185 | −0.012 |
| + Ord λ=1.0 | 0.7647 | −0.028 | 0.6166 | −0.014 |

- **Liều–đáp ứng đơn điệu, cả 6 mức đều âm:** càng thêm giám sát phụ càng tệ, và tệ ở CẢ HAI chỉ số —
  thêm mất mát Cox làm giảm chính C-index. Không chạy đủ 5 seed vì chiều và độ dốc đã rõ.
- Diễn giải: (1) mục tiêu xung đột — SD có PFS dài nhưng mang nhãn "không đáp ứng", Cox kéo họ về
  nguy cơ thấp; (2) mô hình nền đang ở chế độ "huấn luyện non" có tác dụng như regularization
  (xem §2.16: alpha/epoch/lr đều nhạy) — thêm bất kỳ số hạng mất mát nào cũng làm khớp mạnh hơn → overfit.
- Giả thuyết "thêm thông tin thì thắng" KHÔNG đúng cho thông tin phía nhãn; chỉ đúng cho phía đầu vào
  (NLP, mẫu khuyết).
- **VERDICT: XOÁ.**

### 2.22. ★ Tăng cường nhiễu loạn radiomics + đường cong ensemble + con số chốt 20 seed (2026-10-04)

**Mã:** `experiments/modelimp/{run_pertaug,run_enscurve,run_stack,run_stack20}.py`
**Kết quả:** `results/{pertaug,ensemble_curve,final_stack,final_stack20}.json`
**Config:** backbone `uniform_avg`, MODEL_PARAMS chuẩn, 10-fold, phân hoạch đúng harness.

**A. Tăng cường nhiễu loạn (PertAug).** Mỗi tổn thương có 10 bản đặc trưng tính lại dưới nhiễu loạn phân
vùng (`pertubation-radiomics`) — pipeline gốc chỉ dùng để lọc đặc trưng rồi bỏ. Ở đây: mỗi epoch, radiomics
của mỗi bệnh nhân train rút ngẫu nhiên từ {gốc + 10 bản}; số bước tối ưu không đổi. Đã kiểm: 163/21/67
bệnh nhân đều đủ 10 bản; lệch nhiễu loạn ≈ 0.08 (PC), 0.22 (LN) × sd giữa bệnh nhân; tắt aug tái lập ĐÚNG
baseline (0.7927 seed 42). 5 seed:

| Tổ hợp | nền | aug | Δ | aug+TTA | Δ | ens nền → aug |
|---|---|---|---|---|---|---|
| BM1 | 0.7730 | 0.7785 | +0.0055 | 0.7755 | +0.0025 | 0.7838 → 0.7895 |
| Rad+IHC-G+Gen | 0.7510 | 0.7512 | +0.0002 | 0.7526 | +0.0016 | 0.7631 → 0.7657 |
| Rad+Gen | 0.7160 | 0.7159 | −0.0001 | 0.7164 | +0.0004 | 0.7291 → 0.7313 |
| Rad | 0.6661 | 0.6799 | +0.0138 | 0.6776 | +0.0115 | 0.6925 → 0.6938 |

→ Δ TB +0.0048 (3/4), ensemble +0.0029 (4/4). Thay đổi tầng huấn luyện ĐẦU TIÊN không hại ở tổ hợp nào.

**B. Đường cong ensemble (20 seed):**

| Tổ hợp | 1 mô hình | K=3 | K=5 | K=10 | K=15 | K=20 |
|---|---|---|---|---|---|---|
| BM1 | 0.7688 ± 0.014 | 0.7791 | 0.7821 | 0.7841 | 0.7847 | 0.7844 |
| Rad+IHC-G+Gen | 0.7458 ± 0.015 | 0.7571 | 0.7604 | 0.7626 | 0.7636 | 0.7646 |
| Rad+Gen | 0.7092 ± 0.015 | 0.7195 | 0.7220 | 0.7241 | 0.7252 | 0.7238 |

→ Bão hoà ở K≈10 (1→5: +0.013…0.015; 5→20: +0.002…0.004).
→ **CẢNH BÁO: 5 seed chuẩn hơi may ở cả ba tổ hợp** (BM1 0.7730 vs 0.7688; R+G+Gen 0.7510 vs 0.7458;
  Rad+Gen 0.7160 vs 0.7092 → lệch +0.004…+0.007, cùng chiều). Mọi số "5-seed" tuyệt đối (kể cả Bảng 4 của
  bài) nhiều khả năng lạc quan ~0.005; so sánh giữa các phương pháp trên cùng seed thì không bị ảnh hưởng.

**C. CON SỐ CHỐT — chồng đầy đủ trên BM1, 20 seed, đối chứng ghép cặp:**

| Cấu hình | AUC 1 mô hình | ens K=5 | ens K=10 | ens K=20 |
|---|---|---|---|---|
| uniform_avg nền (mục B) | 0.7688 ± 0.014 | 0.7821 | 0.7841 | 0.7844 |
| + NLP + miss_pattern | 0.7884 ± 0.011 | 0.7952 | 0.7973 | 0.7977 |
| **+ NLP + miss + PertAug** | **0.7904 ± 0.013** | 0.7974 | 0.7995 | **0.7998** |

- NLP + miss so với nền: **+0.0196** trên 20 seed (số 5-seed 0.7896 đứng vững: 0.7884).
- PertAug ghép cặp 20 seed: **Δ = +0.0021 ± 0.0055, thắng 11/20, Wilcoxon p = 0.23** → xu hướng dương nhỏ,
  KHÔNG có ý nghĩa thống kê. TTA hại trong chồng đầy đủ (5 seed: 0.7833 vs 0.7919) → bỏ.
- Phụ: dùng CẢ HAI bộ pathology (IHC-G + IHC-A) cùng lúc ở BM1: 0.7661 / ens 0.7786 (−0.007) → không giúp.
- **VERDICT:** GIỮ ensemble (K=10 là đủ). PertAug: giữ như thành phần tuỳ chọn, phải ghi rõ p=0.23.
  Con số tiêu đề trung thực: **≈0.80 (0.7977 không aug / 0.7998 có aug), đo trên 20 seed.**

### 2.23. ★★ Mô hình đề xuất trên backbone OvO: OvO + NLP + miss_pattern — tổ hợp nhiều nguồn (2026-10-04)

**Mã:** `experiments/modelimp/run_ovo_stack.py` · **Kết quả:** `results/ovo_stack.json`,
điểm từng bệnh nhân `results/scores/ovostack_*__PAPER.csv`
**Config:** `train_ovo_safe` (MultiModalDynamicModelOvO), MODEL_PARAMS chuẩn, 5-seed × 10-fold, seeds chuẩn,
`build_l1` + `clean_rad_filters` như §2.9, `no_scale` cho `cnl_nlp` và `miss_pattern`. Discovery n=247.
**Tiêu chí (người dùng chốt 2026-10-04):** mô hình tốt hơn KHÔNG cần thắng ở mọi tổ hợp — chỉ cần cao nhất
và thắng ở các tổ hợp NHIỀU NGUỒN. Phạm vi cố định trước: các tổ hợp k ≥ 5 nguồn (#14–#21; #20/#21 trùng
#17/#18 khi NLP thay Labs → 6 cấu hình phân biệt).

| Tổ hợp | DyAM | OvO | OvO+NLP | **OvO+NLP+miss** | + ensemble | paired bootstrap vs DyAM (2000) |
|---|---|---|---|---|---|---|
| Rad+IHC-A+Gen | 0.7325 | 0.7325 | 0.7439 | **0.7624 ± 0.016** | 0.7746 | Δ +0.0255 [−0.018, +0.068] p=0.232 |
| Rad+IHC-G+Gen | 0.7529 | 0.7509 | 0.7595 | **0.7778 ± 0.016** | 0.7897 | Δ +0.0218 [−0.012, +0.058] p=0.212 |
| Rad+IHC-A+MutAmp | 0.7289 | 0.7403 | 0.7474 | **0.7552 ± 0.019** | 0.7622 | Δ +0.0182 [−0.014, +0.051] p=0.244 |
| Rad+IHC-A+Gen+PDL1 | 0.7399 | 0.7508 | 0.7569 | **0.7629 ± 0.010** | 0.7728 | Δ +0.0175 [−0.019, +0.057] p=0.334 |
| **Rad+IHC-G+Gen+PDL1 (BM1)** | 0.7599 | 0.7704 | 0.7822 | **0.7874 ± 0.012** | **0.7933** | Δ +0.0169 [−0.018, +0.053] p=0.317 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7413 | 0.7581 | 0.7648 | **0.7712 ± 0.012** | 0.7785 | Δ +0.0220 [−0.014, +0.059] p=0.228 |

(Δ bootstrap tính trên điểm trung bình qua seed của hai mô hình; cột AUC là trung bình AUC từng seed.)

- **Thắng DyAM gốc 6/6 tổ hợp nhiều nguồn**, Δ trung bình AUC +0.0269 (từ +0.023 đến +0.030), rất đều.
- **Không tổ hợp nào đạt ý nghĩa thống kê riêng lẻ** (p = 0.21–0.33; mọi CI chứa 0) — đúng dự báo của Note S3
  (cần ~1.200 bệnh nhân cho Δ≈0.02). 6 tổ hợp dùng chung bệnh nhân nên KHÔNG gộp p-value được.
- Thang ablation ở BM1: DyAM 0.7599 → OvO 0.7704 (+0.011) → +NLP 0.7822 (+0.012) → +miss 0.7874 (+0.005)
  → ensemble 0.7933 (+0.006). Đơn điệu ở 5/6 tổ hợp (ngoại lệ nhỏ: #15 bước OvO −0.002).
- So với cùng chồng trên backbone `uniform_avg` (§2.14): Δ TB −0.0029, thắng 1/6 → hai backbone ngang nhau.
- Trong nhóm nhiều nguồn, OvO hơn DyAM trung bình +0.008 (thắng 6/8, hoà 1, thua 1).
- **VERDICT: GIỮ — đây là mô hình đề xuất cho bài báo** theo tiêu chí của người dùng: có cơ chế attention
  (OvO) để trình bày, thắng DyAM ở mọi tổ hợp nhiều nguồn, giá trị cao nhất 0.7874 / 0.7933 (ensemble).
  Khi viết: nêu rõ p-value, và nêu rõ phần tăng đến chủ yếu từ NLP + mẫu khuyết.

### 2.24. ★★ Kiểm chứng: OvO có thực sự tốt hơn ở tổ hợp nhiều nguồn? — 20 seed, 5 phép kiểm (2026-10-04)

**Mã:** `experiments/ovo_proof/{run_runs,analyze}.py` · **Kết quả:** `results/ovo_proof/*.csv` (điểm từng
bệnh nhân, 3 mô hình × 6 tổ hợp × 20 seed = 360 lượt) và `results/ovo_proof/tests.json`
**Config:** MODEL_PARAMS chuẩn, 10-fold, 20 seed (5 chuẩn + 15 mới), CÙNG phân hoạch cho cả ba mô hình mỗi
seed; `train_dyam` (allcombo.run_combo), `train_ovo_safe`, `train_uniform_avg`; bộ lọc `make_filter` như
`run_paper_combos`; vá scipy. So sánh THUẦN cơ chế hợp nhất (không NLP, không miss).
**Phạm vi + phép kiểm cố định TRƯỚC khi chạy:** 6 tổ hợp k ≥ 5; (1) Wilcoxon ghép cặp trên 20 seed;
(2) bootstrap ghép cặp trên bệnh nhân, 5000; (3) DeLong 2 AUC tương quan; (4) hoán vị tráo điểm, 5000;
(5) bootstrap gộp ΔAUC trung bình trên 6 tổ hợp. (2)–(5) dùng điểm ensemble 20 seed.
Bản cài DeLong đã kiểm: AUC khớp sklearn 6 chữ số; sai lầm loại I = 4.2%; công suất 93.7% khi khác thật.

**AUC (20 seed):**

| Tổ hợp | k | DyAM 1 mô hình | OvO 1 mô hình | uniform 1 mô hình | ens DyAM / OvO / uniform |
|---|---|---|---|---|---|
| Rad+IHC-A+Gen | 5 | 0.7280 ± 0.021 | 0.7297 ± 0.017 | 0.7289 ± 0.017 | 0.7513 / 0.7494 / 0.7477 |
| Rad+IHC-G+Gen | 5 | 0.7539 ± 0.016 | 0.7461 ± 0.016 | 0.7458 ± 0.015 | 0.7716 / 0.7637 / 0.7646 |
| Rad+IHC-A+MutAmp | 6 | 0.7296 ± 0.020 | 0.7414 ± 0.015 | 0.7442 ± 0.016 | 0.7471 / 0.7543 / 0.7578 |
| Rad+IHC-A+Gen+PDL1 | 6 | 0.7411 ± 0.015 | 0.7491 ± 0.016 | 0.7509 ± 0.016 | 0.7651 / 0.7645 / 0.7685 |
| Rad+IHC-G+Gen+PDL1 (BM1) | 6 | 0.7600 ± 0.019 | 0.7667 ± 0.015 | 0.7688 ± 0.014 | 0.7799 / 0.7833 / 0.7844 |
| Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7443 ± 0.016 | 0.7622 ± 0.014 | 0.7656 ± 0.015 | 0.7650 / 0.7779 / 0.7799 |
| **Trung bình** | | 0.7428 ± 0.0177 | 0.7492 ± 0.0157 | 0.7507 ± 0.0153 | 0.7633 / 0.7655 / 0.7671 |

**OvO − DyAM:**

| Tổ hợp | Δ/seed | seed OvO thắng | Wilcoxon | Δ ensemble | CI 95% bootstrap | p boot | p DeLong | p hoán vị |
|---|---|---|---|---|---|---|---|---|
| Rad+IHC-A+Gen | +0.0018 | 10/20 | 0.70 | −0.0018 | [−0.013, +0.009] | 0.73 | 0.73 | 0.73 |
| Rad+IHC-G+Gen | **−0.0078** | 5/20 | 0.0014 | **−0.0078** | [−0.015, −0.001] | **0.025** | **0.020** | **0.011** |
| Rad+IHC-A+MutAmp | +0.0117 | 14/20 | 0.019 | +0.0072 | [−0.003, +0.018] | 0.18 | 0.17 | 0.17 |
| Rad+IHC-A+Gen+PDL1 | +0.0080 | 13/20 | 0.13 | −0.0006 | [−0.020, +0.017] | 0.96 | 0.95 | 0.95 |
| Rad+IHC-G+Gen+PDL1 | +0.0067 | 16/20 | 0.008 | +0.0035 | [−0.011, +0.018] | 0.63 | 0.61 | 0.61 |
| Rad+IHC-A+Gen+TMB+PDL1 | +0.0179 | 19/20 | <0.0001 | +0.0129 | [−0.007, +0.035] | 0.22 | 0.22 | 0.21 |
| **GỘP 6 tổ hợp** | **+0.0064** | **14/20** | **0.008** | **+0.0022** | **[−0.005, +0.010]** | **0.57** | | |

- **Ở mức MỘT MÔ HÌNH (ổn định qua phân hoạch): OvO hơn DyAM** — +0.0064, thắng 14/20 seed, Wilcoxon p=0.008;
  3/6 tổ hợp có ý nghĩa nghiêng về OvO, 1/6 có ý nghĩa nghiêng về DyAM (Rad+IHC-G+Gen), 2 hoà.
  Lợi thế TĂNG theo số nguồn: k=5 trung bình −0.003, k=6 +0.009, k=7 +0.018 (19/20 seed).
- **Ở mức BỆNH NHÂN (thứ khái quát sang bệnh nhân mới): KHÔNG có bằng chứng** — Δ gộp +0.0022,
  CI [−0.005, +0.010], p=0.57; không tổ hợp nào có ý nghĩa nghiêng về OvO; một tổ hợp có ý nghĩa nghiêng
  về DyAM (p=0.011–0.025 ở cả ba phép kiểm).
- **Lợi thế của OvO là về ĐỘ ỔN ĐỊNH, không phải độ chính xác:** sd giữa các seed 0.0157 vs 0.0177; ensemble
  đem lại cho DyAM +0.0205 nhưng chỉ +0.0163 cho OvO → khoảng cách co từ +0.0064 xuống +0.0022 sau ensemble.
- **OvO − uniform_avg:** Δ/seed −0.0015, OvO chỉ thắng 4/20 seed (Wilcoxon p=0.0009); Δ ensemble gộp −0.0016,
  CI [−0.006, +0.002], p=0.43 → uniform nhỉnh hơn một cách NHẤT QUÁN nhưng rất nhỏ, không có ý nghĩa ở mức
  bệnh nhân. Giá trị cao nhất (BM1 ensemble): uniform 0.7844 ≥ OvO 0.7833 ≥ DyAM 0.7799.
- **uniform − DyAM:** Δ/seed +0.0079 (16/20, p=0.002); Δ ensemble gộp +0.0038, p=0.45.
- Ghi chú: 5 seed chuẩn đã phóng đại lợi thế của OvO (BM1: +0.0105 → +0.0067; Rad+IHC-G+Gen: −0.002 → −0.008).
- **VERDICT:** phát biểu được số liệu ủng hộ = "OvO ổn định hơn DyAM và, ở mức một mô hình, nhỉnh hơn tại
  các cấu hình nhiều nguồn, lợi thế tăng theo số nguồn". KHÔNG phát biểu được "OvO cho AUC cao hơn DyAM"
  như một kết luận về bệnh nhân mới.

### 2.25. Vòng lặp tìm kiến trúc — đợt 1, C1/C2/C3/C10: quy tắc hợp nhất không tham số (2026-10-04) · XOÁ

**Giao thức + sổ cái:** `document/2026-10-04_arch-search-loop/{protocol,ledger}.md` · harness
`experiments/archloop/harness.py` (đối chứng fusion=mean tái lập ĐÚNG 0.7927 ở seed 42 BM1).
**Config sàng lọc:** backbone chuyên gia `tanh(Wx)` của uniform_avg, MODEL_PARAMS chuẩn, 3 tổ hợp nhiều nguồn
× 5 seed chuẩn, ghép cặp với `uniform_avg` cùng seed (điểm ở `results/ovo_proof/`). Ngưỡng qua: Δ ≥ +0.005.

| Ứng viên | BM1 | Rad+IHC-A+Gen+TMB+PDL1 | Rad+IHC-G+Gen | Δ TB |
|---|---|---|---|---|
| C1 phân cấp theo miền (3 vị trí radiomics = 1 phiếu) | +0.0063 | −0.0012 | +0.0008 | +0.0020 |
| C2 chuẩn hoá z điểm chuyên gia | −0.0176 | −0.0382 | −0.0480 | −0.0346 |
| C3 trung vị | −0.1000 | −0.0850 | −0.1113 | −0.0988 |
| C10 tổng (không chia số nguồn) | −0.0461 | −0.0310 | −0.0334 | −0.0368 |

- Không ứng viên nào qua sàng lọc. Code + điểm thô đã xoá theo giao thức.
- **Phát hiện đáng giữ:** trung bình các điểm tanh không thật sự "đều" — độ lớn |r_i| mã hoá độ tự tin của
  từng chuyên gia, nên mô hình đã có attention NGẦM theo độ lớn. Chuẩn hoá z phá nó (−0.035). Đây là lời
  giải thích cơ chế cho việc mọi attention tường minh không thêm được gì.
- **VERDICT: XOÁ cả bốn.**

### 2.26. Vòng lặp tìm kiến trúc — đợt 1, C15/C8/C7/C16/C4 (2026-10-04) · XOÁ

**Giao thức + sổ cái:** `document/2026-10-04_arch-search-loop/{protocol,ledger}.md`. Sàng lọc: 3 tổ hợp nhiều
nguồn × 5 seed chuẩn, ghép cặp với `uniform_avg`, ngưỡng qua Δ ≥ +0.005. Mọi ứng viên có chế độ tắt tái lập
ĐÚNG 0.7927 (seed 42, BM1). MODEL_PARAMS chuẩn.

| Ứng viên | BM1 | #19 (7 nguồn) | Rad+IHC-G+Gen | Δ TB |
|---|---|---|---|---|
| C15 PCA radiomics không giám sát (10 chiều, fit cả bản nhiễu loạn) | −0.0109 | −0.0027 | −0.0097 | −0.0077 |
| C8 chuyên gia phi tuyến dạng dư (MLP 4 nút, dropout 0.5, khởi đầu = tuyến tính) | −0.0112 | +0.0001 | −0.0165 | −0.0092 |
| C7 mixup không gian điểm (Beta 0.4) | −0.0028 | −0.0043 | −0.0055 | −0.0042 |
| C16 gộp cohort ngoài chỉ-có-pathology (path_valid, 52 bệnh nhân) vào fold train | −0.0247 | +0.0073 | −0.0272 | −0.0148 |
| C4 chọn đặc trưng radiomics ổn định (8 bootstrap, giữ ≥5/8) — dừng sớm 9/15 lượt | −0.0076 | −0.0196 | — | ≈ −0.014 |

- Không ứng viên nào qua sàng lọc; code + điểm thô đã xoá.
- C16: GLCM pathology của cohort ngoài lệch phân bố (IHC-G −0.025/−0.027) dù IHC-A nhỉnh (+0.007).
- C4: chọn đặc trưng "ổn định" hơn lại làm MẤT tín hiệu → bước elastic-net theo từng fold của pipeline gốc
  không phải nguồn phương sai có hại như giả thuyết.
- **VERDICT: XOÁ cả năm.** Tổng 9/10 ứng viên kiến trúc/huấn luyện của đợt 1 thất bại.

### 2.27. ★★★ C18 — lấp PD-L1 thiếu bằng điểm lâm sàng, và CHỒNG ĐẦY ĐỦ (2026-10-04) · GIỮ

**Mã:** `experiments/archloop/{cands,harness,run_stack_final}.py` · **Điểm từng bệnh nhân:**
`results/ovo_proof/{c18_pdl1fill,c18stack}__*.csv` · **Kiểm định:** `results/ovo_proof/tests_c18{,stack}.json`
**Phát hiện dữ liệu:** kênh `cnl_pdl1_score` dùng điểm chấm lại (js), chỉ có ở 201/247 bệnh nhân; cột
`clinical_pdl1_score` trong omnibus có ở **246/247** (AUC đơn biến 0.738 vs 0.744). 45 bệnh nhân đang bị coi
là thiếu PD-L1 dù có sẵn điểm lâm sàng. C18 = dùng điểm chấm lại khi có, không thì lấy điểm lâm sàng (−TPS/100).
**Config:** backbone `uniform_avg`, MODEL_PARAMS chuẩn, 20 seed × 10-fold, phân hoạch đúng harness; chế độ tắt
tái lập đúng 0.7927. ĐÂY LÀ THÊM DỮ LIỆU, KHÔNG PHẢI KIẾN TRÚC.

**C18 so với uniform_avg (5 phép kiểm, 20 seed):**

| Tổ hợp | Δ/seed | seed thắng | AUC ens C18 vs uni | Δ ens | p boot / DeLong / hoán vị |
|---|---|---|---|---|---|
| Rad+IHC-A+Gen, Rad+IHC-G+Gen (không PD-L1) | 0 | — | trùng khít | 0 | — |
| Rad+IHC-A+MutAmp | +0.0073 | 18/20 | 0.7616 vs 0.7578 | +0.0038 | 0.60 / 0.59 / 0.59 |
| Rad+IHC-A+Gen+PDL1 | +0.0145 | 19/20 | 0.7786 vs 0.7685 | +0.0100 | 0.22 / 0.21 / 0.21 |
| **BM1** | +0.0178 | **20/20** | **0.7994** vs 0.7844 | **+0.0150** | **0.053 / 0.050 / 0.042** |
| Rad+IHC-A+Gen+TMB+PDL1 | +0.0131 | 20/20 | 0.7909 vs 0.7799 | +0.0111 | 0.073 / 0.078 / 0.059 |
| **Gộp 6 tổ hợp (tiêu chí định trước)** | +0.0088 | 20/20 | | **+0.0067** | CI [−0.0018, +0.0155], **p=0.118** |

- **Theo tiêu chí chấp nhận định trước: KHÔNG ĐẠT** (cần Δ gộp ≥+0.010, p<0.05, thắng ≥5/6; đạt +0.0067,
  p=0.118, 4/6 + 2 hoà theo cấu trúc). Hai tổ hợp không có PD-L1 kéo trung bình gộp về 0.
- **Hậu nghiệm, chỉ 4 tổ hợp có PD-L1:** vs uniform +0.0100 [−0.003, +0.023] p=0.129; vs OvO +0.0126
  [−0.000, +0.026] p=0.054; **vs DyAM +0.0184 [+0.002, +0.035] p=0.028** (thắng 4/4, 19/20 seed).
  Chưa hiệu chỉnh đa so sánh (ứng viên thứ 36 trên cùng 247 bệnh nhân); vs DyAM gộp cả đổi backbone.

**CHỒNG ĐẦY ĐỦ = uniform_avg + NLP + miss_pattern + C18 (20 seed):**

| Tổ hợp | DyAM | uniform | C18 | **CHỒNG** | ensemble: DyAM / uniform / C18 / **chồng** |
|---|---|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.7296 | 0.7442 | 0.7515 | **0.7603 ± 0.010** | 0.7471 / 0.7578 / 0.7616 / **0.7681** |
| Rad+IHC-A+Gen+PDL1 | 0.7411 | 0.7509 | 0.7654 | **0.7751 ± 0.011** | 0.7651 / 0.7685 / 0.7786 / **0.7839** |
| **BM1 Rad+IHC-G+Gen+PDL1** | 0.7600 | 0.7688 | 0.7865 | **0.7957 ± 0.011** | 0.7799 / 0.7844 / 0.7994 / **0.8039** |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7443 | 0.7656 | 0.7787 | **0.7849 ± 0.009** | 0.7650 / 0.7799 / 0.7909 / **0.7919** |

- Chồng − DyAM: theo seed **+0.0353, thắng 20/20**; ensemble gộp +0.0227, CI [−0.009, +0.054], p=0.16;
  từng tổ hợp +0.019…+0.027 (p 0.12–0.36). Chồng − uniform: +0.0143 (p=0.29). Chồng − C18: +0.0043 (p=0.74).
- Nghịch lý đáng ghi: chồng có ước lượng điểm LỚN hơn C18 (+0.0227 vs +0.0184 so với DyAM) nhưng p KÉM hơn
  (0.16 vs 0.028) — thêm NLP/miss làm điểm của mô hình bớt tương quan với DyAM nên kiểm định ghép cặp mất
  công suất.
- **CON SỐ CHỐT MỚI CỦA DỰ ÁN: BM1 = 0.7957 ± 0.011 (một mô hình, 20 seed) / 0.8039 (ensemble 20)**,
  so với DyAM gốc 0.7600 / 0.7799.
- **VERDICT: GIỮ C18 và chồng đầy đủ.** Khi viết: nêu rõ C18 là mở rộng độ phủ dữ liệu; nêu p-value thật;
  phần Methods của bài vốn đã viết PD-L1 "lấy từ báo cáo pathology lâm sàng" nên C18 còn làm code khớp với
  mô tả hơn.

### 2.28. Vòng lặp tìm kiến trúc — ĐỢT 2: balanced multimodal learning + nhiệm vụ phụ (2026-10-04) · 9 XOÁ, 1 GIỮ (chưa xác nhận)

**Câu hỏi:** các phương pháp cân bằng modality trong tài liệu (OGM-GE CVPR 2022, MLA CVPR 2024, ReconBoost ICML 2024,
BalanceBenchmark 2025, arXiv 2304.00379) có giúp `uniform_avg` ở các tổ hợp nhiều nguồn không?

**Script / đầu ra:** `experiments/archloop/harness.py {screen,confirm}` + `experiments/archloop/cands.py`; giao thức và sổ cái ở
`document/2026-10-04_arch-search-loop/{protocol.md §7, ledger.md}`. Điểm thô của ứng viên thất bại đã xoá; D6b giữ ở
`experiments/results/ovo_proof/d6b_auxlow__*.csv`, kiểm định ở `experiments/results/ovo_proof/tests_d6b.json`.

**Cấu hình:** chuẩn §0 — `train_uniform_avg`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled=False`,
10-fold `KFold(shuffle=True, random_state=0)` trên chỉ số hoán vị theo seed, cohort discovery (n = 247), l1_filter radiomics chuẩn.
Sàng lọc: 3 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1, Rad+IHC-G+Gen) × 5 seed chuẩn. Xác nhận: 6 tổ hợp k ≥ 5 × 20 seed.
**Lệch chuẩn:** KHÔNG chạy đủ 21 tổ hợp (vòng lặp chỉ xét 6 tổ hợp nhiều nguồn theo tiêu chí người dùng); D6b bỏ qua bước sàng lọc
và có luật "chỉ IHC-A" đặt hậu nghiệm (ngoại lệ ghi trong ledger). Mọi ứng viên có chế độ tắt tái lập 0.7927 (seed 42, BM1).

**Chẩn đoán trước khi chạy** (BM1, 3 seed) — AUC chuyên gia trong mô hình chung / huấn luyện riêng: rad PC 0.631/0.639, PL 0.553/0.556,
LN 0.647/0.671, pathology **0.680/0.617**, genomics 0.666/0.674, PD-L1 0.714/0.715 → không có mất cân bằng modality.

| # | Ứng viên | Δ sàng lọc vs uniform | Kết luận |
|---|---|---|---|
| D5 | Cập nhật luân phiên từng chuyên gia (kiểu ReconBoost) | +0.0006 | XOÁ, trung tính |
| D2 | OGM-GE điều biến gradient, α = 0.3 | −0.0006 | XOÁ, trung tính |
| D1 | Giám sát phụ từng chuyên gia, λ = 0.5 | −0.0017 | XOÁ, trung tính |
| D10 | Chuyên gia tương tác chéo PD-L1 × genomics | +0.0030 (BM1 +0.0053, #19 +0.0038, mỗi cái 4/5) | XOÁ, sát ngưỡng nhất |
| D4 | Dropout 0.2 trên đặc trưng đầu vào | −0.0062 | XOÁ |
| D3 | Chuyên gia huấn luyện độc lập (MLA) | −0.0127 (thua 15/15) | XOÁ; huấn luyện chung hơn độc lập |
| D9 | Hợp nhất trước phi tuyến tanh(TB W_i x_i) | −0.0335 | XOÁ |
| D6 | Lớp ẩn 4 nút của chuyên gia pathology + nhiệm vụ phụ dự đoán PD-L1 lâm sàng, λ = 0.5 | −0.0043 (IHC-A +0.0137 5/5; IHC-G −0.010, −0.017) | XOÁ dạng tổng quát |
| D7 | Tự chưng cất: 5 thầy bootstrap trong fold, nhãn mềm out-of-bag, w = 0.5 | −0.0014 (BM1 −0.0039, #19 −0.0001, R+G+Gen −0.0001) | XOÁ, trung tính |
| **D6b** | D6 chỉ áp khi pathology ≤ 32 đặc trưng (IHC-A) | — | **GIỮ, chưa xác nhận** |

**D6b − uniform, 20 seed** (AUC ensemble; p bootstrap trên bệnh nhân): Rad+IHC-A+Gen 0.7602 vs 0.7477 (+0.0126, p 0.048) ·
Rad+IHC-A+MutAmp 0.7690 vs 0.7578 (+0.0112, p 0.067) · Rad+IHC-A+Gen+PDL1 0.7776 vs 0.7685 (+0.0091, p 0.140) ·
Rad+IHC-A+Gen+TMB+PDL1 0.7892 vs 0.7799 (+0.0093, p 0.170) · hai tổ hợp IHC-G: 0 (trùng uniform).
Gộp 6 tổ hợp **+0.0070 [−0.0009, +0.0152], p = 0.078**, thắng 4/6; gộp 4 tổ hợp IHC-A +0.0105, p = 0.072.
Theo seed: +0.0081 TB 6 tổ hợp, 19/20, Wilcoxon p < 1e-5; trên 15 seed mới (4 tổ hợp IHC-A) +0.0115, 14/15.
D6b − DyAM: gộp 6 tổ hợp +0.0109, p = 0.121; gộp 4 tổ hợp IHC-A +0.0169 [+0.0003, +0.0352], p = 0.046, 20/20 seed.

**Đối chứng** (4 tổ hợp IHC-A × 20 seed, Δ AUC một mô hình vs uniform): chỉ lớp ẩn −0.0035 (8/20) · đích phụ bị xáo ngẫu nhiên
**+0.0084** (16/20) · PD-L1 thật +0.0121 (19/20). PD-L1 thật − đích xáo = +0.0037 (18/20, p = 0.0003).
→ khoảng 70% mức tăng là điều chuẩn chung của đầu hồi quy phụ; phần gắn với ý nghĩa sinh học của PD-L1 chỉ +0.004.

**Verdict:** ĐỢT 2 KHÔNG có ứng viên nào đạt tiêu chí chấp nhận. Ba phương pháp cân bằng modality trung tính (mô hình không mắc
hiện tượng mà chúng chữa). D6b là thay đổi kiến trúc ĐẦU TIÊN dương nhất quán qua seed (+0.012 trên tổ hợp IHC-A) nhưng trượt
ngưỡng (+0.0070 < +0.010, p = 0.078, 4/6), không tác động BM1 nên KHÔNG nâng AUC cao nhất, và luật IHC-A là hậu nghiệm → positive
yếu, chưa xác nhận; không đưa vào bài báo như một kết quả thắng.

### 2.29. D6b chồng lên bộ tốt nhất (NLP + miss_pattern + C18) trên tổ hợp IHC-A (2026-10-04) · CỘNG DỒN MỘT NỬA, chưa xác nhận

**Câu hỏi:** nhiệm vụ phụ D6b (§2.28) có cộng dồn với bộ tốt nhất `uniform_avg + NLP + miss_pattern + C18` (§2.27) không?

**Script / đầu ra:** `experiments/archloop/run_stack_d6b.py` → `experiments/results/ovo_proof/d6bstack__{slug}.csv` (điểm từng bệnh
nhân, 20 seed); kiểm định ở `experiments/results/ovo_proof/tests_d6bstack.json`. Mốc so: `c18stack__{slug}.csv` cùng seed.

**Cấu hình:** chuẩn §0 — `train_uniform_avg` (qua `train_pdl1fill` + `ModelAux`, hid 4 nút, λ = 0.5), epochs 125, lr 0.01, alpha 0.001,
beta 0.0, `cross_modality_enabled=False`, 10-fold shuffle trên chỉ số hoán vị theo seed, 20 seed, cohort discovery, l1_filter radiomics
chuẩn; thêm modality `cnl_nlp` và `miss_pattern` (`no_scale`). **Lệch chuẩn:** chỉ 3 tổ hợp IHC-A có sẵn mốc c18stack
(Rad+IHC-A+MutAmp, Rad+IHC-A+Gen+PDL1, Rad+IHC-A+Gen+TMB+PDL1), không phải 21 tổ hợp; 20 seed thay vì 5.

| Tổ hợp | DyAM | uniform | bộ tốt nhất (c18stack) | **+ D6b** | Δ vs c18stack (seed thắng) | Δ ensemble, p bootstrap |
|---|---|---|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.7296 ± 0.020 | 0.7442 ± 0.016 | 0.7603 ± 0.010 | **0.7667 ± 0.010** | +0.0064 (18/20) | 0.7718 vs 0.7681, +0.0037, p 0.50 |
| Rad+IHC-A+Gen+PDL1 | 0.7411 ± 0.015 | 0.7509 ± 0.016 | 0.7751 ± 0.011 | **0.7814 ± 0.009** | +0.0063 (16/20) | 0.7893 vs 0.7839, +0.0054, p 0.35 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7443 ± 0.016 | 0.7656 ± 0.015 | 0.7849 ± 0.009 | **0.7890 ± 0.009** | +0.0041 (15/20) | 0.7947 vs 0.7919, +0.0028, p 0.67 |

Gộp 3 tổ hợp vs c18stack: Δ ensemble +0.0040 [−0.0062, +0.0151], p = 0.48; theo seed +0.0056, 16/20, Wilcoxon p = 0.0003.
Vs DyAM: theo seed +0.0407, 20/20; Δ ensemble gộp +0.0262 [−0.0095, +0.0635], p = 0.15.

**Verdict:** positive yếu. D6b cộng dồn khoảng MỘT NỬA (+0.006 thay vì +0.012 khi đứng riêng) — NLP và miss_pattern đã lấy bớt phần
lợi ích; ổn định qua seed nhưng không có ý nghĩa ở mức bệnh nhân. Không đổi AUC cao nhất của dự án: BM1 (IHC-G) vẫn 0.7957 /
ens 0.8039 (§2.27); tổ hợp IHC-A tốt nhất lên 0.7890 / ens 0.7947, vẫn dưới BM1. GIỮ code, không tuyên bố thắng.

### 2.30. Đợt dữ liệu đầu vào — biến nền chưa dùng trong omnibus, trên bộ tốt nhất (2026-10-04) · XOÁ cả hai

**Câu hỏi:** năm biến nền chưa dùng (`sex`, `smoking_status`, `histo` vảy/không vảy, `n_lesions`, `pdl1_tiss_site` phổi/hạch/khác)
có nâng được bộ tốt nhất `uniform_avg + NLP + miss_pattern + C18` (§2.27) không?

**Script / đầu ra:** `experiments/archloop/run_data_e.py {e1,e2} --seeds=5 --only=0,1` (ĐÃ XOÁ sau khi thất bại, cùng điểm thô
`e{1,2}stack__*.csv` và cache `datasets/nlp_clinical_embed_v3.parquet`). Mốc so: `experiments/results/ovo_proof/c18stack__*.csv` cùng seed.

**Cấu hình:** chuẩn §0 — `train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0,
`cross_modality_enabled=False`, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery, l1_filter radiomics chuẩn, modality
`cnl_nlp` + `miss_pattern` (`no_scale`). **Lệch chuẩn:** chỉ sàng lọc 2 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1) × 5 seed chuẩn, không
chạy 21 tổ hợp vì cả hai ứng viên trượt sàng lọc.

| Ứng viên | BM1 (mốc 0.7956) | Rad+IHC-A+Gen+TMB+PDL1 (mốc 0.7808) | Δ TB |
|---|---|---|---|
| E1 — thêm chuyên gia số `cnl_extra` (7 đặc trưng: nữ, chưa từng hút, đang hút, vảy, n_lesions, hạch, vị trí khác) | 0.7798 (−0.0158, 0/5) | 0.7668 (−0.0140, 0/5) | **−0.0149** |
| E2 — nối các biến đó vào câu mô tả rồi nhúng lại bằng MiniLM (cosine với embedding cũ 0.954) | 0.7956 (+0.0000, 2/5) | 0.7810 (+0.0001, 2/5) | **+0.0001** |

**Verdict:** negative. Một chuyên gia yếu (AUC đơn biến ≤ 0.60) nhận trọn một phiếu trong phép trung bình nên kéo cả mô hình xuống
(cùng cơ chế vì sao radiomics PL làm hại); đưa cùng thông tin qua câu chữ thì vô hại nhưng cũng vô ích. Các biến nền còn lại trong
omnibus KHÔNG mang thêm tín hiệu dùng được. Không dùng tham số máy CT / scanner / mã định danh (AUC đơn biến tới 0.61 nhưng là nhiễu
kỹ thuật và xu hướng thời gian — đáng nêu ở phần hạn chế). Bộ tốt nhất vẫn là §2.27: BM1 0.7957 ± 0.011, ensemble 0.8039.

### 2.31. ★★ Mô hình ĐẦY ĐỦ trên cả 21 tổ hợp + thang ablation BM1 tách bước ở 20 seed (2026-10-04) · GIỮ

**Câu hỏi:** theo Full evaluation rule — mô hình đầy đủ (`uniform_avg` + NLP lâm sàng + `miss_pattern` + C18) so với DyAM gốc trên
toàn bộ 21 tổ hợp của bài; và tách riêng đóng góp của NLP với `miss_pattern` ở BM1 trên 20 seed.

**Script / đầu ra:** `experiments/archloop/run_full21.py all --part=i/2` → `experiments/results/full21/{slug}.json` (mỗi tổ hợp một
file); `run_full21.py nlp20` → `experiments/results/ovo_proof/uninlp__Rad_IHCG_Gen_PDL1.csv`. Mốc so: `experiments/results/paper_combos.json`
(DyAM, uniform; §2.10) và `experiments/results/missingness.json` (NLP+miss; §2.14), cùng 5 seed.

**Cấu hình:** chuẩn §0 — `train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0,
`cross_modality_enabled=False`, 5 seed chuẩn × 10-fold (shuffle trên chỉ số hoán vị theo seed), cohort discovery, l1_filter radiomics
theo `make_filter`; `cnl_nlp` + `miss_pattern` với `no_scale`; `cnl_dem_labs` thô được thay bằng NLP (nên #20/#21 trùng #17/#18).
**Lưu ý so sánh:** mô hình đầy đủ luôn chấm cả 247 bệnh nhân (NLP và miss có ở mọi người); DyAM ở #2, #3, #5, #6, #9, #10 chỉ chấm
bệnh nhân có dữ liệu (cột n) → ở các dòng đó hai AUC tính trên HAI TẬP KHÁC NHAU, không so trực tiếp được.

| # | Tổ hợp | k | n (DyAM) | DyAM | uniform | NLP+miss | **Đầy đủ** (n = 247) | ens 5 | Δ vs DyAM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | 247 | 0.6161 | 0.6161 | 0.6103 | 0.6103 ± 0.002 | 0.6112 | −0.0057 |
| 2 | PDL1 | 1 | 201 | 0.7198 | 0.7198 | 0.6609 | 0.7007 ± 0.013 | 0.6951 | −0.0190 |
| 3 | IHC-A | 1 | 105 | 0.6352 | 0.6352 | 0.6656 | 0.6656 ± 0.020 | 0.6574 | +0.0304 |
| 4 | Gen | 1 | 247 | 0.6763 | 0.6763 | 0.6667 | 0.6667 ± 0.008 | 0.6671 | −0.0095 |
| 5 | Rad | 3 | 187 | 0.6692 | 0.6661 | 0.6549 | 0.6549 ± 0.016 | 0.6602 | −0.0143 |
| 6 | Rad-LU | 4 | 187 | 0.6465 | 0.6463 | 0.6491 | 0.6491 ± 0.019 | 0.6554 | +0.0026 |
| 7 | TMB+PDL1 | 2 | 247 | 0.6642 | 0.7096 | 0.7023 | 0.7441 ± 0.002 | 0.7475 | +0.0799 |
| 8 | PDL1+Gen | 2 | 247 | 0.7072 | 0.7191 | 0.7388 | 0.7636 ± 0.006 | 0.7653 | +0.0564 |
| 9 | Rad+IHC-A | 4 | 211 | 0.6846 | 0.6836 | 0.7323 | 0.7323 ± 0.010 | 0.7412 | +0.0478 |
| 10 | Rad+IHC-G | 4 | 211 | 0.7112 | 0.7089 | 0.7609 | 0.7609 ± 0.018 | 0.7694 | +0.0498 |
| 11 | Rad+Gen | 4 | 247 | 0.7093 | 0.7160 | 0.7151 | 0.7151 ± 0.012 | 0.7219 | +0.0058 |
| 12 | IHC-A+Gen | 2 | 247 | 0.6934 | 0.6983 | 0.7295 | 0.7295 ± 0.009 | 0.7331 | +0.0361 |
| 13 | IHC-G+Gen | 2 | 247 | 0.7470 | 0.7184 | 0.7697 | 0.7697 ± 0.007 | 0.7729 | +0.0227 |
| 14 | Rad+IHC-A+Gen | 5 | 247 | 0.7325 | 0.7307 | 0.7652 | 0.7652 ± 0.008 | 0.7761 | +0.0327 |
| 15 | Rad+IHC-G+Gen | 5 | 247 | 0.7529 | 0.7510 | 0.7871 | 0.7871 ± 0.014 | 0.7956 | +0.0343 |
| 16 | Rad+IHC-A+MutAmp | 6 | 247 | 0.7289 | 0.7425 | 0.7525 | 0.7579 ± 0.010 | 0.7650 | +0.0290 |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 247 | 0.7399 | 0.7547 | 0.7673 | 0.7736 ± 0.014 | 0.7813 | +0.0337 |
| 18 | **Rad+IHC-G+Gen+PDL1 (BM1)** | 6 | 247 | 0.7599 | 0.7730 | 0.7896 | **0.7956 ± 0.011** | **0.8041** | +0.0357 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 247 | 0.7413 | 0.7618 | 0.7726 | 0.7808 ± 0.008 | 0.7878 | +0.0395 |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 247 | 0.7360 | 0.7455 | (= #17) | (= #17) 0.7736 | 0.7813 | +0.0376 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs | 7 | 247 | 0.7592 | 0.7686 | (= #18) | (= #18) 0.7956 | 0.8041 | +0.0364 |

- Đầy đủ > DyAM ở **17/21** tổ hợp; > uniform 16/21. Trung bình 21 tổ hợp: DyAM 0.7062 · uniform 0.7115 · NLP+miss 0.7261 · đầy đủ 0.7330.
- Theo số nguồn: k = 1 thắng 1/4 (Δ TB −0.001; ba dòng thua là TMB, PDL1, Gen — thêm NLP/miss vào một chỉ dấu đơn lẻ làm loãng nó;
  #2 còn khác tập bệnh nhân) · k = 2–4 thắng 8/9 (+0.032; thua duy nhất là Rad) · **k ≥ 5 thắng 8/8 (+0.035, từ +0.029 đến +0.040)**.
- C18 chỉ tác động 9 tổ hợp có PD-L1: Δ so với NLP+miss TB +0.016 (lớn nhất ở #2 +0.040, #7 +0.042, #8 +0.025; ở k ≥ 5 chỉ +0.005…+0.008);
  12 tổ hợp không PD-L1 trùng khít NLP+miss (kiểm chứng mã).
- BM1 5 seed tái lập đúng số 20 seed của §2.27 (0.7956 vs 0.7957).

**Thang ablation BM1, 20 seed (cùng phân hoạch):**

| Cấu hình | AUC một mô hình | ensemble 20 |
|---|---|---|
| DyAM gốc | 0.7600 ± 0.019 | 0.7799 |
| uniform_avg | 0.7688 ± 0.014 | 0.7844 |
| + NLP lâm sàng | 0.7841 ± 0.013 | 0.7946 |
| + miss_pattern (§2.22) | 0.7884 ± 0.011 | 0.7977 |
| + C18 | **0.7957 ± 0.011** | **0.8039** |

→ từng bước: +0.009 (bỏ attention) · +0.015 (NLP) · +0.004 (miss) · +0.007 (C18). Ở 20 seed, NLP là thành phần lớn nhất; đóng góp
riêng của miss_pattern ở BM1 nhỏ hơn con số 5 seed (+0.006) và nhỏ hơn nhiều so với ở tổ hợp ít nguồn (#10 +0.023, #13 +0.018).

**Verdict:** positive theo tiêu chí người dùng — mô hình đầy đủ cho AUC cao nhất dự án (BM1 0.796 / ens 0.804) và thắng DyAM ở mọi tổ
hợp k ≥ 5 (8/8) cùng 17/21 tổ hợp nói chung. Chưa có kiểm định mức bệnh nhân cho 21 tổ hợp (chỉ có cho 4 tổ hợp ở §2.27: p = 0.16).
KHÔNG phát biểu được "tốt hơn ở mọi tổ hợp": với chỉ dấu đơn lẻ (TMB, PDL1, Gen, Rad) mô hình gốc ngang hoặc nhỉnh hơn.

### 2.32. ★★★ Đặc trưng mô hình nền từ ẢNH GỐC trên bộ tốt nhất: Phikon (tiêu bản) GIỮ · fmcib (CT) XOÁ (2026-10-05)

**Câu hỏi:** embedding mô hình nền trích từ ảnh gốc (đợt 2026-07, §2.6 — khi đó THAY đặc trưng thủ công và âm) có giúp nếu đưa vào
theo cách khác: (F1) THÊM Phikon làm chuyên gia pathology thứ hai trên bộ tốt nhất; (F2) GHÉP embedding CT vào chuyên gia radiomics?

**Script / đầu ra:** `experiments/archloop/run_fm.py {f1,f1cv,f2}` → `experiments/results/ovo_proof/{f1,f1cv}stack__{slug}.csv` (điểm
từng bệnh nhân); kiểm định ở `experiments/results/ovo_proof/tests_f1cvstack.json`. Mốc so: `c18stack__{slug}.csv` cùng seed (§2.27).
Embedding: `datasets/path_fm_embed_discovery.parquet` (Phikon ViT-B, trung bình ≤ 300 ô 20x trong vùng u HALO, 768 chiều,
**163 bệnh nhân** — GLCM/IHC-A chỉ có 105), `datasets/ct_fmcib_embed_discovery.parquet` (4096 chiều, 187 bệnh nhân).

**Cấu hình:** chuẩn §0 — `train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0,
`cross_modality_enabled=False`, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery (n = 247), l1_filter radiomics chuẩn,
`cnl_nlp` + `miss_pattern` (`no_scale`; miss_pattern tính trên modality gốc, không gồm Phikon). Số chiều PCA = 16, định trước.
- `f1`: PCA khớp trên toàn bộ bệnh nhân có embedding (không dùng nhãn) — chỉ dùng để sàng lọc, 5 seed × 2 tổ hợp.
- `f1cv`: RobustScaler + PCA khớp TRONG từng fold, chỉ trên bệnh nhân train (`ModelFoldPCA`) — bản xác nhận, 20 seed × 4 tổ hợp.
- `f2`: PCA-16 của fmcib nối vào đặc trưng `rad_lesion_pc`, không qua bộ lọc radiomics; 5 seed × 2 tổ hợp.
**Lệch chuẩn:** 4 tổ hợp nhiều nguồn (không phải 21); 20 seed.

**Sàng lọc (5 seed, Δ vs c18stack):** F1 BM1 0.7956 → 0.8104 (+0.0148, 5/5), #19 0.7808 → 0.7969 (+0.0161, 5/5), TB **+0.0154**.
F2 BM1 −0.0085 (2/5), #19 −0.0027 (2/5), TB **−0.0056** → XOÁ (khớp §2.6: mô hình nền trên CT không có tín hiệu, kể cả khi ghép).

**Xác nhận F1 với PCA trong fold (`f1cv`), 20 seed — AUC một mô hình | ensemble 20:**

| Tổ hợp | DyAM | uniform | bộ tốt nhất (c18stack) | **+ Phikon** | Δ vs c18stack (seed thắng) | Δ ens, CI 95%, p boot |
|---|---|---|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.7296 | 0.7442 | 0.7603 \| 0.7681 | **0.7795 ± 0.010 \| 0.7888** | +0.0192 (20/20) | +0.0207 [+0.002, +0.042] p 0.036 |
| Rad+IHC-A+Gen+PDL1 | 0.7411 | 0.7509 | 0.7751 \| 0.7839 | **0.7909 ± 0.011 \| 0.7995** | +0.0158 (19/20) | +0.0156 [−0.005, +0.037] p 0.143 |
| **Rad+IHC-G+Gen+PDL1 (BM1)** | 0.7600 | 0.7688 | 0.7957 \| 0.8039 | **0.8095 ± 0.010 \| 0.8182** | +0.0138 (19/20) | +0.0143 [−0.008, +0.036] p 0.188 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7443 | 0.7656 | 0.7849 \| 0.7919 | **0.7969 ± 0.009 \| 0.8048** | +0.0119 (19/20) | +0.0129 [−0.005, +0.032] p 0.161 |

- **vs bộ tốt nhất:** theo seed +0.0152, 20/20, Wilcoxon p < 1e-5; Δ ensemble gộp **+0.0159 [−0.0032, +0.0360], p = 0.102**; thắng 4/4.
  Theo tiêu chí chấp nhận của protocol: đạt Δ ≥ +0.010 và thắng mọi tổ hợp, KHÔNG đạt p < 0.05 → "xu hướng dương, chưa xác nhận".
- **vs uniform:** theo seed +0.0368, 20/20; Δ ensemble gộp +0.0302 [−0.004, +0.063], p = 0.083.
- **vs DyAM gốc:** theo seed **+0.0504, 20/20**; Δ ensemble gộp **+0.0386 [−0.0003, +0.0764], p = 0.052**; riêng lẻ có ý nghĩa ở
  Rad+IHC-A+MutAmp (p 0.022) và Rad+IHC-A+Gen+TMB+PDL1 (p 0.037).
- PCA trong fold cho kết quả ngang PCA toàn cục (BM1 5 seed 0.8104 vs 20 seed trong-fold 0.8095) → mức tăng không phải do rò rỉ.

> **ĐÍNH CHÍNH 2026-10-05 (xem §2.33):** mức tăng này KHÔNG đến từ nội dung ảnh. ~73% là tín hiệu "bệnh nhân có tiêu bản hay không"
> (một chuyên gia chỉ có bias trên đúng 163 bệnh nhân đó cho +0.0111; ở BM1 0.8087 so với 0.8095 của Phikon). Verdict dưới đây
> phải đọc cùng §2.33.

**Verdict:** positive — mức tăng lớn nhất kể từ NLP và là **CON SỐ CHỐT MỚI CỦA DỰ ÁN: BM1 = 0.8095 ± 0.010 (một mô hình, 20 seed) /
0.8182 (ensemble 20)**, so với DyAM gốc 0.7600 / 0.7799. GIỮ. Khi viết phải nêu: (i) p mức bệnh nhân 0.05–0.10, chưa qua ngưỡng;
(ii) đây là ứng viên thứ ~46 trên cùng 247 bệnh nhân; (iii) chưa tách được phần tăng đến từ ĐỘ PHỦ (163 vs 105 bệnh nhân có
pathology) hay từ chất lượng đặc trưng; (iv) ở tập ngoài, Phikon đơn lẻ kém GLCM (0.697 vs 0.765, §2.6); (v) chưa chạy 21 tổ hợp.
Trái với kết luận §2.6 ("embedding trong fusion luôn kéo tụt"): khi đó Phikon THAY GLCM ở PCA-64; ở đây nó được THÊM ở PCA-16.

### 2.33. ★★★ Phikon: tách độ phủ / tín hiệu "có tiêu bản" / nội dung ảnh, và 21 tổ hợp (2026-10-05) · ĐÍNH CHÍNH §2.32

**Câu hỏi:** mức tăng +0.015 của chuyên gia Phikon (§2.32) đến từ đâu — chất lượng đặc trưng mô hình nền, độ phủ thêm 69 bệnh nhân,
hay chỉ từ việc mô hình biết AI CÓ TIÊU BẢN?

**Script / đầu ra:** `experiments/archloop/run_fm.py {f1cov,f1avail,all21}` → `experiments/results/ovo_proof/{f1cov,f1avail}stack__*.csv`,
`experiments/results/full21_phikon/{slug}.json`; kiểm định `experiments/results/ovo_proof/tests_f1avail.json`.

**Cấu hình:** như §2.32 (chuẩn §0, `train_uniform_avg` qua `train_pdl1fill`, NLP + miss_pattern + C18, PCA-16 khớp trong fold),
20 seed × 4 tổ hợp cho hai đối chứng; 5 seed chuẩn cho 21 tổ hợp.
- `f1cov`: Phikon chỉ được dùng ở bệnh nhân ĐÃ có pathology thủ công (94 người) → không thêm độ phủ.
- `f1avail`: thay embedding bằng MỘT cột hằng trên đúng 163 bệnh nhân có tiêu bản → chuyên gia chỉ còn bias, không có nội dung ảnh.
- `all21`: Phikon chỉ thêm ở 13 tổ hợp có pathology (IHC-A/IHC-G) để không đổi nghĩa tổ hợp; 8 tổ hợp còn lại lấy từ §2.31.

**Kết quả (AUC một mô hình, 20 seed; Δ so với bộ tốt nhất cũ c18stack):**

| Tổ hợp | bộ cũ | Phikon không thêm phủ | chỉ "có tiêu bản" (bias) | Phikon đầy đủ | Phikon − bias |
|---|---|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.7603 | 0.7524 (−0.0080) | 0.7716 (+0.0113) | 0.7795 (+0.0192) | +0.0079 (17/20) |
| Rad+IHC-A+Gen+PDL1 | 0.7751 | 0.7685 (−0.0066) | 0.7866 (+0.0115) | 0.7909 (+0.0158) | +0.0043 (15/20) |
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7957 | 0.7864 (−0.0093) | **0.8087 (+0.0130)** | 0.8095 (+0.0138) | +0.0008 (11/20) |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7849 | 0.7745 (−0.0104) | 0.7937 (+0.0088) | 0.7969 (+0.0119) | +0.0031 (15/20) |
| **TB** | | **−0.0086** (0/20 seed) | **+0.0111** (20/20) | **+0.0152** (20/20) | **+0.0040** (16/20, Wilcoxon p 0.021) |

Ensemble 20 (cũ / bias / Phikon): 0.7681/0.7794/0.7888 · 0.7839/0.7966/0.7995 · 0.8039/0.8155/0.8182 · 0.7919/0.8010/0.8048.
Mức bệnh nhân (bootstrap gộp): bias − cũ +0.0112 [−0.005, +0.028] p 0.18 · Phikon − bias **+0.0047 [−0.017, +0.024] p 0.66** ·
Phikon − Phikon-không-thêm-phủ +0.0257 [+0.009, +0.045] p 0.001 · bias − DyAM +0.0339 [−0.001, +0.071] p 0.057.

- **Trên cùng tập bệnh nhân, Phikon LÀM HẠI** (−0.009, thua 20/20 seed) — khớp §2.6: embedding không hơn đặc trưng thủ công.
- **~73% mức tăng của §2.32 tái lập được bằng một chuyên gia KHÔNG CÓ ẢNH** (bias trên 163 bệnh nhân có tiêu bản): +0.0111 trong +0.0152.
  Ở BM1 gần như toàn bộ (0.8087 vs 0.8095). `miss_pattern` hiện chỉ mã hoá "có đặc trưng pathology thủ công" (105), không phải
  "có tiêu bản" (163) — biến thứ hai mang thêm tín hiệu quy trình chăm sóc.
- **Phần gắn với nội dung ảnh: +0.004** (16/20 seed, p 0.021 theo seed; p 0.66 ở mức bệnh nhân) — không đủ để tuyên bố.

**21 tổ hợp với Phikon (5 seed):** hơn mô hình đầy đủ §2.31 ở 13/13 tổ hợp có pathology (TB +0.019, từ +0.005 đến +0.054);
vs DyAM: k ≥ 5 thắng 8/8 (TB +0.048), k = 2–4 thắng 8/9 (+0.040), k = 1 thắng 1/4; TB 21 tổ hợp DyAM 0.7062 → 0.7438.
BM1 0.8103 ± 0.008 (ens 0.8185); Rad+IHC-G+Gen 0.8021 ± 0.011 (ens 0.8146). Các số này mang cùng lưu ý ở trên.

**Verdict:** con số 0.8095 / 0.8182 là THẬT và tái lập được, nhưng KHÔNG phải là bằng chứng cho đặc trưng mô hình nền: nó chủ yếu là
tín hiệu "có tiêu bản hay không" (quy trình chăm sóc), cùng bản chất với `miss_pattern` và cùng rủi ro không khái quát. Nếu dùng,
phải mô tả là "mở rộng vector sẵn-có thêm biến có-tiêu-bản" chứ không phải "đặc trưng học sâu từ ảnh". Con số chốt KHÔNG phụ thuộc
tín hiệu này vẫn là §2.27: BM1 0.7957 / ens 0.8039. Nội dung ảnh Phikon (trung bình ô, PCA-16): đóng góp +0.004, chưa xác nhận.

### 2.34. Trích lại tiêu bản gốc bằng mô hình nền mới hơn: Phikon-v2 và Midnight (2026-10-05) · KHÔNG vượt đối chứng "có tiêu bản"

**Câu hỏi:** mô hình nền pathology mới hơn Phikon v1 có mang NỘI DUNG ẢNH hữu ích không, tức có vượt được đối chứng `f1avail`
(chuyên gia chỉ biết bệnh nhân có tiêu bản hay không, §2.33)?

**Script / đầu ra:** trích `experiments/foundation_embed/tile_fm_extract.py` (env `doan-env`, torch 2.5.1+cu121, RTX 3060, fp16; 44 phút)
→ `experiments/foundation_embed/cache/tiles/{phikon2,midnight}/{slide}.npy` (đặc trưng TỪNG Ô, float16); gộp
`experiments/foundation_embed/tile_pool.py` → `datasets/path_{phikon2,midnight}_mean_discovery.parquet`; huấn luyện
`experiments/archloop/run_fm.py f1cv --emb=... --tag={g1,g2}` → `experiments/results/ovo_proof/{g1,g2}stack__*.csv`;
kiểm định `experiments/results/ovo_proof/tests_g1_g2.json`.

**Cấu hình trích:** `owkin/phikon-v2` (ViT-L, CLS 1024 chiều) và `kaiko-ai/midnight` (ViT-g, CLS + trung bình patch 3072 chiều) — hai mô
hình mở; UNI/CONCH/Virchow2/H-optimus-0/Hibou/GigaPath bị chặn vì chưa có token Hugging Face. ≤ 500 ô 256 px @level 0 trong vùng u
HALO, lọc nền; 163/163 slide discovery thành công; số ô/slide trung vị 217 (1–500).
**Cấu hình huấn luyện:** chuẩn §0 — `train_uniform_avg` qua `train_pdl1fill`, NLP + miss_pattern + C18, chuyên gia thêm = trung bình ô →
RobustScaler + PCA-16 khớp trong fold; epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold shuffle, 20 seed × 4 tổ hợp nhiều nguồn.
**Lệch chuẩn:** 4 tổ hợp, không phải 21.

**AUC một mô hình | ensemble 20:**

| Tổ hợp | bộ cũ | chỉ "có tiêu bản" | Phikon v1 | G1 Phikon-v2 | G2 Midnight |
|---|---|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.7603 \| 0.7681 | 0.7716 \| 0.7794 | 0.7795 \| 0.7888 | 0.7781 \| 0.7862 | 0.7769 \| 0.7865 |
| Rad+IHC-A+Gen+PDL1 | 0.7751 \| 0.7839 | 0.7866 \| 0.7966 | 0.7909 \| 0.7995 | 0.7884 \| 0.7974 | 0.7869 \| 0.7990 |
| BM1 Rad+IHC-G+Gen+PDL1 | 0.7957 \| 0.8039 | 0.8087 \| 0.8155 | 0.8095 \| 0.8182 | 0.8101 \| 0.8194 | 0.8086 \| 0.8195 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7849 \| 0.7919 | 0.7937 \| 0.8010 | 0.7969 \| 0.8048 | 0.7944 \| 0.8033 | 0.7928 \| 0.8027 |

| So với đối chứng "có tiêu bản" | Δ theo seed | seed thắng | Wilcoxon | Δ ens gộp, CI 95% | p boot |
|---|---|---|---|---|---|
| Phikon v1 (§2.33) | +0.0040 | 16/20 | 0.021 | +0.0047 [−0.017, +0.024] | 0.66 |
| G1 Phikon-v2 | +0.0026 | 14/20 | 0.076 | +0.0035 [−0.012, +0.020] | 0.65 |
| G2 Midnight | +0.0012 | 13/20 | 0.26 | +0.0038 [−0.015, +0.023] | 0.69 |

G1/G2 so với Phikon v1: −0.0014 / −0.0029 (không hơn). So với DyAM (ens gộp): G1 +0.0373 [+0.001, +0.075] p 0.046; G2 +0.0377 p 0.043 —
nhưng đối chứng không-ảnh đã đạt +0.0339, p 0.057, nên ý nghĩa này đến từ tín hiệu sẵn-có chứ không phải từ ảnh.

**Chẩn đoán (163 bệnh nhân có tiêu bản, LR + PCA-16 trong fold, 5 seed × 10-fold; thăm dò, không CI):**

| Embedding | AUC đáp ứng (đơn nguồn) | Spearman(dự đoán TPS, TPS thật) | AUC chỉ TPS | AUC TPS + embedding |
|---|---|---|---|---|
| Phikon v1 | 0.690 | 0.77 | 0.662 | 0.699 (+0.037) |
| Phikon-v2 | 0.679 | 0.82 | 0.662 | 0.676 (+0.014) |
| Midnight | 0.686 | 0.78 | 0.662 | 0.715 (+0.053) |

→ tiêu bản là tiêu bản NHUỘM PD-L1, nên embedding chủ yếu mã hoá chính mức nhuộm PD-L1 (tương quan 0.77–0.82 với TPS) — thông tin
mô hình đã có qua kênh PD-L1. Phần dư ngoài TPS có ở mức đơn giản (+0.01…+0.05 trong LR hai nguồn) nhưng không còn đo được khi đã
có đủ các nguồn khác.

**Verdict:** negative cho nội dung ảnh. Ba mô hình nền (Phikon v1, Phikon-v2, Midnight) với gộp trung bình đều chỉ hơn đối chứng không-ảnh
+0.001…+0.004, p mức bệnh nhân 0.65–0.69. Điều kiện làm G3 (attention MIL: cần ≥ +0.005 so với đối chứng) KHÔNG đạt → không chạy.
GIỮ đặc trưng mức ô đã trích (dùng lại được cho UNI/CONCH-so-sánh hoặc MIL sau này) và hai parquet; không đưa vào mô hình.
Con số chốt không đổi: §2.27 (BM1 0.7957 / ens 0.8039), hoặc 0.8087 / 0.8155 nếu chấp nhận thêm biến "có tiêu bản" (§2.33).

### 2.35. Đợt TỰ THIẾT KẾ: bagged evidence network · nhãn mềm DCB · khuếch đại đồng thuận (2026-10-06) · XOÁ cả bốn

**Câu hỏi:** theo yêu cầu "tự sáng tạo kiến trúc, không bám bài báo", ba thiết kế bắt nguồn từ phát hiện của chính dự án có nâng được
bộ tốt nhất `c18stack` (uniform + NLP + miss_pattern + C18; §2.27) không?

**Script / đầu ra:** `experiments/archloop/run_novel.py` (ĐÃ XOÁ cùng điểm thô `n{1,2a,2b,3}stack__*.csv`); kế hoạch và quy tắc đánh giá ghi
TRƯỚC khi chạy ở `document/2026-10-04_arch-search-loop/ledger.md` (mục "Đợt tự thiết kế"). Mốc: `c18stack__{slug}.csv` cùng seed.

**Cấu hình:** chuẩn §0 — `train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled=False`,
10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery (n = 247), NLP + miss_pattern (`no_scale`) + C18. Sàng lọc: 3 tổ hợp (BM1,
Rad+IHC-A+Gen+TMB+PDL1, Rad+IHC-A+MutAmp) × 5 seed chuẩn, ngưỡng Δ ≥ +0.005. Chế độ tắt tái lập đúng c18stack (BM1 seed 42: 0.8117 = 0.8117).
**Lệch chuẩn:** chỉ sàng lọc, không chạy xác nhận 20 seed vì không ứng viên nào qua; N2b dừng sau 1 tổ hợp vì âm rất lớn.

| # | Thiết kế | BM1 | #19 | MutAmp | Δ TB 1 mô hình | Δ TB ENSEMBLE | Kết luận |
|---|---|---|---|---|---|---|---|
| N1 | Bagged evidence network: 8 thành viên bootstrap trong mỗi fold, trung bình đầu ra | −0.0130 (0/5) | +0.0077 (4/5) | +0.0024 (4/5) | −0.0010 | **−0.0037** | XOÁ — không hơn ensemble theo seed sẵn có |
| N2a | Nhãn mềm: SD có lợi ích bền vững (bor == 3 & PFS ≥ 6 tháng, 29/48) = 0.5 khi huấn luyện | −0.0225 | −0.0160 | −0.0159 (0/15 lượt) | −0.0181 | −0.0163 | XOÁ |
| N2b | Như N2a nhưng = 0 (coi là đáp ứng khi huấn luyện) | −0.0534 (0/5) | — | — | −0.0534 | −0.0476 | XOÁ; dừng sớm |
| N3 | Khuếch đại đồng thuận s = m·(1 + γ·c), γ học được | −0.0029 | +0.0040 | +0.0019 | +0.0010 | +0.0018 | XOÁ — đối chứng "k_only" không cần chạy |

**Đọc kết quả:**
- **N2:** nhãn kiểm tra vẫn là nhãn gốc (29 bệnh nhân DCB là "không đáp ứng"), nên làm mềm nhãn huấn luyện của họ kéo mô hình lệch khỏi điểm cuối đang đo; thua
  tỉ lệ thuận với mức làm mềm (0.5 → −0.018, 0.0 → −0.053). Kết quả âm này KHÔNG chứng minh DCB là điểm cuối tồi — chưa đo AUC trên điểm cuối DCB.
- **N1:** bagging trong fold không tái tạo được lợi ích của ensemble theo seed (khác nhau ở BM1: −0.017). Ensemble theo seed trộn nhiều PHÂN HOẠCH, bagging trong fold chỉ
  lấy lại mẫu trên cùng tập train, nên đa dạng ít hơn và các thành viên có ít dữ liệu hiệu dụng hơn (~63%).
- **N3:** không tác dụng đáng kể (+0.0010); các biến thể hợp nhất vẫn không thêm gì ngoài trung bình tanh.
- Cộng dồn: khoảng 49 ứng viên kiến trúc/huấn luyện đã thử trên cùng 247 bệnh nhân, 0 đạt tiêu chí.

### 2.36. ★ Kiểm chứng NGOÀI bằng rad_valid + path_valid: mã hoá lâm sàng rút gọn và mô hình ba nguồn (2026-10-06)

**Câu hỏi:** hai cohort ngoài có bảng lâm sàng, nhãn `bor` và (path_valid) điểm PD-L1 lâm sàng 71/71 — kiểm chứng ngoài được (A) mã hoá lâm sàng bằng câu văn
và (B) mô hình pathology + PD-L1 + lâm sàng không? Kế hoạch ghi ở `ledger.md` TRƯỚC khi chạy; chạy MỘT lần, không chỉnh sau khi nhìn kết quả ngoài.

**Script / đầu ra:** `experiments/validation/run_external_clinical.py` → `experiments/results/external_clinical.json`. Số liệu bối cảnh phía discovery
(không đăng ký trước, chỉ mô tả) tính bằng script tạm.
**Cấu hình:** huấn luyện trên TOÀN BỘ discovery (n = 247), kiểm tra một lần trên ngoài; siêu tham số chỉ chọn bằng CV 10-fold trong discovery. Cohort ngoài gộp theo hàng
omnibus: 71 (path_valid) + 50 (rad_valid) − 2 hàng trùng = 119, 31 đáp ứng / 88 không. **Lệch chuẩn so với mô hình chính:** chỉ 6/13 biến lâm sàng dựng lại được
(age, pack_years, ecog, albumin, dnlr, hist_adeno); thiếu tumor_burden, therapy_line, brain/liver_mets, combo_therapy, pdl1_therapy (và site_lung ở rad_valid) → câu NLP RÚT GỌN
6 biến, huấn luyện lại trên discovery bằng đúng 6 biến đó. PD-L1 = −clinical_pdl1_score/100 cho cả hai phía (cách C18). B: `uniform_avg` 5 hạt khởi tạo trung bình
(DyAM: `MultiModalDynamicModel`), không `miss_pattern`, không bộ lọc radiomics, GLCM = 150 cột chung.

**A — mã hoá lâm sàng (LR, C chọn bằng CV discovery), AUC ngoài n = 119:**

| Mã hoá | AUC ngoài [CI 95%] | path_valid | rad_valid | CV discovery cùng 6 biến |
|---|---|---|---|---|
| raw6 | 0.673 [0.564, 0.776] | 0.642 | 0.729 | **0.497** (C = 10) |
| NLP6 | 0.554 [0.433, 0.672] | 0.524 | 0.597 | **0.452** (C = 0.01) |
| NLP6 − raw6 | **−0.120** [−0.224, −0.017], p = 0.021 | | | |

**Bối cảnh discovery** (LR, C = 1, 5 seed × 10-fold): 6 biến có ở ngoài **0.496** (≈ ngẫu nhiên); 7 biến KHÔNG có ở ngoài **0.602**; cả 13 biến 0.595.
AUC đơn biến cao nhất thuộc nhóm thiếu (therapy_line 0.594, tumor_burden 0.579); nhóm có ở ngoài ≤ 0.561.

**B — path_valid n = 71 (21 đáp ứng, 50 không đáp ứng), AUC ngoài; cột phải = AUC CV discovery cùng cánh (5 seed):**

| Cánh | n | AUC ngoài [CI 95%] | CV discovery |
|---|---|---|---|
| PD-L1 lâm sàng | 71 | 0.739 [0.622, 0.845] | 0.728 ± 0.013 |
| GLCM (pathology) | 52 | 0.755 [0.598, 0.889] | — |
| NLP6 một mình | 71 | 0.522 [0.366, 0.674] | — |
| GLCM + PD-L1 | 71 | 0.729 [0.604, 0.844] | 0.741 ± 0.004 |
| GLCM + PD-L1 + NLP6 | 71 | 0.721 [0.590, 0.833] | 0.725 ± 0.018 |
| DyAM GLCM + PD-L1 + NLP6 | 71 | 0.708 [0.570, 0.824] | — |

Chênh lệch ngoài (bootstrap ghép cặp 5000): ba nguồn − PD-L1 **−0.018** [−0.115, +0.077] p = 0.71 · ba nguồn − hai nguồn **−0.008** [−0.032, +0.012] p = 0.48 ·
uniform − DyAM (ba nguồn) **+0.013** [−0.008, +0.039] p = 0.25 · GLCM+PD-L1 − PD-L1 **−0.010** [−0.097, +0.078] p = 0.82.

**Đọc kết quả (tách điều đã đo khỏi suy luận):**
- **PD-L1 và pathology khái quát** — đo được: AUC ngoài 0.739 và 0.755, sát nội bộ (0.728; ~0.62–0.77 ở §2.3). Đây là bằng chứng ngoài mạnh nhất cho hai nguồn này.
- **Phép kiểm A KHÔNG kiểm được tuyên bố NLP của bài.** 6 biến còn lại gần như vô tín hiệu ở discovery (CV 0.496), tín hiệu nằm ở các biến bị thiếu; so raw6 với NLP6 ở đây là so hai mã hoá
  của dữ liệu ngang ngẫu nhiên. Kết quả −0.120 (p = 0.021) là có thật nhưng KHÔNG nên đọc là "NLP kém hơn ở bệnh nhân mới": mô hình NLP13 chưa hề được kiểm chứng ngoài.
  Điểm lạ chưa giải thích được: raw6 đạt 0.673 ngoài dù CV discovery 0.497 → có thể hai cohort ngoài khác discovery về cách chọn mẫu (suy luận, chưa kiểm chứng).
- **Hợp nhất không hơn PD-L1 một mình ở n = 71**, nhưng cũng không bị bác bỏ (CI ±0.1). Nội bộ cùng các cánh cũng cho cùng kiểu: 0.728 → 0.741 (+0.013) → 0.725 với NLP6;
  tức mô hình hợp nhất hai–ba nguồn ở đây không hơn nhiều PD-L1 một mình ở CẢ hai phía. Phép kiểm này chỉ có công suất cho chênh lệch ≳ 0.1.
- uniform ≥ DyAM ở ngoài (+0.013, p = 0.25), cùng chiều với nội bộ nhưng không có ý nghĩa.

**Verdict:** hỗn hợp, không đổi kết luận chính. Dương: bằng chứng ngoài đầu tiên cho PD-L1 và pathology trong khung `uniform_avg`. Không kết luận được: mã hoá NLP, lợi ích hợp nhất.
Với khai báo trung thực: mô hình đầy đủ (radiomics + genomics + NLP13 + C18) CHƯA có kiểm chứng ngoài; cohort ngoài thiếu 7/13 biến lâm sàng, toàn bộ genomics và radiomics của path_valid.

### 2.37. Khoảng tin cậy cấp bệnh nhân, điểm cuối phụ (PFS, DCB) và kiểm chứng ngoài genomics trên MSK-IMPACT (2026-10-06)

**Câu hỏi:** (R1) con số chốt có khoảng tin cậy cấp bệnh nhân thế nào; (R2) lợi thế so với DyAM có giữ trên các điểm cuối khác không; (R5) chuyên gia genomics có khái quát
ra cohort công khai không. Quy tắc đánh giá ghi ở `ledger.md` TRƯỚC khi chạy (kể cả: bỏ hiệu chuẩn vì điểm là z-score theo fold, không phải xác suất).

**Script / đầu ra:** `experiments/archloop/analyze_endpoints.py` → `experiments/results/endpoints.json` (R1, R2, trên điểm đã lưu `{dyam,uniform,c18stack}__*.csv`, 20 seed);
`experiments/archloop/analyze_samstein.py` → `experiments/results/samstein_genomics.json` (R5; dữ liệu cBioPortal `tmb_mskcc_2018`, đột biến lấy qua API công khai
vì tệp LFS trên GitHub không còn). **Cấu hình:** không huấn luyện lại ở R1/R2; ensemble = trung bình z-score qua 20 seed; bootstrap 5000 (CI AUC), 1000 (CI C-index) trên bệnh nhân; PFS: E = `pfs_censor`
(1 = sự kiện), C-index = `concordance_index(T, −điểm, E)`, HR theo 1 SD điểm ensemble, log-rank theo trung vị; DCB = `bor` == 3 & `pfs` ≥ 6 tháng (29/48 SD), loại 1 SD bị kiểm duyệt trước 6 tháng
(n = 246, 155 không lợi ích). **Lưu ý:** PFS KHÔNG độc lập với nhãn (nhãn sinh từ BOR; PFS dự đoán nhãn với AUC 0.93).

**R1 — AUC ensemble 20 seed [CI 95% cấp bệnh nhân] | AUC mỗi seed:**

| Tổ hợp | DyAM | uniform | mô hình chốt |
|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | 0.780 [0.711, 0.843] \| 0.760 ± 0.019 | 0.784 [0.718, 0.843] \| 0.769 ± 0.014 | **0.804 [0.740, 0.861]** \| 0.796 ± 0.011 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.765 [0.698, 0.827] \| 0.744 ± 0.016 | 0.780 [0.716, 0.839] \| 0.766 ± 0.015 | 0.792 [0.729, 0.848] \| 0.785 ± 0.009 |
| Rad+IHC-A+Gen+PDL1 | 0.765 [0.698, 0.829] \| 0.741 ± 0.015 | 0.769 [0.704, 0.832] \| 0.751 ± 0.016 | 0.784 [0.719, 0.845] \| 0.775 ± 0.011 |
| Rad+IHC-A+MutAmp | 0.747 [0.677, 0.812] \| 0.730 ± 0.020 | 0.758 [0.690, 0.821] \| 0.744 ± 0.016 | 0.768 [0.701, 0.829] \| 0.760 ± 0.010 |

→ các khoảng tin cậy chồng lên nhau gần hoàn toàn (rộng ≈ 0.12–0.13); khác biệt mô hình nằm trong sai số lấy mẫu bệnh nhân, như đã thấy ở các kiểm định ghép cặp (§2.27, p = 0.16).

**R2 — điểm cuối phụ (điểm ensemble):**

| Tổ hợp | Mô hình | C-index PFS [CI] | HR / 1 SD [CI] | log-rank p | AUC DCB [CI] |
|---|---|---|---|---|---|
| BM1 | DyAM | 0.626 [0.585, 0.665] | 1.49 [1.29, 1.71] | 1e-4 | 0.725 [0.662, 0.787] |
| | uniform | 0.625 [0.585, 0.661] | 1.53 [1.32, 1.77] | 1e-4 | 0.731 [0.668, 0.792] |
| | mô hình chốt | 0.621 [0.580, 0.660] | 1.54 [1.33, 1.79] | < 1e-4 | 0.724 [0.657, 0.784] |
| #19 (7 nguồn) | DyAM / uniform / chốt | 0.609 / 0.624 / 0.619 | 1.47 / 1.61 / 1.56 | ≤ 2e-4 | 0.692 / 0.731 / 0.723 |
| #17 | DyAM / uniform / chốt | 0.622 / 0.615 / 0.613 | 1.55 / 1.52 / 1.51 | ≤ 1e-4 | 0.717 / 0.712 / 0.708 |
| MutAmp | DyAM / uniform / chốt | 0.593 / 0.596 / 0.597 | 1.37 / 1.40 / 1.38 | ≤ 0.010 | 0.662 / 0.678 / 0.685 |

→ điểm của MỌI mô hình đều tiên lượng PFS (HR ≈ 1.4–1.6 / SD) nhưng **lợi thế AUC đáp ứng của mô hình chốt KHÔNG chuyển sang PFS** (chênh C-index so DyAM −0.005, +0.010, −0.009, +0.004) và **hầu như không
chuyển sang DCB** (−0.001, +0.031, −0.009, +0.023; TB +0.011). Phần cải thiện gắn với nhãn đáp ứng nhị phân gốc; không nên nói mô hình "tiên lượng tốt hơn".

**R5 — kiểm chứng ngoài genomics (sống còn), Samstein NSCLC không trùng, n = 307 (194 sự kiện; 288 đơn trị):** huấn luyện LR 7 đặc trưng trên discovery (TMB + EGFR, ERBB2, BRAF, MET, STK11, ARID1A;
AUC CV đáp ứng 0.668, C = 10) → C-index OS **0.539 [0.494, 0.583]** (đơn trị 0.537 [0.490, 0.584]); chỉ TMB 0.529 [0.484, 0.574]. Thang đo khớp (TMB trung vị 7.0 vs 6.85; tỉ lệ đột biến cùng cỡ).
HR / SD của mô hình 1.08 [0.95, 1.23], log-rank p = 0.17. **Giới hạn:** đặc trưng ≠ `driver_binarized` (đột biến "bất kỳ không đồng nghĩa", không có CNA nên mất 4 khuếch đại); điểm cuối là OS chứ không phải đáp ứng;
chưa loại được trùng với hai cohort ngoài (mã khác dạng). → genomics KHÔNG được xác nhận ngoài ở mức có ý nghĩa (C-index ≈ ngẫu nhiên); phù hợp với việc đây là nguồn yếu nhất, không phải bằng chứng nó sai.

**Verdict:** tăng độ trung thực, không tăng con số. Dùng cho bài: khoảng tin cậy cấp bệnh nhân (R1), một câu rõ rằng lợi thế không chuyển sang PFS/DCB (R2), một dòng giới hạn cho genomics (R5).

### 2.38. Bộ mã hoá câu cho 13 biến lâm sàng: mpnet, bge-small, trung bình 4 mẫu câu (2026-10-06) · XOÁ cả ba

**Câu hỏi:** đổi bộ mã hoá câu hoặc dùng nhiều mẫu câu có nâng được mô hình chốt `c18stack` (uniform + NLP + miss_pattern + C18; §2.27) không? NLP là thành phần lớn nhất (+0.015 ở BM1, §2.31).

**Script / đầu ra:** `experiments/archloop/run_enc.py` (ĐÃ XOÁ cùng điểm thô `e{1,2,3}stack__*.csv` và ba parquet `nlp_clinical_embed_e{1,2,3}.parquet`); biến thể và mẫu câu P1–P4 ghi ở
`ledger.md` TRƯỚC khi chạy. Mốc: `c18stack__{slug}.csv` cùng seed. **Cấu hình:** chuẩn §0 (`train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold shuffle trên chỉ số hoán vị theo seed,
cohort discovery n = 247), NLP + miss_pattern `no_scale`; mỗi embedding được L2-chuẩn hoá tường minh (chuẩn hàng = 1.000), parquet riêng, không đụng cache chung. Sàng lọc 3 tổ hợp × 5 seed chuẩn,
ngưỡng Δ ≥ +0.005. **Lệch chuẩn:** không chạy 21 tổ hợp vì không biến thể nào qua sàng lọc.

| Biến thể | BM1 | #19 | MutAmp | Δ TB 1 mô hình | Δ TB ensemble |
|---|---|---|---|---|---|
| e1 `all-mpnet-base-v2` (768d) | +0.0004 | −0.0045 | +0.0001 | −0.0013 | −0.0008 |
| e2 `bge-small-en-v1.5` (384d) | +0.0002 | +0.0000 | +0.0002 | +0.0001 | +0.0001 |
| e3 MiniLM, trung bình 4 mẫu câu | −0.0000 | +0.0003 | +0.0001 | +0.0001 | +0.0002 |

**Đọc kết quả:** cả ba nằm trong ±0.001 của mốc → **lợi ích của việc diễn đạt biến lâm sàng thành câu không phụ thuộc bộ mã hoá hay cách đặt câu** (kể cả mô hình khác họ, bge, và trung bình nhiều câu).
Đây là kết quả bền vững có ích cho bài (không phải "may mắn với MiniLM"), nhưng KHÔNG nâng con số. Khớp với §2.8 (BioClinBERT, mô hình miền y khoa, còn kém hơn có ý nghĩa p = 0.028).
Đếm đa so sánh: thêm 3 ứng viên vào ~52 đã thử trên cùng 247 bệnh nhân.

### 2.39. ★ Bỏ từng nguồn một (leave-one-source-out) trên mô hình chốt, BM1, 20 seed (2026-10-06)

**Câu hỏi:** mỗi nguồn đóng góp bao nhiêu TRONG mô hình chốt (khác với thang cộng dần §2.31, vốn phụ thuộc thứ tự)? Phần đóng góp của pathology là nội dung hay chỉ là tín hiệu "có tiêu bản"?

**Script / đầu ra:** `experiments/archloop/run_loso.py` → `experiments/results/ovo_proof/loso_{rad,path,path_avail,gen,pdl1,pdl1_avail,nlp,miss}__Rad_IHCG_Gen_PDL1.csv`.
Mốc `c18stack__Rad_IHCG_Gen_PDL1.csv` cùng seed. **Cấu hình:** chuẩn §0 (`train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort
discovery n = 247), NLP + miss_pattern + C18; `miss_pattern` ĐÓNG BĂNG từ danh sách nguồn đầy đủ nên các nhánh không `_avail` chỉ bỏ NỘI DUNG; hai nhánh `+bit` bỏ thêm cột `has_*` tương ứng và trừ khỏi
`n_available`; bỏ radiomics dùng `l1 = {}`; `no_scale` tính lại. So sánh ghép cặp CHỈ trên bệnh nhân có điểm ở cả hai lần chạy (hai seed của mốc thiếu ~25 bệnh nhân vì một fold bị bỏ qua khi bộ lọc radiomics trả
0 đặc trưng). **Lệch chuẩn:** chỉ BM1, không phải 21 tổ hợp.

| Nhánh (BM1, mô hình chốt `c18stack` 0.7957 mỗi seed / ensemble 0.8039) | AUC mỗi seed | Δ mỗi seed | Thua mốc | Wilcoxon | Δ ensemble [CI 95% bệnh nhân] |
|---|---|---|---|---|---|
| bỏ radiomics (3 chuyên gia) | 0.7828 | −0.0129 | 18/20 | 0.0002 | −0.0182 [−0.044, +0.008] |
| bỏ pathology, giữ bit có-mặt | 0.7844 | −0.0114 | 17/20 | 0.0002 | −0.0105 [−0.032, +0.013] |
| **bỏ pathology + bit có-mặt** | 0.7627 | **−0.0330** | 19/20 | < 1e-4 | −0.0310 [−0.065, +0.002] |
| bỏ genomics | 0.7672 | **−0.0285** | **20/20** | < 1e-4 | −0.0293 [−0.063, −0.0001] |
| bỏ PD-L1, giữ bit có-mặt | 0.7813 | −0.0144 | 19/20 | < 1e-4 | −0.0139 [−0.038, +0.009] |
| bỏ PD-L1 + bit có-mặt | 0.7828 | −0.0129 | 19/20 | < 1e-4 | −0.0112 [−0.034, +0.011] |
| bỏ NLP lâm sàng | 0.7923 | −0.0034 | 16/20 | 0.027 | −0.0038 [−0.009, +0.001] |
| bỏ miss_pattern | 0.7947 | −0.0010 | 11/20 | 0.47 | +0.0015 [−0.019, +0.022] |

**Đọc kết quả:**
- **Ổn định qua phân hoạch rất rõ, ở mức bệnh nhân chỉ genomics tách khỏi 0** (CI [−0.063, −0.0001]); các nhánh khác có CI chứa 0 do cỡ mẫu (rộng ≈ 0.07).
- **Genomics là nguồn có giá trị lớn nhất trong mô hình** (−0.029, thua 20/20) dù đứng riêng chỉ AUC ≈ 0.67 — nó bổ sung cho các nguồn khác chứ không thừa.
- **Pathology: khoảng 2/3 giá trị là tín hiệu "có tiêu bản".** Bỏ nội dung nhưng giữ bit có-mặt: −0.011; bỏ cả nội dung lẫn bit: −0.033 → bit có-mặt đóng góp ≈ 0.022. Với PD-L1 (phủ 246/247 nhờ C18, nên sẵn-có gần như hằng số)
  bit không đóng góp thêm (−0.0144 vs −0.0129). Khớp §2.33 (chuyên gia chỉ có bias trên 163 bệnh nhân có tiêu bản cho +0.011).
- **NLP chỉ đóng góp −0.0034 khi bị bỏ khỏi mô hình chốt** (16/20 seed, p = 0.027), và **miss_pattern gần như 0** (−0.0010, p = 0.47): thông tin sẵn-có đã được mang SẴN bởi phép trung bình trên các chuyên gia có mặt, nên bit
  riêng thừa khi chuyên gia còn đó. **Điều chỉnh cho thang §2.31:** mức "+0.015 NLP, +0.004 miss" là mức tăng khi thêm VÀO uniform_avg trần; sau khi có đủ các nguồn khác và C18, phần tăng biên còn lại nhỏ hơn nhiều.
  Không được viết "NLP là thành phần quan trọng nhất của mô hình chốt" mà không kèm điều kiện thứ tự này.

**Verdict:** phân tích thuần giải thích, không đổi con số. Dùng cho bài: bảng bỏ-từng-nguồn (đóng góp theo nguồn), và câu nói rõ phần lớn giá trị của pathology là tín hiệu sẵn-có.

### 2.40. Vòng lặp từ tài liệu: mức thổi phồng do chọn mô hình, độ bền của tín hiệu sẵn-có, mốc tabular, TRIPOD+AI (2026-10-06)

**Câu hỏi:** theo các bài đã tìm (Vabalas 2019 về thiên lệch kiểm chứng chéo cỡ mẫu nhỏ; các bài về khuyết dữ liệu mang thông tin; Chowell 2022; TRIPOD+AI / PROBAST+AI): (H1) mức tăng của mô hình chốt có thể chỉ là hệ quả của việc chọn
trong ~52 ứng viên không; (H2) tín hiệu sẵn-có có bền khi cơ chế khuyết đổi không; (H5) mốc tabular kiểu Chowell; (H3) bản thảo đáp ứng TRIPOD+AI tới đâu; (H4) MSK-CHORD. Kế hoạch ở `ledger.md` TRƯỚC khi chạy.

**H1 — mức thổi phồng do chọn** (`experiments/archloop/analyze_selection_bias.py` → `experiments/results/selection_bias.json`; không huấn luyện). SE của hiệu AUC ensemble BM1 (20 seed, bootstrap bệnh nhân 3000) giữa các cặp
đã lưu: bỏ NLP − chốt 0.0024 · bỏ miss − chốt 0.0102 · bỏ genomics − chốt 0.0160 · chỉ-có-tiêu-bản − chốt 0.0090 · Phikon − chốt 0.0112 · Phikon − chỉ-có-tiêu-bản 0.0108 · chốt − uniform 0.0146 · uniform − DyAM 0.0098 · chốt − DyAM 0.0190.
SE đại diện cho một thay đổi nhỏ (trung vị các cặp loso/f1) = **0.0105**. E[max của K = 52 chuẩn độc lập] = 2.26 σ. Mô phỏng 200 000 lần với tương quan ρ giữa các ứng viên:

| ρ | E[max Δ] khi KHÔNG có hiệu ứng thật | P(max ≥ mức quan sát chốt − uniform = +0.0195) |
|---|---|---|
| 0.0 | 0.0237 | 0.81 |
| 0.5 | 0.0168 | 0.37 |
| 0.8 | 0.0107 | 0.18 |

→ mức tăng +0.0195 của mô hình chốt so với trọng số đều (+0.0241 so với DyAM) **nằm trong hoặc ngay sát dải kỳ vọng của riêng việc chọn** (0.011–0.024); không loại trừ được rằng một phần đáng kể là thổi phồng do chọn.
Đây là cận trên thô (ứng viên không độc lập, thành phần như NLP có lý do đưa vào từ trước), không phải hiệu chỉnh chính xác. Khớp Vabalas 2019: CV cỡ mẫu nhỏ thiên lệch, nested CV/tập giữ ngoài mới không.

**H2 — độ bền của tín hiệu sẵn-có** (`experiments/archloop/run_shift.py` → `experiments/results/ovo_proof/shift_{A,B}_p{25,50}__Rad_IHCG_Gen_PDL1.csv`). BM1, 20 seed, cấu hình chuẩn §0; A = mô hình chốt CÓ `miss_pattern`, B = KHÔNG;
ở tập kiểm tra mỗi bệnh nhân có pathology bị đặt "không có" với xác suất p (A cập nhật cả bit has_path và n_available). Kiểm tra mã: p = 0 tái lập đúng c18stack (0.8117) và loso_miss (0.8144) ở seed 42.

| p | A: có miss_pattern | B: không có | A − B | A thắng B | Wilcoxon |
|---|---|---|---|---|---|
| 0 | 0.7957 ± 0.011 | 0.7947 ± 0.012 | +0.0010 | 11/20 | 0.47 |
| 0.25 | 0.7855 ± 0.015 | 0.7870 ± 0.012 | −0.0015 | 9/20 | 0.52 |
| 0.50 | 0.7732 ± 0.016 | 0.7782 ± 0.013 | **−0.0050** | 9/20 | 0.14 |

Mất AUC so với p = 0: A −0.0102 (p = 0.25), **−0.0225** (p = 0.5, giảm ở 20/20 seed); B −0.0077, −0.0165 (19/20). Mức mất thêm của A so với B ở p = 0.5: **−0.0060** (13/20 seed, Wilcoxon p = 0.019).
→ cả hai mất điểm khi thiếu nội dung pathology; mô hình có vector mẫu khuyết mất thêm ~0.006 và KHÔNG có lợi thế nào lúc p = 0 (+0.001, p = 0.47). Kết hợp với §2.39 (bỏ miss_pattern chỉ −0.001): **bằng chứng ủng hộ bỏ `miss_pattern` khỏi mô hình chính** — không mất AUC, bớt phụ thuộc quy trình chăm sóc, bền hơn khi cơ chế khuyết đổi.
Giới hạn: một kiểu nhiễu (ngẫu nhiên đều) trên một tổ hợp (BM1); khuyết thật có tương quan với bệnh nặng nên nhiễu thực tế khác.

**H5 — mốc tabular kiểu Chowell** (script tạm; discovery n = 247, 10-fold × 5 seed): random forest 500 cây trên PD-L1 một mình 0.613 ± 0.020 (RF kém với biến đơn, rời rạc; LR/uniform trên PD-L1: 0.728), lâm sàng 13 biến 0.517, genomics 11 đặc trưng 0.647;
**lâm sàng 13 + genomics 11 + PD-L1: RF 0.727 ± 0.006, hồi quy logistic 0.743 ± 0.010**, so với mô hình chốt 0.796 (thêm radiomics, pathology). → hai nguồn ảnh cộng sẵn-có đưa BM1 lên ≈ +0.05 so với mốc tabular tốt nhất, nhưng §2.39 cho thấy ~2/3 phần pathology là tín hiệu "có tiêu bản".
Khớp §2.15 (XGBoost thua 0.06–0.10) và §2.7 (TabPFN). Không có mô hình tabular nào thắng PD-L1 + genomics đã qua NLP/uniform (PDL1+Gen: 0.7636, §2.31).

**H3 — TRIPOD+AI / PROBAST+AI:** `document/2026-10-06_reporting-checklists/tripod-ai-probast-self-assessment.md`. Danh mục 27 mục lấy từ bảng 2 của bài gốc (Europe PMC PMC11019967); PROBAST+AI đánh giá theo MIỀN vì không đọc được danh sách câu hỏi.
Phát hiện chính: chỗ giữ chỗ chưa điền ở phê duyệt đạo đức (`[INSTITUTION]`, `[IRB-NUMBER]`), tài trợ, địa chỉ kho mã; tuyên bố đồng ý bằng văn bản cần tác giả xác nhận; thiếu khai báo ~52 ứng viên (mục 12c), thiếu hiệu chuẩn (12e), số liệu công suất sai (mục 10, SK-3);
miền "phân tích" tự đánh giá nguy cơ thiên lệch CAO.

**H4 — MSK-CHORD: KHÔNG làm.** Qua API công khai kiểm tra: study `msk_chord_2024` có OS, PD-L1 (NLP), TMB, đột biến/CNA nhưng KHÔNG có nhãn đáp ứng ICI hay mốc bắt đầu ICI ở mức thuộc tính; phơi nhiễm ICI phải dựng lại từ sự kiện dòng thời gian, kèm xử lý
thời gian bất tử (immortal time) và gần như chắc chắn trùng bệnh nhân MSK với cohort hiện có. Đó là một dự án riêng, không phải thí nghiệm trong phiên này. **H6 (cohort đa phương thức độc lập): không làm được** (cần hợp tác).

> **ĐÍNH CHÍNH 2026-10-06 (xem §2.41):** khuyến nghị "bỏ `miss_pattern`" dưới đây chỉ đúng cho các tổ hợp CÓ PD-L1; ở hai tổ hợp k = 5 không có PD-L1, bỏ nó mất 0.017–0.018 (20/20 và 19/20 seed). Quy tắc định trước cho kết quả "GIỮ".

**Verdict:** không có hướng nào nâng con số; hai kết quả đổi quyết định viết bài: (i) bỏ `miss_pattern` khỏi mô hình chính (H2 + §2.39), (ii) khai báo rõ rằng mức tăng so với trọng số đều có thể một phần là do chọn (H1).

### 2.41. ★ Giữ hay bỏ `miss_pattern` trên sáu tổ hợp k ≥ 5 — giá trị của nó phụ thuộc vào việc có PD-L1 (2026-10-06)

**Câu hỏi:** bằng chứng bỏ vector mẫu khuyết (§2.39, §2.40) chỉ ở BM1; nó có đúng ở mọi tổ hợp từ 5 nguồn trở lên không? Quy tắc quyết định ghi ở `ledger.md` TRƯỚC khi chạy: giữ nếu Δ(có − không) trung bình
qua sáu tổ hợp ≥ +0.005, hoặc có tổ hợp Δ ≥ +0.010 với Wilcoxon p < 0.05.

**Script / đầu ra:** `experiments/archloop/run_nomiss.py` → `experiments/results/ovo_proof/nm_{miss,nomiss}__{slug}.csv`; BM1 và các nhánh "có" của ba tổ hợp có PD-L1 lấy từ `loso_miss` và `c18stack`.
**Cấu hình:** chuẩn §0 (`train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery n = 247), uniform + NLP + C18, 20 seed;
so ghép cặp theo AUC từng seed trên bệnh nhân có điểm ở cả hai lần chạy. **Lệch chuẩn:** AUC mỗi seed, không có kiểm định cấp bệnh nhân.

| Tổ hợp | k | AUC CÓ miss_pattern | AUC KHÔNG | Δ (có − không) | Có thắng | Wilcoxon | ensemble có / không |
|---|---|---|---|---|---|---|---|
| Rad+IHC-A+Gen (không PD-L1) | 5 | 0.7647 | 0.7481 | **+0.0166** | 20/20 | < 1e-4 | 0.7760 / 0.7643 |
| Rad+IHC-G+Gen (không PD-L1) | 5 | 0.7828 | 0.7648 | **+0.0180** | 19/20 | < 1e-4 | 0.7928 / 0.7775 |
| Rad+IHC-A+MutAmp | 6 | 0.7603 | 0.7596 | +0.0007 | 10/20 | 0.67 | 0.7681 / 0.7670 |
| Rad+IHC-A+Gen+PDL1 | 6 | 0.7751 | 0.7747 | +0.0003 | 9/20 | 0.87 | 0.7839 / 0.7861 |
| Rad+IHC-G+Gen+PDL1 (BM1) | 6 | 0.7957 | 0.7947 | +0.0010 | 11/20 | 0.47 | 0.8039 / 0.8054 |
| Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7849 | 0.7844 | +0.0005 | 11/20 | 0.62 | 0.7919 / 0.7956 |

**Trung bình Δ qua sáu tổ hợp: +0.0062 → theo quy tắc định trước: GIỮ** (trung bình ≥ +0.005 và hai tổ hợp đạt Δ ≥ +0.010 với p < 0.05).

**Đọc kết quả:**
- Hiệu ứng phân đôi rõ: **chỉ ở hai tổ hợp KHÔNG có PD-L1** vector mẫu khuyết giúp (+0.017, +0.018; thắng 20/20 và 19/20 seed). Ở bốn tổ hợp có PD-L1 (kể cả BM1) nó ≈ 0 (+0.0003 đến +0.0010, p 0.47–0.87).
- Giả thuyết (CHƯA kiểm chứng): khi PD-L1 có ở gần mọi bệnh nhân nhờ C18 (246/247), chuyên gia mạnh nhất luôn hiện diện nên thông tin "ai thiếu nguồn nào" đã được mang theo phép trung bình; thiếu PD-L1 thì vector là thứ duy nhất mang
  tín hiệu sẵn-có của các nguồn còn lại. Khớp §2.14 (lợi ích của miss_pattern tập trung ở tổ hợp không PD-L1).
- Vẫn là tín hiệu QUY TRÌNH CHĂM SÓC, không phải sinh học (§2.14, §2.39, §2.40 H2: khi 50% bệnh nhân bị ẩn pathology, có vector mất thêm 0.006, p = 0.019).

**Verdict:** GIỮ theo quy tắc định trước, nhưng chỉ có giá trị khi tổ hợp thiếu PD-L1. Với mô hình chính BM1 (có PD-L1) bản KHÔNG có vector ngang bản có (0.7947 vs 0.7957) và đơn giản hơn, bền hơn; nên **báo cáo cả hai phiên bản** và nêu rõ điều kiện.

### 2.42. Làm giàu genomics bằng dữ liệu công khai của chính 247 bệnh nhân + ngưỡng PD-L1 lâm sàng (2026-10-06) · XOÁ cả bốn

**Câu hỏi:** thêm THÔNG TIN (họ duy nhất từng thắng) có nâng được mô hình chốt `c18stack` (uniform + NLP + miss_pattern + C18; §2.27) không? Genomics là nguồn quan trọng nhất trong mô hình (§2.39) nhưng chuyên gia chỉ dùng 11 đặc trưng.
Bộ MIND công khai trên cBioPortal (`lung_msk_mind_2020`, cùng 247 bệnh nhân, tải qua API công khai) có đột biến toàn gen, FGA (gánh nặng bất thường số bản sao), điểm MSI.

**Script / đầu ra:** `experiments/archloop/run_genx.py` (ĐÃ XOÁ cùng điểm thô `{gx1,gx2,gx3,pl1}stack__*.csv` và `datasets/genomic_enriched_mind.parquet`; dựng lại bằng API
`https://www.cbioportal.org/api/molecular-profiles/lung_msk_mind_2020_mutations/mutations/fetch` + thuộc tính mẫu FRACTION_GENOME_ALTERED, MSI_SCORE). Kế hoạch định trước ở `ledger.md`; danh sách gen chọn theo tài liệu, KHÔNG theo dữ liệu.
**Cấu hình:** chuẩn §0 (`train_uniform_avg` qua `train_pdl1fill`, epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery n = 247), NLP + miss_pattern; miss_pattern tính từ mask GỐC
(lần đầu để sai thứ tự làm lệch 0.0002 so với mốc, đã sửa; chế độ tắt tái lập c18stack: BM1 seed 42 = 0.8117 = 0.8117). Sàng lọc 5 seed chuẩn, ngưỡng Δ ≥ +0.005. **Lệch chuẩn:** mã hỏng ở tổ hợp MutAmp của ba nhánh genomics
(chỉ mục trùng ở `gen_driver_non_tmb`) nên chỉ có 2 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1); cả hai đều âm rõ nên không sửa.

| # | Ứng viên (định trước) | BM1 | Rad+IHC-A+Gen+TMB+PDL1 | Δ TB 1 mô hình | Kết luận |
|---|---|---|---|---|---|
| GX1 | thêm FGA, MSI_SCORE vào chuyên gia genomics | −0.0152 (0/5) | −0.0094 (0/5) | −0.0123 | XOÁ |
| GX2 | thêm KRAS, TP53, KEAP1, SMARCA4 (đột biến không đồng nghĩa) | −0.0055 | −0.0059 | −0.0057 | XOÁ |
| GX3 | GX1 + GX2 (6 đặc trưng) | −0.0236 | −0.0201 | −0.0219 | XOÁ |
| PL1 | chuyên gia PD-L1 thêm chỉ báo TPS ≥ 1% và ≥ 50% (C18 điền trước) | +0.0024 | +0.0050 | **+0.0049** (3 tổ hợp, 12/15 lượt; +0.0074 ở MutAmp) | sát ngưỡng → xác nhận THĂM DÒ |

**PL1 xác nhận thăm dò (20 seed × 4 tổ hợp; không tính là chiến thắng vì đặt sau khi thấy +0.0049 < ngưỡng +0.005):** Δ mỗi seed +0.0011 (thắng 12/20, Wilcoxon p = 0.37); Δ ensemble gộp **−0.0005** [−0.009, +0.009], p = 0.89, thắng 1/4;
từng tổ hợp +0.0027 / +0.0006 / +0.0016 / −0.0004 → **sàng lọc 5 seed đã phóng đại (+0.0049 → +0.0011)**, thêm một bằng chứng cho §2.22 rằng 5 seed chuẩn lạc quan.

**Đọc kết quả:** thêm đặc trưng genomics — dù có cơ sở sinh học trong tài liệu — làm HẠI, và càng thêm càng hại (2 → −0.012, 4 → −0.006, 6 → −0.022): cùng quy luật "thêm tham số ở chuyên gia thì thua" với 247 bệnh nhân. Thông tin mới phải vừa mới vừa
nhỏ gọn (điểm PD-L1 lâm sàng, C18) mới có lợi. Đếm đa so sánh: thêm 4 ứng viên vào ~52 đã thử trên cùng 247 bệnh nhân.

### 2.43. ★ Kết hợp với hai cohort ICI công khai (Rizvi 2018, Hellmann 2018): kiểm chứng ngoài PD-L1 + genomics và điểm tiên nghiệm (2026-10-07)

**Câu hỏi:** (V1) mô hình PD-L1 + genomics có khái quát ra bệnh nhân ICI ung thư phổi công khai không; (V2) một điểm tiên nghiệm huấn luyện CHỈ trên dữ liệu ngoài có nâng mô hình chốt `c18stack` (§2.27) không.
Kế hoạch và quy tắc ghi ở `ledger.md` TRƯỚC khi chạy.

**Dữ liệu (cBioPortal, API công khai):** `nsclc_pd1_msk_2018` (Rizvi 2018) 240 → loại 23 trùng với bệnh nhân của dự án → loại 9 DCB không đánh giá được → **208** (không lợi ích 147; PD-L1 có ở 63); `nsclc_mskcc_2018` (Hellmann 2018)
**75** (không lợi ích 38; PD-L1 có ở 70; giải trình tự exome nên thang TMB khác panel). Gộp ngoài **n = 283**. Điểm cuối thống nhất = DCB (nhãn 1 = KHÔNG lợi ích bền vững); với MIND: DCB = PR/CR hoặc SD có PFS ≥ 6 tháng.
Đặc trưng (8): PD-L1 (TPS/100), TMB, đột biến "bất kỳ không đồng nghĩa" ở EGFR, ERBB2, BRAF, MET, STK11, ARID1A (KHÔNG có khuếch đại).
**Script / đầu ra:** `experiments/external_prior/{data,run_external,run_prior_stack}.py` → `experiments/results/external_prior.json`, `experiments/results/external_prior_scores.csv`,
`experiments/results/ovo_proof/priorstack__*.csv`, `tests_priorstack.json`. **Cấu hình:** mô hình `uniform_avg` hai chuyên gia (PD-L1; genomics 7 đặc trưng), cấu hình chuẩn §0, 5 hạt khởi tạo trung bình, huấn luyện trên toàn bộ MIND (V1) hoặc toàn bộ dữ liệu ngoài (V2),
kiểm tra MỘT lần; V2 vào bộ chốt (uniform + NLP + miss + C18 + 1 cột tiên nghiệm) với cấu hình chuẩn 20 seed. **Lệch chuẩn:** điểm cuối DCB khác nhãn gốc; cohort ngoài khác thời kỳ, phác đồ, cách chấm PD-L1.

**ĐÍNH CHÍNH thiết kế đánh giá V1:** lần chạy đầu tính cánh "PD-L1" trên cả tập dù chỉ 133/283 bệnh nhân có PD-L1 (điểm hằng ở số còn lại → AUC bị kéo về 0.549, chênh với mô hình hợp nhất bị phóng đại +0.110). Đã sửa bằng cách đánh giá thêm trên tập con CÓ PD-L1
(mọi cánh cùng bệnh nhân); chỉ số sau đây mới đáng tin.

**V1 — AUC ngoài, bệnh nhân CÓ PD-L1 (n = 133, 79 không lợi ích) [CI 95% bootstrap]:**

| Cánh | Gộp | Rizvi (n = 63) | Hellmann (n = 70) |
|---|---|---|---|
| PD-L1 | 0.615 [0.514, 0.711] | 0.660 | 0.578 |
| TMB một mình | 0.649 [0.553, 0.743] | 0.613 | 0.702 |
| genomics (TMB + 6 đột biến) | 0.629 [0.529, 0.724] | 0.627 | 0.638 |
| **PD-L1 + genomics** | **0.689 [0.594, 0.784]** | 0.711 | 0.698 |

Hợp nhất − PD-L1: +0.075 [−0.018, +0.172], p = 0.115 · hợp nhất − genomics: **+0.060 [+0.002, +0.121], p = 0.043** (Rizvi +0.084, Hellmann +0.060, không riêng lẻ có ý nghĩa). Trên TOÀN BỘ 283 (bệnh nhân thiếu PD-L1 dùng genomics):
PD-L1 + genomics 0.659 [0.592, 0.724], genomics 0.637, TMB 0.618.
→ hướng khớp nội bộ (hợp nhất hơn từng nguồn) nhưng khoảng tin cậy rộng; **PD-L1 yếu hơn nhiều ở ngoài (0.615 so với ≈ 0.73 nội bộ)** — chưa phân biệt được do cách chấm / thời kỳ / điểm cuối DCB.

**V2 — điểm tiên nghiệm.** Mô hình PD-L1 + genomics huấn luyện CHỈ trên 283 bệnh nhân ngoài, áp lên 247 MIND: AUC nhãn gốc **0.725**, DCB 0.728 (PD-L1 một mình 0.738). Thêm làm modality cố định vào bộ chốt:
sàng lọc 5 seed Δ TB +0.0084 (BM1 −0.0004, #19 +0.0048, MutAmp +0.0208) → qua ngưỡng +0.005 → xác nhận 20 seed × 4 tổ hợp:

| Tổ hợp | AUC mỗi seed: chốt → + tiên nghiệm | Δ mỗi seed (thắng) | Δ ensemble [CI 95% bệnh nhân], p |
|---|---|---|---|
| Rad+IHC-A+MutAmp (KHÔNG có TMB) | 0.7603 → 0.7811 | **+0.0208 (20/20)** | **+0.0180 [+0.001, +0.036], p = 0.034** |
| Rad+IHC-A+Gen+PDL1 | 0.7751 → 0.7821 | +0.0071 (18/20) | +0.0058 [−0.010, +0.022], p = 0.46 |
| BM1 | 0.7957 → 0.7988 | +0.0031 (13/20) | −0.0003 [−0.014, +0.014], p = 0.97 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7849 → 0.7887 | +0.0037 (16/20) | +0.0035 [−0.011, +0.018], p = 0.64 |

Gộp 4 tổ hợp: Δ ensemble **+0.0068 [−0.007, +0.021], p = 0.34**, thắng 3/4; theo seed +0.0087, 20/20, Wilcoxon p < 1e-5. vs DyAM: Δ ensemble gộp +0.0295 [−0.002, +0.063], p = 0.068, theo seed +0.0439 (20/20); riêng MutAmp +0.039 (p = 0.017).

**Đọc kết quả:** mức tăng TẬP TRUNG ở tổ hợp không có TMB (MutAmp +0.018); ở tổ hợp đã có TMB (BM1, #19) gần như bằng 0 → phần lớn là điểm tiên nghiệm điền lại TMB (và PD-L1 + đột biến huấn luyện trên cỡ mẫu lớn hơn) chứ không phải thông tin mới cho mô hình tốt nhất. Theo tiêu chí chấp nhận của protocol
(Δ ensemble gộp ≥ +0.010, p < 0.05, thắng ≥ 5/6): KHÔNG đạt (+0.0068, p = 0.34) → "xu hướng dương, chưa xác nhận". Con số tiêu đề BM1 không đổi (0.796 → 0.799 mỗi seed, ensemble ≈ không đổi).
Đây là cách kết hợp dữ liệu ngoài KHÔNG rò rỉ (dữ liệu ngoài đã loại 23 bệnh nhân trùng; điểm không dùng nhãn của 247). Đếm đa so sánh: thêm 1 ứng viên vào ~57 đã thử trên cùng 247 bệnh nhân.

**Verdict:** V1 dương nhưng khoảng tin cậy rộng (bằng chứng ngoài đầu tiên cỡ 133 bệnh nhân cho "hợp nhất hơn từng nguồn"); V2 xu hướng dương chưa xác nhận, GIỮ mã, không tuyên bố thắng.

### 2.44. Dùng tiếp dữ liệu ngoài công khai: tiên nghiệm giàu đặc trưng và gộp Hellmann vào huấn luyện (2026-10-07) · XOÁ cả hai

**Câu hỏi:** hai cách chưa khai thác hết hai cohort ICI công khai (§2.43): (H1) một chuyên gia tiên nghiệm giàu đặc trưng hơn, huấn luyện CHỈ trên dữ liệu ngoài (283 bệnh nhân); (H2) gộp Hellmann (nhãn BOR trùng định nghĩa nhãn gốc)
vào tập huấn luyện của mô hình đầy đủ. Kế hoạch và quy tắc ghi ở `ledger.md` TRƯỚC khi chạy.

**Script:** `experiments/external_prior/{run_prior2,run_pool}.py` (ĐÃ XOÁ cùng điểm thô `pool__*.csv`, `external_prior2*`); dữ liệu thô tải từ cBioPortal API như §2.43. **Cấu hình chung:** chuẩn §0 (`train_uniform_avg`, epochs 125, lr 0.01, alpha 0.001,
beta 0.0, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery n = 247).

**H1 — chọn bộ đặc trưng CHỈ bằng CV trong dữ liệu ngoài (10-fold × 5 seed, AUC DCB, n = 283):** S0 (PD-L1 + TMB + 6 đột biến, 8 đặc trưng) **0.6548 ± 0.003** · S1 (+ KRAS, TP53, KEAP1, SMARCA4, 12 đặc trưng) 0.6334 ± 0.007 ·
S2 (+ hút thuốc, giới, tuổi, 15 đặc trưng) 0.6064 ± 0.014 → chọn S0 (= điểm tiên nghiệm của §2.43, không có gì mới để thử). Mô tả, không dùng để chọn: áp lên 247 MIND, AUC nhãn gốc S0 0.725 · S1 0.718 · S2 0.686.
→ **thêm đặc trưng vẫn HẠI ở cỡ mẫu 283**, cùng thứ tự như ở 247: vấn đề không riêng cỡ mẫu 247.

**H2 — gộp Hellmann (68 bệnh nhân có BOR: 44 không đáp ứng, 64 có PD-L1) vào tập HUẤN LUYỆN, bản KHÔNG có `miss_pattern`** (đối chiếu `loso_miss` / `nm_nomiss`, 5 seed; chế độ tắt tái lập 0.8144 = 0.8144 ở BM1 seed 42).
Bệnh nhân ngoài chỉ có PD-L1 và genomics (nguồn ảnh, NLP: NaN, mask 0); genomics ánh xạ theo tên cột (TMB; "driver" → đột biến không đồng nghĩa bất kỳ, xấp xỉ: EGFR 22/22, ERBB2 19/19, MET 10/12, STK11 51/44, ARID1A 21/16, BRAF 14/7 so với cột DRIVER ở MIND; khuếch đại → thiếu).

| Tổ hợp | đối chiếu → gộp Hellmann (mỗi seed) | Δ (thắng) | ensemble: đối chiếu → gộp |
|---|---|---|---|
| BM1 | 0.7923 → 0.7937 | +0.0014 (2/5) | 0.8003 → 0.7985 (−0.0017) |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7768 → 0.7854 | **+0.0085** (5/5) | 0.7861 → 0.7950 (+0.0089) |
| Rad+IHC-A+MutAmp | 0.7545 → 0.7578 | +0.0033 (4/5) | 0.7609 → 0.7674 (+0.0065) |
| **Trung bình** | | **+0.0044** | **+0.0046** |

→ Δ TB +0.0044 < ngưỡng định trước +0.005 → KHÔNG qua, không xác nhận (BM1 gần như không đổi, hướng dương ở hai tổ hợp còn lại; theo bài học §2.42/§2.22, sàng lọc 5 seed thường phóng đại ~0.004).

**Sự cố quy trình:** lần khởi chạy đầu của H2 lỗi ngay (gọi `.astype` trên số nguyên) nhưng ghi nhận là "đang chạy" khi chưa kiểm tra nhật ký, vì tiến trình khởi chạy tách rời không hiện trong danh sách shell nền. Đã sửa, chạy thử một seed, rồi chạy lại theo dõi được. Số liệu trên là từ lần chạy lại.

**Verdict:** negative. Đếm đa so sánh: ~59 ứng viên đã thử trên cùng 247 bệnh nhân.

### 2.45. ★ Kiểm chứng giữ-ngoài theo thứ tự số chụp CT (proxy) và đường cong học theo cỡ mẫu (2026-10-07)

**Câu hỏi:** (T1) mô hình chốt có giữ lợi thế so với DyAM khi huấn luyện trên bệnh nhân "sớm" và kiểm tra trên bệnh nhân "muộn" hơn; (T2) AUC thay đổi thế nào theo cỡ mẫu huấn luyện, tức thêm bệnh nhân có giúp được bao nhiêu.
Kế hoạch ghi ở `ledger.md` TRƯỚC khi chạy (kể cả giới hạn dữ liệu bên dưới).

**Giới hạn dữ liệu — KHÔNG có thời gian lịch:** thuộc tính `DRUG_START` của MIND công khai là TUỔI tính bằng ngày tại lúc bắt đầu thuốc (83 tuổi → 30 582), không phải ngày lịch. Proxy duy nhất = số thứ tự chụp CT (`radiology_accession_number`): liên hệ với thế hệ máy chụp không
thống nhất chiều (Revolution cao hơn p = 0.0014, Discovery thấp hơn p = 0.0044), tỉ lệ không đáp ứng tăng theo ngũ phân vị số chụp (0.62 · 0.67 · 0.84 · 0.84 · 0.78; AUC số chụp → nhãn 0.613). **Không xác minh được số chụp là thời gian**; T1 chỉ là kiểm chứng giữ-ngoài theo thứ tự số chụp, CÓ lệch tỉ lệ nhãn.

**Script / đầu ra:** `experiments/validation/run_order_curve.py {t1, t2 <kind>}` → `experiments/results/order_curve_t1.json`, `order_curve_t2_{stack_nomiss,dyam}.json`. **Cấu hình:** chuẩn §0 (epochs 125, lr 0.01, alpha 0.001, beta 0.0), cohort discovery n = 247, chọn đặc trưng radiomics CHỈ trên bệnh nhân huấn luyện
(loại cả bệnh nhân kiểm tra lẫn tương lai); khi chọn đặc trưng thất bại hoặc trả 0 cột (quá ít bệnh nhân), chuyên gia radiomics đó coi như không có (mask 0) thay vì bỏ cả fold — **lệch so với harness chuẩn** (harness bỏ fold). Bốn cánh T1: DyAM · uniform · chốt CÓ miss_pattern · chốt KHÔNG miss_pattern.

**T1 — forward chaining theo số chụp** (huấn luyện 50% đầu → kiểm tra 17% kế; 67% → 17% kế; 83% → 17% cuối; gộp dự đoán cho 124 bệnh nhân sau cùng: 103 không đáp ứng, 21 đáp ứng; 3 hạt khởi tạo trung bình). AUC gộp [CI 95% bootstrap bệnh nhân]; ba khối:

| Tổ hợp | DyAM | uniform | chốt CÓ miss | chốt KHÔNG miss |
|---|---|---|---|---|
| BM1 | 0.686 [0.566, 0.798] (0.590/0.676/0.741) | 0.681 [0.560, 0.802] | **0.722** [0.601, 0.827] (0.676/0.762/0.734) | 0.704 [0.581, 0.818] (0.638/0.719/0.737) |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.665 [0.538, 0.788] | 0.678 [0.547, 0.796] | **0.716** [0.603, 0.834] | 0.707 [0.581, 0.820] |

→ AUC ngoài-thứ-tự (0.67–0.72) THẤP HƠN nhiều so với CV (0.76–0.80). Thứ hạng giữ nguyên (chốt > uniform ≥ DyAM; BM1 chốt − DyAM +0.018 đến +0.036; #19 +0.042 đến +0.051) nhưng các khoảng tin cậy chồng nhau gần hoàn toàn (rộng ±0.12 với 21 bệnh nhân đáp ứng).
Ghi chú diễn giải: phần giảm không chỉ do cỡ mẫu huấn luyện nhỏ hơn (123–205 bệnh nhân; theo đường cong học T2 ≈ 0.75–0.78 ở cỡ đó) — phần còn lại (~0.05) có thể do lệch tỉ lệ nhãn (nửa sau 83% không đáp ứng) hoặc nhiễu mẫu; không phân biệt được.

**T2 — đường cong học, BM1** (10-fold × 5 seed; lấy ngẫu nhiên phân tầng 40/60/80/100% bệnh nhân huấn luyện trong mỗi fold, chọn đặc trưng chỉ trên phần lấy ra; AUC trên cùng fold kiểm tra):

| Cỡ huấn luyện | mô hình chốt (không miss) | DyAM | chốt − DyAM |
|---|---|---|---|
| 40% (≈ 89) | 0.7265 ± 0.020 | 0.6681 ± 0.011 | +0.058 |
| 60% (≈ 133) | 0.7567 ± 0.015 | 0.7069 ± 0.030 | +0.050 |
| 80% (≈ 178) | 0.7742 ± 0.008 | 0.7319 ± 0.016 | +0.042 |
| 100% (≈ 222) | 0.7931 ± 0.010 | 0.7567 ± 0.017 | +0.036 |

→ **chưa có dấu hiệu phẳng** trong khoảng 89–222: mỗi lần thêm khoảng 45 bệnh nhân tăng AUC +0.017 đến +0.030 (chốt) và +0.025 đến +0.039 (DyAM); mỗi lần nhân đôi cỡ huấn luyện (89 → 178) ≈ +0.048 (chốt), +0.064 (DyAM). Lợi thế của mô hình chốt so với DyAM **thu hẹp** khi cỡ mẫu tăng (0.058 → 0.036): mô hình đơn giản hơn có ích nhất ở dữ liệu nhỏ.
(Chưa xác nhận được xu hướng tiếp tục ngoài 222 — không ngoại suy.) Số lần chuyên gia radiomics bị bỏ vì quá ít bệnh nhân: 34/56/62/62 trên 50 fold, chủ yếu chuyên gia màng phổi (21 bệnh nhân).

**Verdict:** hai phân tích mô tả, không đổi con số. Cho bài: (i) lập luận định lượng rằng nút thắt là cỡ mẫu — AUC còn tăng gần tuyến tính theo cỡ mẫu trong khoảng đã quan sát; (ii) kiểm chứng giữ-ngoài (proxy) cho thấy AUC giảm tới ≈ 0.70 và khoảng tin cậy rộng, nên con số CV 0.796 không nên đọc như hiệu năng trên bệnh nhân mới; (iii) dữ liệu hiện có không có thời gian lịch — nêu trong Limitations.

### 2.46. Đính chính: "có tiêu bản" liên quan đến đáp ứng nhưng KHÔNG giải thích được bằng mức độ nặng đo được (2026-10-07)

Các mục §2.14, §2.33, §2.39–§2.41 và nhiều ghi chú trước gọi tín hiệu mẫu-có-mặt là "quy trình chăm sóc". Kiểm lại trên 247 bệnh nhân discovery (script tạm, phân tích mô tả): có tiêu bản (163; Phikon) tỉ lệ không đáp ứng **0.70**, không có (84) **0.85**, Fisher p = 0.013;
13 biến lâm sàng dự đoán "có tiêu bản" với AUC CV **0.48**; ECOG 0.97/1.04 (p = 0.25), albumin (p = 0.89), tổng khối u (p = 0.72), dòng điều trị (p = 0.33), di căn não (p = 0.97) không khác; sát ngưỡng: dNLR 2.90/3.96 (p = 0.083), di căn gan 0.22/0.33 (p = 0.057), pack-years (p = 0.051).
→ "quy trình chăm sóc" chỉ là GIẢ THUYẾT; cơ chế chưa xác định (chăm sóc không đo được, cách lắp ráp dữ liệu, thời kỳ, hoặc một phần sinh học như khả năng sinh thiết). Thuật ngữ nên dùng trong bài: "mẫu có mặt của các nguồn dữ liệu (cơ chế chưa xác định)". Số liệu và kết luận về hiệu năng không đổi; chỉ đổi cách diễn giải nguyên nhân.

### 2.47. ★ Đặc trưng pathology thủ công chỉ tồn tại ở bệnh nhân PD-L1 DƯƠNG: bit "có pathology" = PD-L1 dương tính (2026-10-07)

**Câu hỏi:** vì sao đặc trưng pathology thủ công (GLCM, IHC-A) chỉ có ở 105 bệnh nhân trong khi 163 bệnh nhân có tiêu bản quét; bài gốc có nói gì không.
**Cách kiểm:** tab chéo `ctx.modality_MASK` (path_ihc_glcm = path_ihc_pdl1 = 105) với điểm PD-L1 chấm lại (`cnl_pdl1_score` lưu −TPS/100) và với danh sách 163 tiêu bản (Phikon); đọc bài gốc (Vanguri 2022, PMC9586871).
**Kết quả (247 bệnh nhân discovery):** điểm chấm lại có ở 201 (dương 110, âm TPS = 0 91, không có điểm 46). **Đặc trưng thủ công: 105/105 người đều là PD-L1 dương; 0/91 người âm; 5/110 người dương không có đặc trưng.** Có trong 163 tiêu bản quét: dương 97, âm 60, không điểm 6.
Tỉ lệ không đáp ứng: dương 0.60, âm 0.90, không điểm 0.80. **Bài gốc:** "n = 201 (81%) có tiêu bản đạt kiểm soát chất lượng (≥ 100 tế bào u)", "105 (52%) dương tính (TPS ≥ 1%)", và mô hình che nguồn khi khối u PD-L1 âm tính. **Cập nhật 2026-10-09:** đã đọc bài gốc (PMC9586871, Results): "A total of 105 (52%) tumor slides showed positive PD-L1 IHC staining (TPS ≥ 1%) and were used to extract IHC texture, a characterization of PD-L1 IHC based on the spatial distribution of expression" — bài xác nhận đặc trưng chỉ tính cho 105 tiêu bản dương, lý do chỉ được ngụ ý (kết cấu mô tả PHÂN BỐ KHÔNG GIAN của biểu hiện PD-L1, nên không có ý nghĩa khi không có biểu hiện); không có câu nêu thẳng cho tiêu bản âm. Quy trình: HALO AI (DenseNet V2) khoanh vùng u, khử chập màu tách kênh PD-L1 khỏi hematoxylin, pyradiomics tính GLCM theo điểm ảnh (mean, s.d., autocorrelation) → "channel 1" = kênh PD-L1. Lệch nhỏ: bài ghi 105 dương, dữ liệu của dự án có 110 người TPS ≥ 1% (chấm lại) nhưng 5 người trong số đó (TPS 20–90%) không có đặc trưng — lý do không rõ (có thể trích đặc trưng thất bại).
**Hệ quả:** (i) bit "có pathology" trong `miss_pattern` ≈ biến PD-L1 dương tính — dấu ấn sinh học, KHÔNG phải quy trình chăm sóc (§2.14 viết sai nguyên nhân); (ii) giải thích kết quả §2.41: vector chỉ giúp ở tổ hợp không có PD-L1 vì khi đó nó mang thông tin dương/âm tính, còn có chuyên gia PD-L1 thì thừa;
(iii) "163 so với 105" không phải mất dữ liệu: 105 = dương tính, còn 58 người thêm của tiêu bản quét phần lớn là PD-L1 âm (60 người âm có tiêu bản); (iv) trong 247 bệnh nhân, chỉ 163 trong 201 người có tiêu bản đạt chất lượng của bài gốc có tệp tiêu bản trên đĩa (lý do không rõ).
**Chưa kiểm chứng:** giả thuyết trước ("chuyên gia PD-L1 luôn bỏ phiếu nên vector thừa") có thể thay bằng giải thích (ii); chưa thử trực tiếp bằng cách thêm biến "PD-L1 dương" riêng vào tổ hợp không có PD-L1.

### 2.48. Cải thiện OvO THUẦN KIẾN TRÚC (không NLP, không sửa dữ liệu): bốn biến thể cổng attention (2026-10-10) · XOÁ cả bốn

**Câu hỏi:** theo yêu cầu "làm OvO tốt hơn DyAM bằng kiến trúc, giữ nguyên dữ liệu", sửa cổng attention của OvO (vốn phẳng, §2.19) có nâng được nó không. Kế hoạch ghi ở `ledger.md` TRƯỚC khi chạy.
**Script:** `experiments/ovo_improve/run_ovo_variants.py` (ĐÃ XOÁ cùng điểm thô `ovx_*.csv`). **Cấu hình:** `train_ovo_safe` (MultiModalDynamicModelOvO), chuẩn §0 (epochs 125, lr 0.01, alpha 0.001, beta 0.0 trừ OV-D, 10-fold shuffle trên chỉ số hoán vị theo seed, cohort discovery n = 247),
dữ liệu GỐC (không NLP, không miss_pattern, không C18), 3 tổ hợp × 5 seed chuẩn, cùng phân hoạch với OvO gốc và DyAM của §2.24. Chế độ tắt tái lập OvO gốc (BM1 seed 42: 0.7872 = 0.7872). Lỗi mã đã sửa trước khi chạy: không được kế thừa trực tiếp `AttentionMatrixOvO` vì `__init__` gốc gọi `super(AttentionMatrixOvO, ...)` theo TÊN toàn cục (đệ quy khi vá tên).

| Biến thể (định trước) | BM1 | Rad+IHC-A+Gen+TMB+PDL1 | MutAmp | Δ vs OvO gốc | Δ vs DyAM |
|---|---|---|---|---|---|
| OvO gốc | 0.7704 | 0.7581 | 0.7403 | — | +0.0129 |
| DyAM | 0.7599 | 0.7413 | 0.7289 | | |
| OV-A: không chia logit attention cho số đặc trưng | 0.7713 | 0.7547 | 0.7357 | **−0.0024** | +0.0105 |
| OV-B: chia cho căn bậc hai số đặc trưng | 0.7700 | 0.7581 | 0.7388 | −0.0006 | +0.0123 |
| OV-C: cổng theo độ tin cậy |tanh r|, γ học được | 0.7688 | 0.7572 | 0.7380 | −0.0016 | +0.0113 |
| OV-D: OV-A + beta = 0.1 | 0.7749 | 0.7498 | 0.7330 | −0.0037 | +0.0092 |

**Đọc kết quả:** không biến thể nào hơn OvO gốc (ngưỡng +0.005; tốt nhất −0.0006). Bỏ phép chia cho số đặc trưng (OV-A, làm cổng bớt phẳng) làm nhẹ kém đi — cùng quy luật "cho cổng nhiều tự do hơn thì giống DyAM hơn và kém hơn".
OvO gốc đã hơn DyAM +0.0129 ở 3 tổ hợp này (5 seed), nhất quán với §2.24 (+0.0064 mỗi seed trên 6 tổ hợp, 20 seed; ở mức bệnh nhân p = 0.57). Không biến thể nào làm khoảng cách đó rộng hơn. Đếm đa so sánh: thêm 4 ứng viên vào ~63 đã thử trên cùng 247 bệnh nhân.

### 2.49. ★ DyAM + NLP so với OvO + NLP (2026-10-10)

**Câu hỏi:** khi cả hai cùng thêm mã hoá NLP của 13 biến lâm sàng (thay `cnl_dem_labs`; không miss_pattern, không C18, PD-L1 gốc), OvO có hơn DyAM không. Kế hoạch ở `ledger.md` trước khi chạy; theo quyết định 2026-10-10 báo hướng và độ lớn kèm khoảng tin cậy, không lấy p-value làm kết luận.
**Script / đầu ra:** `experiments/nlp_verify/run_dyam_nlp.py` → `experiments/results/dyam_nlp_combos_part{0,1}.json` (pha 1, 21 tổ hợp, `run_variant` có `clean_rad_filters`, cùng đường chạy với `ovo_nlp_combos.json`); `experiments/nlp_verify/run_nlp20.py {dyam,ovo}` → `experiments/results/ovo_proof/nlp20_{model}__{slug}.csv` (pha 2, 6 tổ hợp k ≥ 5, 20 seed, điểm từng bệnh nhân).
**Cấu hình:** chuẩn §0 (epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold, cohort discovery n = 247), `cnl_nlp` với `no_scale`. **Lưu ý:** đường chạy dùng `clean_rad_filters` nên số tuyệt đối thấp hơn nhẹ so với `paper_combos.json` (ví dụ DyAM không NLP ở BM1 0.7508 so với 0.7595); chỉ so sánh trong cùng đường chạy.

**Pha 1 — 21 tổ hợp, 5 seed:** AUC CAO NHẤT trong 21 tổ hợp: DyAM 0.7508 · OvO 0.7670 · **DyAM+NLP 0.7732 · OvO+NLP 0.7822**; một lần chạy cao nhất: DyAM+NLP 0.8038 · **OvO+NLP 0.8133** (tổ hợp bảy nguồn). OvO+NLP cao hơn DyAM+NLP ở 11/21 tổ hợp (ở 5/6 tổ hợp phân biệt từ 5 nguồn trở lên; #20, #21 trùng #17, #18), trung bình 21 tổ hợp 0.7157 so với 0.7123; ở tổ hợp một nguồn hai mô hình trùng nhau (NLP làm giảm nhẹ cả hai).

**Pha 2 — 20 seed, AUC trung bình mỗi seed | ensemble 20 seed [CI 95% cấp bệnh nhân của Δ ensemble]:**

| Tổ hợp | DyAM+NLP | OvO+NLP | Δ mỗi seed (OvO thắng) | ens DyAM+NLP → OvO+NLP | Δ ens |
|---|---|---|---|---|---|
| Rad+IHC-A+Gen | 0.7429 | 0.7375 | −0.0054 (6/20) | 0.7645 → 0.7534 | −0.0112 [−0.027, +0.002] |
| Rad+IHC-G+Gen | 0.7653 | 0.7637 | −0.0016 (9/20) | 0.7769 → 0.7750 | −0.0019 [−0.010, +0.006] |
| Rad+IHC-A+MutAmp | 0.7326 | 0.7419 | +0.0093 (15/20) | 0.7510 → 0.7536 | +0.0026 [−0.012, +0.017] |
| Rad+IHC-A+Gen+PDL1 | 0.7446 | 0.7514 | +0.0068 (14/20) | 0.7651 → 0.7653 | +0.0002 [−0.013, +0.014] |
| BM1 | 0.7706 | **0.7800** | +0.0094 (15/20) | 0.7866 → 0.7912 | +0.0046 [−0.005, +0.014] |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.7367 | 0.7641 | **+0.0275 (19/20)** | 0.7527 → 0.7763 | +0.0235 [+0.001, +0.049] |
| **Trung bình** | **0.7488** | **0.7564** | **+0.0077** (16/20 seed) | | |

OvO+NLP cao hơn DyAM+NLP ở 4/6 tổ hợp (cả theo trung bình lẫn ensemble); cao nhất trong 6 tổ hợp: trung bình 20 seed 0.7800 so với 0.7706, ensemble 0.7912 so với 0.7866. Hai tổ hợp DyAM+NLP nhỉnh hơn là Rad+IHC-A+Gen và Rad+IHC-G+Gen (cùng hai tổ hợp không có PD-L1).
**Đọc kết quả:** chiều nhất quán với không NLP (OvO nhỉnh hơn DyAM, rõ nhất ở bảy nguồn), độ lớn trung bình ≈ +0.008; khoảng tin cậy cấp bệnh nhân rộng (chỉ tổ hợp bảy nguồn có khoảng không chứa 0). NLP nâng cả hai mô hình.

### 2.50. MC-OvO — bù chuyên gia thiếu kiểu MCMoE rút gọn trên OvO (2026-10-10)

**Câu hỏi:** bộ sinh có cổng bù chuyên gia thiếu (arXiv 2511.17397, rút gọn) có làm OvO thuần kiến trúc tốt hơn không. Kế hoạch ở `ledger.md` trước khi chạy.
**Script:** `experiments/mcmoe/run_mcmoe.py screen` (đã xoá), điểm thô `ovo_proof/mcovo__*.csv` (đã xoá). **Cấu hình:** chuẩn §0, OvO (`MultiModalDynamicModelOvO`), không NLP, không sửa dữ liệu; bộ sinh g_i từ r_j của các chuyên gia còn lại (detach), chuyên gia thiếu dùng g_i với điểm attention × 0.5, mất mát tái dựng MSE trọng số 0.5 (qua beta); siêu tham số cố định, không dò. Sàng lọc BM1, bảy nguồn, Rad+IHC-G+Gen × 5 seed, đối chiếu OvO gốc cùng seed.
**Lần chạy 1 lỗi cài đặt** (gradient tái dựng chảy vào chuyên gia: Δ −0.123), đã sửa bằng detach đầu vào bộ sinh và chạy lại; chỉ lần 2 tính.
**Kết quả:** MC-OvO − OvO = −0.0065 (BM1 −0.0065, bảy nguồn −0.0068, Rad+IHC-G+Gen −0.0062; thắng 2/5 seed mỗi tổ hợp); ngưỡng +0.005.
**Verdict:** DELETE — âm; bù chuyên gia thiếu không giúp, hợp với việc điền PD-L1 (C18) chỉ cải thiện nhỏ và cơ chế mask đã đủ.

### 2.51. OvO cổng Laplace kiểu FuseMoE (2026-10-12)

**Câu hỏi:** thay điểm attention tuyến tính (chia n_i nên cổng phẳng) bằng khoảng cách Laplace −‖x_i − c_i‖/√n_i (arXiv 2402.03226) có làm OvO tốt hơn không. Kế hoạch ở `ledger.md` trước khi chạy.
**Script:** `experiments/lapoo/run_lap.py` (đã xoá), điểm thô `ovo_proof/lapovo__*.csv` (đã xoá). **Cấu hình:** chuẩn §0, `MultiModalDynamicModelOvO` với lớp attention thay thế, không NLP, không sửa dữ liệu, tâm c_i học được (khởi tạo 0), không siêu tham số dò. Sàng lọc BM1, bảy nguồn, Rad+IHC-G+Gen × 5 seed, đối chiếu OvO gốc cùng seed.
**Kết quả:** Δ TB +0.0024 (BM1 −0.0010 · bảy nguồn **+0.0095 (4/5 seed)** · Rad+IHC-G+Gen −0.0014); ngưỡng +0.005.
**Verdict:** DELETE — chưa đạt ngưỡng; tín hiệu dương chỉ ở bảy nguồn (một tổ hợp, 5 seed, hậu nghiệm khi nhìn riêng, không tính là xác nhận).

### 2.52. DyAM và OvO có Labs ở #20, #21 — hoàn chỉnh bảng 21 tổ hợp một đường chạy (2026-10-12)

**Câu hỏi:** bảng 21 tổ hợp của bài (sau quyết định SK-2 B và Q1 B) cần DyAM, OvO, DyAM+NLP, OvO+NLP cùng một đường chạy (`run_variant` có `clean_rad_filters`); riêng #20, #21 (cấu hình CÓ Labs) chưa có cột DyAM/OvO không-NLP trong đường chạy này (cột `base` của §2.49 bỏ Labs nên trùng #17, #18).
**Script / đầu ra:** `experiments/nlp_verify/run_labs_base.py` → `experiments/results/dyam_ovo_labs21.json`. **Cấu hình:** chuẩn §0 (epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold, 5 seed + một lần chạy tự nhiên, cohort discovery n = 247, `cross_modality_enabled` False), không NLP, Labs thô là nguồn lâm sàng; DyAM (`train`) và OvO (`train_ovo_safe`).
**Kết quả (trung bình 5 seed):** #20 DyAM 0.7248 · OvO 0.7400; #21 (BM2) DyAM 0.7484 · OvO 0.7640. So với +NLP cùng đường chạy: #20 0.739 / 0.757, #21 0.773 / 0.782 (NLP thay Labs cao hơn ở cả bốn cặp, +0.014 đến +0.025).
**Verdict:** KEEP làm số liệu cho Bảng 4 mới (cùng đường chạy). Bảng tiếp theo do `paper/make_results_tables.py` sinh từ `dyam_nlp_combos_part*.json`, `ovo_nlp_combos.json`, `dyam_ovo_labs21.json`, `verify_nlp_5seed.json` và `ovo_proof/nlp20_*.csv`.

### 2.53. Bỏ từng nguồn một trên OvO+NLP, BM1 (2026-10-12)

**Câu hỏi:** đóng góp của từng nguồn trong mô hình chính của bài (OvO + NLP). Kế hoạch ở `ledger.md` (W1) trước khi chạy.
**Script / đầu ra:** `experiments/ovo_final/run_loso_ovo.py` → `experiments/results/ovo_proof/loso_ovo_{rad,path,gen,pdl1,nlp}.csv`; đối chiếu `nlp20_ovo__Rad_IHCG_Gen_PDL1.csv` cùng seed. **Cấu hình:** chuẩn §0 (epochs 125, lr 0.01, alpha 0.001, beta 0.0, 10-fold, 20 seed, discovery n = 247), `train_ovo_safe`, `cnl_nlp` no_scale, đường chạy `build_l1` (cùng `nlp20_*`), không miss_pattern, không C18, PD-L1 gốc; bỏ nguồn thì `l1`/`no_scale` tính lại theo danh sách còn lại.
**Kết quả (AUC trung bình 20 seed; mô hình đầy đủ 0.7800):** bỏ pathology IHC-G 0.7475 (Δ −0.0325, 20/20 seed giảm) · bỏ radiomics 0.7588 (−0.0212, 19/20) · bỏ PD-L1 0.7637 (−0.0163, 19/20) · bỏ NLP 0.7646 (−0.0154, 19/20) · bỏ genomics 0.7656 (−0.0145, 18/20).
**Verdict:** KEEP làm phân tích bổ sung (phụ lục): mọi nguồn đều đóng góp; pathology lớn nhất (một phần là bit có-mặt, xem project_pathology_features_pdl1_positive); đóng góp của NLP (+0.015) cùng cỡ genomics.

### 2.54. OvO+NLP với và không có vector sẵn-có (miss_pattern), 6 tổ hợp k ≥ 5 (2026-10-12)

**Câu hỏi:** quyết định Q2 A: vector sẵn-có có giúp mô hình chính (OvO+NLP) không, và ở đâu. Kế hoạch ở `ledger.md` (W2).
**Script / đầu ra:** `experiments/ovo_final/run_miss_ovo.py` → `ovo_proof/nlp20miss_ovo__*.csv`; đối chiếu `nlp20_ovo__*.csv` cùng seed. **Cấu hình:** chuẩn §0, 20 seed, `train_ovo_safe`, `cnl_nlp` và `miss_pattern` đều no_scale, miss_pattern tính từ mask gốc (trước mọi chỉnh sửa), không C18, `build_l1`.
**Kết quả (AUC trung bình 20 seed, OvO+NLP → +vector; Δ; số seed có vector hơn):** Rad+IHC-A+Gen 0.7375 → 0.7550 (+0.0175, 20/20) · Rad+IHC-G+Gen 0.7637 → 0.7803 (+0.0165, 19/20) · Rad+IHC-A+MutAmp 0.7419 → 0.7466 (+0.0047, 14/20) · Rad+IHC-A+Gen+PDL1 0.7514 → 0.7600 (+0.0086, 19/20) · BM1 0.7800 → 0.7833 (+0.0033, 12/20) · bảy nguồn 0.7641 → 0.7678 (+0.0036, 15/20).
**Verdict:** KEEP làm phân tích bổ sung (phụ lục), KHÔNG vào mô hình chính; lợi ích lớn ở hai tổ hợp không có PD-L1 (+0.017), nhỏ ở các tổ hợp có PD-L1; hợp với §2.41 trên nền trọng số đều.

### 2.55. Giữ-ngoài theo số chụp (T1) và đường cong học (T2) cho DyAM+NLP và OvO+NLP (2026-10-12)

**Câu hỏi:** lặp T1/T2 của §2.45 trên hai mô hình của bài. Kế hoạch ở `ledger.md` (W3). **Script / đầu ra:** `experiments/ovo_final/run_order_ovo.py {t1,t2}` → `results/order_ovo_t1.json`, `order_ovo_t2.json`.
**Cấu hình:** chuẩn §0, `MultiModalDynamicModel` / `MultiModalDynamicModelOvO`, `cnl_nlp` no_scale, `build_l1`, không miss, không C18; T1 3 hạt khởi tạo trung bình, huấn luyện 50% / 67% / 83% đầu theo số chụp, kiểm tra 124 bệnh nhân cuối (21 đáp ứng); T2 10-fold × 5 seed, lấy ngẫu nhiên phân tầng 40/60/80/100%. **Lệch chuẩn:** chạy trên môi trường GPU `doan-env` (torch 2.5.1+cu121, scikit-learn 1.1.1, scipy 1.8.1, numpy 1.22.4) vì CPU đang bận; chỉ so sánh nội bộ hai mô hình, không so trực tiếp với các bảng chạy ở môi trường chính (DyAM+NLP 100% ở T2 là 0.780 so với 0.773 ở bảng chính).
**T1 (AUC gộp, [CI 95% bootstrap bệnh nhân]):** BM1 DyAM+NLP 0.708 [0.586, 0.824] · OvO+NLP 0.743 [0.634, 0.849]; bảy nguồn 0.684 [0.546, 0.812] · 0.715 [0.587, 0.829].
**T2 (BM1, trung bình ± sd 5 seed):** 89 bệnh nhân DyAM+NLP 0.679 ± 0.025, OvO+NLP 0.709 ± 0.023; 133: 0.717 ± 0.012 / 0.735 ± 0.013; 178: 0.749 ± 0.009 / 0.760 ± 0.007; 222: 0.780 ± 0.018 / 0.782 ± 0.006. Khoảng cách 0.030 → 0.018 → 0.011 → 0.002; chưa phẳng.
**Verdict:** KEEP (bài + Discussion). OvO+NLP cao hơn ở cả hai phép kiểm, khoảng tin cậy chồng nhau; lợi thế mô hình nhỏ thu hẹp khi cỡ mẫu tăng.

### 2.56. tdAUC 5 seed cho 11 biến thể lâm sàng (SK-6 rút gọn) (2026-10-12)

**Câu hỏi:** SK-6 (chọn "nâng numpy ≥ 2 trong env riêng rồi kiểm"): tdAUC (6/12/18 tháng) có hỗ trợ tuyên bố cũ "NLP cải thiện tdAUC ở 12–18 tháng". IBS KHÔNG tính: mã IBS của bản gốc không còn trong kho nên không tái tạo được định nghĩa.
**Script / đầu ra:** `experiments/survival/run_survival_scores.py` (env chính; lưu điểm ngoài-fold `results/survival_scores/*.csv`) → `experiments/survival/compute_tdauc.py` (venv `C:/Users/Admin/venv_np2`: numpy 2.5.4, scikit-survival 0.28.0) → `results/tdauc_5seed.json`. **Cấu hình:** DyAM (`train`), chuẩn §0, 5 seed × 10-fold, cohort discovery n = 247, đúng 11 biến thể của `verify_nlp_5seed`; tdAUC tích luỹ/động (Uno), PFS (tháng), `pfs_censor` = 1 là sự kiện.
**Kết quả (mean 5 seed; 6 / 12 / 18 tháng):** IHC-A no-clinical 0.685 / 0.692 / 0.660 · labs 0.681 / 0.685 / 0.652 · NLP raw 0.678 / 0.689 / 0.665 · NLP-PCA16 0.678 / 0.688 / 0.666 · BioClinBERT 0.672 / 0.687 / 0.657; IHC-G no-clinical 0.692 / 0.683 / 0.647 · labs 0.695 / 0.684 / 0.652 · NLP raw 0.697 / 0.691 / 0.660 · NLP-PCA16 0.694 / 0.688 / 0.663. Chênh giữa các biến thể trong một nhánh ≲ 0.02, sd 0.01–0.045.
**Verdict:** KEEP (đưa vào Bảng 8 của bài). Tuyên bố cũ "+0.024 ở 12 tháng, +0.020 ở 18 tháng cho NLP-PCA16" KHÔNG tái lập (IHC-A: NLP-PCA16 0.688 so với không lâm sàng 0.692 ở 12 tháng).

## 3. Bảng tổng hợp số liệu chốt cho paper (Master Numbers)

Nguồn: `paper/drafts/master_numbers.md` (nguồn duy nhất cho số liệu paper).

### Cohort
| Cohort | n | Event rate | Median PFS | PFS range |
|---|---|---|---|---|
| Discovery | 247 | 84.6% (209/247) | 2.7 tháng | 0.1–49.1 tháng |
| Radiology validation | 50 | — | — | — |
| Pathology validation | 71 | — | — | — |

### Best overall
- **OvO Rad+IHC-G+Gen+PDL1 (single phân hoạch cố định)** = AUC **0.8003** [0.739–0.862] — con số gốc, **không phải** trung bình 5-seed robust.
- **5-seed robust, đáng tin cậy nhất cho paper:** `uniform_avg` BM1 = 0.7746±0.019 (ens 0.7850); **+NLP BM1** = 0.7832±0.016 (uniform_avg, 5-seed) / 0.7822±0.017 (OvO, 5-seed, §2.9) — hai backbone cho kết quả tương đương, cao nhất trong mọi biến thể robust.
- **Single-run cao nhất toàn project (không phải số robust, chỉ để đối chiếu bài báo gốc):** 0.8133 —
  OvO+NLP, combo #19 Rad+IHC-A+Gen+TMB+PDL1 (§2.9); trước đó là 0.8105/0.8091 (uniform_avg+NLP, §2.8).
- **External validation mạnh nhất:** IHC-G (GLCM) pathology, AUC 0.767 [0.62–0.89], C-index 0.618 — generalization gap gần 0.

---

## 4. Câu chuyện tổng kết (dùng cho paper, 3 đóng góp)

Sau 7 hướng thử nghiệm (attention DyAM/OvO, stacked fusion, feature engineering, foundation embedding ảnh gốc,
TabPFN — tất cả ÂM hoặc NGANG baseline), chỉ có **NLP-clinical** cho cải thiện dương bền vững:

1. **NLP-clinical embedding** — mã hoá 13 biến lâm sàng thô bằng câu chữ tiếng Anh → MiniLM 384d, dương nhất
   quán trong fusion đa nguồn (§2.8), và là independent prognostic factor mạnh nhất trong Cox (HR 6.07, §2.1).
2. **Generalization theo modality** — pathology GLCM transfer tốt (external AUC 0.767, C-index 0.618, gap ~0),
   radiomics KHÔNG generalize (external ≤0.46, dưới ngẫu nhiên) (§2.3).
3. **Ablation trung thực** — model/feature phức tạp (attention học, stacked fusion, foundation embedding,
   TabPFN) không vượt baseline đơn giản (`uniform_avg`/LR); trần tín hiệu nằm ở dữ liệu (n=247), không phải
   kiến trúc (§2.2, 2.4, 2.5, 2.6, 2.7).

Backbone báo cáo khuyến nghị: `uniform_avg` + seed-ensemble (nội bộ), single-run chỉ để đối chiếu kiểu bài báo gốc.

---

## 5. Quy tắc cập nhật file này (BẮT BUỘC)

> **Từ nay, sau mỗi lần chạy training/thực nghiệm xong, PHẢI thêm một mục mới vào file này** (không ghi đè
> mục cũ), gồm tối thiểu:
> 1. Tên thực nghiệm/câu hỏi đang trả lời + ngày chạy.
> 2. Script/notebook đã chạy + đường dẫn kết quả gốc (`experiments/results/*.json`, `excel/*.xlsx`, ...).
> 3. **Cấu hình training đầy đủ**: epochs, lr, alpha, beta, folds, seeds, `cross_modality_enabled`,
>    `attention_gate_enabled`, model type (`train`/`train_ovo`/`train_gmu`/`train_uniform_avg`/`train_LR`/...),
>    modality combo, cohort/dataset dùng, l1_filter nếu có — nếu khác cấu hình chuẩn ở mục 0, phải ghi rõ khác gì.
> 4. Kết quả số: AUC (+CI nếu có), hoặc C-index/HR/tdAUC nếu là survival, kèm mean±sd nếu multi-seed.
> 5. Một câu kết luận ngắn (GIỮ/XOÁ, dương/âm/ngang baseline).
>
> Việc này áp dụng cho **mọi lần train** kể cả kết quả âm — kết quả âm cũng phải ghi lại (đã là văn hoá của
> dự án, xem các mục "XOÁ" ở trên) để tránh chạy lại thí nghiệm đã biết thất bại.

---

## MMIST-LUNG pilot — OvO vs DyAM, 12-month mortality (2026-10-12)

**Question:** does OvO's advantage generalize to another public multimodal lung cohort (MMIST-LUNG = TCGA+CPTAC, 1,365 pts, NOT an ICI cohort)?
**Script:** `experiments/mmist/run_pilot.py` · **Raw:** `experiments/results/mmist_pilot.json` · data cloned from github.com/ritacmendes/MMIST-LUNG (transcriptomics via git-lfs).
**Config (deviations from §0 marked):** epochs 125, lr 0.01, alpha 0.001, beta 0, `cross_modality_enabled=False` (OvO drops it), `attention_gate_enabled` default, `train`/`train_ovo`, 10-fold, **1 seed (42) — deviates from 5-seed standard**, no l1_filter. **Only 3 modalities → 7 combos, not the 21 paper combos** (CT 71 / PET 33 pts unusable; WSI h5 tile features not used in this pilot). Modalities: clin_demo (7 cols; days_to_death / days_to_last_fup / vital_status_* dropped as leakage), clin_diag (8 TNM/stage cols), tx (top-200 variance genes of 10,270; unsupervised filter). Median imputation for tabular NaN. Label = death within 12 months (events 196/1365, 14%).

| Combo | DyAM | OvO |
|---|---|---|
| clin_demo | 0.546 | 0.546 |
| clin_diag | 0.608 | 0.608 |
| tx | 0.555 | 0.555 |
| demo+diag | 0.617 | 0.587 |
| demo+tx | 0.568 | 0.575 |
| diag+tx | 0.604 | 0.604 |
| demo+diag+tx | 0.615 | 0.591 |

**Verdict: NEGATIVE / wash.** OvO ≤ DyAM on both 3-modality-ish combos (−0.03 / −0.025), ≈ equal elsewhere; absolute AUC is low (~0.55–0.62) so this is near-noise on a 14%-event outcome. Single-seed, so differences of ~0.03 are not conclusive. Do NOT add to the paper as supporting evidence. Open oddity: `diag+tx` gives identical AUC for both models (0.6044) — attention likely degenerate; not investigated.

---

## Institut Curie NSCLC 1L-ICI cohort (Captier 2025) — OvO vs DyAM, 7 combos (2026-10-12)

**Question:** does OvO vs DyAM hold on a second ICI cohort (n=294 labelled; Zenodo 10.5281/zenodo.14293431, CC-BY-NC-4.0)? Second cohort, NOT external validation (PET not CT, RNA not mutation, no PD-L1; 196 chemo-IO / 121 mono).
**Script:** `experiments/mmist/run_curie.py` · **Raw:** `experiments/results/curie_ici_pilot.json`.
**Config:** epochs 125, lr 0.01, alpha 0.001, beta 0, `cross_modality_enabled=False`, `train`/`train_ovo`, 10-fold, **5 seeds (42,7,123,2024,31337)**, no l1_filter. **Only 3 modalities → 7 combos (not 21)**: rad (PET, 30 feats, 190 pts), path (134 feats, 219), tx (33 + one-hot biopsy site = 48, 121). Label: best RECIST PR/CR=0 vs SD/POD=1 (149/145); 23 w/o RECIST dropped. Median imputation of residual NaN (unsupervised, over all pts). `chemotherapy` NOT used as covariate.

| Combo | n | DyAM | OvO | OvO>DyAM seeds |
|---|---|---|---|---|
| rad | 190 | 0.492±0.016 | 0.489±0.016 | 1/5 |
| path | 219 | 0.535±0.020 | 0.535±0.020 | 0/5 |
| tx | 121 | 0.613±0.034 | 0.613±0.034 | 0/5 |
| rad+path | 254 | 0.556±0.027 | 0.549±0.034 | 2/5 |
| rad+tx | 225 | 0.553±0.018 | 0.542±0.024 | 1/5 |
| path+tx | 240 | 0.537±0.017 | 0.534±0.024 | 2/5 |
| rad+path+tx | 263 | 0.531±0.016 | 0.539±0.012 | 4/5 |

**Verdict: NEGATIVE / wash.** All AUC ≈ 0.49–0.61 (near chance; only tx > 0.6). Differences OvO−DyAM within ±0.011 except 3-way (+0.008, within noise). Single-modality rows are identical by construction (attention trivial). No evidence OvO helps here; do not cite as support. Cohort too small/heterogeneous (mono+chemo-IO) for the endpoint; Captier's own binary-response AUCs were also weak (≤0.67 for 6-mo progression).

### 2.57. Tái tạo ~0.8 theo cấu hình Figures-Finalized.ipynb + OvO/NLP, 21 tổ hợp (2026-10-12)

**Mã:** `experiments/nlp_verify/run_notebook_repro.py` (3 phần song song) · **Kết quả:** `experiments/results/notebook_repro_part{0,1,2}.json` (+ `.log`).
**Config (giống cell 13/18 notebook):** epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, KFold 10 shuffle `random_state=0`, cohort discovery n=247, bộ lọc radiomics `l1_strength 0.1` (`ctx.rad_filters`, KHÔNG `clean_rad_filters`), vá scipy>=1.14 của nlp_run. Model: DyAM (`allcombo.run_combo.train`, bản có guard 0-feature), OvO (`train_ovo_safe`), +NLP (`cnl_nlp`, `no_scale`; thay Labs ở 2 tổ hợp có Labs, thêm vào base ở các tổ hợp còn lại). Mỗi ô: single-run (seed 42, thứ tự tự nhiên như bài) + 5 seed [42,7,123,2024,31337] hoán vị nhãn. Lệch so notebook: không dùng `attention_gate_enabled=False` cho đơn nguồn (không ảnh hưởng, DyAM=OvO hệt nhau ở 1 nguồn).

| Tổ hợp | DyAM 1r | OvO 1r | DyAM+NLP 1r | OvO+NLP 1r | DyAM 5s | OvO 5s | DyAM+NLP 5s | OvO+NLP 5s |
|---|---|---|---|---|---|---|---|---|
| Rad+IHC-G+Gen+PDL1 (BM1) | 0.7816 | **0.8015** | **0.8123** | **0.8015** | 0.7599 | 0.7704 | 0.7789 | **0.7847** |
| Rad+IHC-G+Gen+PDL1+Labs (BM2) | 0.7886 | 0.7851 | **0.8123** | 0.8015 | 0.7592 | 0.7692 | 0.7789 | 0.7847 |
| PDL1+Gen (BM4) | 0.6932 | 0.7157 | 0.7266 | 0.7372 | 0.7072 | 0.7183 | 0.7379 | 0.7472 |
| Rad+Gen (BM3) | 0.7392 | 0.7563 | 0.7476 | 0.7514 | 0.7093 | 0.7163 | 0.7186 | 0.7191 |
| **Trung bình 21 tổ hợp** | 0.7166 | 0.7182 | 0.7368 | 0.7334 | 0.7062 | 0.7110 | 0.7163 | 0.7192 |

- Tái tạo đối chiếu bài gốc (single-run): DyAM BM1 0.7816 (bài 0.7839), BM2 0.7886 (0.7879), BM3 0.7392 (0.7384), BM4 0.6932 (0.6931) → khớp ≤0.003. Con số "~0.8" xuất hiện ở single-run: OvO BM1 0.8015 (bài ghi 0.8003), DyAM+NLP 0.8123.
- 5 seed thấp hơn single-run cả 4 model ở các tổ hợp lớn (BM1: DyAM −0.022, OvO −0.031, DyAM+NLP −0.034, OvO+NLP −0.017): ~0.8 là một phân hoạch thuận lợi; ước lượng ổn định 0.76–0.78.
- Số tổ hợp OvO > DyAM (ngưỡng 0.0005): single-run 9/6, 5 seed 11/3; OvO+NLP > DyAM+NLP: single-run 5/14 (DyAM+NLP nhỉnh), 5 seed 10/6. OvO+NLP − OvO: +0.015 (1r), +0.008 (5s); DyAM+NLP − DyAM: +0.020, +0.010.
- Thứ hạng giữa OvO+NLP và DyAM+NLP ĐẢO giữa single-run (−0.003) và 5 seed (+0.003); khác biệt trong nhiễu một phân hoạch.
- Đường chạy khác `clean_rad_filters` nên số 5 seed khác nhẹ Bảng 4/5 của bài (BM1 OvO+NLP 0.7847 vs 0.78 ở 20 seed).

**Verdict:** KEEP làm kiểm tra tái lập; không đổi số trong bài. Xác nhận: ~0.8 chỉ đạt ở single-run; 5 seed cho 0.76–0.785, OvO+NLP cao nhất ở BM1 5 seed.

### 2.58. Vì sao lần chạy đơn (single-run) dao động: tách seed khởi tạo vs phân hoạch fold (2026-10-12)

**Mã:** `experiments/nlp_verify/run_single_variance.py` (BM1/BM2/BM3) · **Kết quả:** `experiments/results/single_variance_{BM1,BM2,BM3}.json` (+ `.log`).
**Config:** dùng HÀM GỐC `lung_helpers.train` (DyAM) và `train_ovo` (đã đối chiếu với `main:lung_helpers.py`: cùng logic, chỉ khác xử lý GPU/kiểm tra khoá). epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, KFold 10 shuffle `random_state=0`, cohort discovery n=247, bộ lọc rad `l1_strength 0.1`, không NLP. (A) 20 seed khởi tạo trên phân hoạch tự nhiên; (B) seed 42 + 20 phân hoạch (hoán vị thứ tự bệnh nhân); (D) seed 42 lặp lại; (C) BM1 với `clean_rad_filters`. Hàm gốc crash `0 feature(s)` ở 2/20 phân hoạch (thứ 9, 10) ở cả DyAM và OvO — ghi None, không bỏ qua fold.

| | DyAM 1r | OvO 1r | DyAM, đổi phân hoạch (n=18) mean±sd [min,max] | OvO, đổi phân hoạch | OvO−DyAM (18 phân hoạch chung) |
|---|---|---|---|---|---|
| BM1 | 0.7816 | 0.8015 | 0.7581±0.019 [0.719,0.798] | 0.7670±0.016 [0.735,0.794] | +0.009, OvO hơn 16/18 |
| BM2 | 0.7886 | 0.7851 | 0.7588±0.014 [0.738,0.778] | 0.7690±0.012 [0.741,0.785] | +0.010, 17/18 |
| BM3 | 0.7392 | 0.7563 | 0.7118±0.015 [0.684,0.732] | 0.7090±0.016 [0.683,0.738] | −0.003, 7/18 |

- **Seed khởi tạo không có tác dụng:** 20 seed cho cùng AUC đến 16 chữ số; seed 42 lặp lại hệt nhau. Nguyên nhân: `AttentionMatrix.__init__` và `AttentionMatrixOvO.__init__` gọi cứng `torch.manual_seed(42)` (cả trong `main`), huấn luyện full-batch nên không còn ngẫu nhiên.
- **Toàn bộ dao động đến từ cách chia fold** (sd 0.012–0.019, khoảng 0.04–0.08). Phân hoạch tự nhiên của notebook cao hơn 89–100% các phân hoạch khác → lần chạy đơn ~0.78/0.79/0.80 là một phân hoạch thuận lợi.
- Bộ lọc sạch BM1 (1 lần, vì seed không ảnh hưởng): DyAM 0.7960, OvO 0.7909 (gốc 0.7816 / 0.8015) — đổi bộ lọc dịch ±0.02, cùng cỡ với dao động phân hoạch. (Lần đầu OvO "lọc sạch" ra 0.8015 do cờ `_CLEANED` vẫn bật sau khi nạp lại dữ liệu → không dùng; chạy lại đúng.)
- Chiều OvO > DyAM nhất quán ở BM1, BM2 (16–17/18) nhưng không ở BM3 (7/18).

**Verdict:** KEEP làm chẩn đoán; không đổi số trong bài. Khuyến nghị: không tuyên bố bằng một lần chạy đơn; dùng ≥ nhiều phân hoạch (hoán vị) như harness 5/20 seed.

### 2.59. Dựng môi trường ghim của bài gốc và chạy code `main` (2026-10-12)

**Script:** scratchpad `run_orig.py` (gọi `lung_helpers.py` nguyên bản từ `git show main:lung_helpers.py`, `train()` không có tham số seed) · **Kết quả:** `experiments/results/orig_env_repro.{json,log}` · Tài liệu: `document/2026-10-12_original-environment/original-environment.md`.
**Môi trường:** `conda env doan-orig`: Python 3.9.13, torch 1.11.0+cpu, numpy 1.22.4, scipy 1.8.1, scikit-learn 1.1.1, pandas 1.4.2, lifelines 0.27.1 (đúng `requirements.txt` của `main` và siêu dữ liệu notebook).
**Config:** epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, KFold 10 shuffle `random_state=0`, cohort discovery n=247, bộ lọc rad `l1_strength 0.1`, 1 lần chạy; DyAM; không NLP/OvO.
**Kết quả:** Rad+IHC-A+Gen+PDL1 0.7640 (html 0.7982), Rad+IHC-G+Gen+PDL1 0.7839 (0.7833), Rad+IHC-G+Gen 0.7871 (0.7826), Rad+Gen 0.7384 (0.7593), Rad 0.6871 (0.7014), IHC-A+Gen 0.7148 (0.7150).
- Trùng đến 6 chữ số với output của `Figures-Finalized.ipynb` → môi trường + code gốc tái hiện đúng file `.ipynb`.
- Khác `figures-finalized.html` ở các tổ hợp có radiomics (tới −0.034); các hàng không radiomics khớp ≤0.0002 → HTML là lần chạy khác của tác giả với đầu vào/bước lọc radiomics khác, không do phiên bản thư viện.
**Verdict:** KEEP làm bằng chứng nguồn gốc; con số 0.798 không tái hiện được bằng dữ liệu/code hiện có.

### 2.60. Chạy từng ô Figures-Finalized.ipynb (main) trong môi trường ghim, đối chiếu HTML (2026-10-12)

**Script:** scratchpad `run_cells.py` (exec các ô 1–13, 16, 17 của notebook `main`; `lung_helpers.py` của `main`; nạp dữ liệu từ đầu, không đệm; cwd `D:\code\master\doan`) · **Kết quả:** `experiments/results/notebook_cells_orig_env.{log,json}` · Chi tiết + danh sách sai khác: `document/2026-10-12_original-environment/sai-khac-can-dieu-tra.md`.
**Config:** môi trường `doan-orig` (Python 3.9.13, torch 1.11.0+cpu, numpy 1.22.4, scipy 1.8.1, scikit-learn 1.1.1, pandas 1.4.2, lifelines 0.27.1); cấu hình trong notebook (epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, `attention_gate_enabled` False cho đơn nguồn, l1_strength 0.1, KFold 10 `random_state=0`, cohort discovery n=247, 1 lần chạy).
**Kết quả:** mọi hàng không radiomics khớp HTML ≤0.0004 (LR Clinical/IHC/PDL1/Gen/Path, DyAM IHC-A/Gen/IHC-A+Gen/PDL1+Gen); mọi hàng có radiomics lệch: LR Rad-LN +0.054, LR Rad-Average +0.012, DyAM Rad −0.014, Rad+IHC-A −0.008, Rad+Gen −0.021, Rad+IHC-A+Gen+PDL1 −0.034 (0.7640 so với 0.7982), Rad+IHC-G+Gen+PDL1 +0.0005.
- Lệch xuất hiện cả ở LR đơn nguồn → nguyên nhân là dữ liệu/lọc/chọn tổn thương radiomics, không phải DyAM. Bộ nhớ đệm của harness không phải nguyên nhân (số trùng khi nạp từ đầu).
**Verdict:** KEEP (chẩn đoán); nguyên nhân chưa xác định, ghi ở sai-khac-can-dieu-tra.md (D1, D7–D9).

#### Đính chính §2.59–§2.60 (2026-10-12)
Các mục §2.59, §2.60 và `original-environment.md` ban đầu coi output của `Figures-Finalized.ipynb` ở thư mục làm việc là "output của tác giả". Sai: file đó là bản đã chạy lại trên máy này. So MD5 với cache Synapse (`~/.synapseCache`): 9 file dữ liệu và các file code của nhánh `main` (`lung_helpers.py`, `.ipynb`, `requirements.txt`, `.html`) KHỚP bản gốc trên Synapse (bản phát hành 2022-08-18; 2025-04-30 là ngày tải về). Output lưu trong `.ipynb` của `main` giống hệt `.html` (DyAM Rad+IHC-A+Gen+PDL1 0.798169). Kết luận đúng: chạy lại code + dữ liệu giống hệt, trong môi trường ghim, cho 0.763993 — KHÔNG tái hiện được output lưu của tác giả ở các hàng có radiomics (hàng không radiomics khớp ≤0.0004). Nhiễu số học (seed saga, tol, seed toàn cục) và 19 biến thể siêu tham số của bước chọn đặc trưng không giải thích được (`experiments/results/radiomics_{sensitivity,grid,grid2}.*`). Xem `document/2026-10-12_original-environment/sai-khac-can-dieu-tra.md` D11.

### 2.61. Điều tra nguyên nhân lệch radiomics so với output tác giả (2026-10-12)

**Mã/kết quả:** scratchpad `sens.py`, `grid.py`, `order.py`, `pdver.py`; `experiments/results/radiomics_{sensitivity,grid,grid2,order}.{log,json}`, `pandas_{1.5.3,2.0.3,2.2.3}.log`. Chi tiết: `document/2026-10-12_original-environment/sai-khac-can-dieu-tra.md` (D11, D12).
**Config:** môi trường `doan-orig` (Python 3.9.13, torch 1.11.0, numpy 1.22.4, scipy 1.8.1, scikit-learn 1.1.1, pandas 1.4.2); `lung_helpers.py` của `main`; ô nạp dữ liệu 1–12 của notebook; `train_LR` (LR Rad-PC, LR Rad-LN), cấu hình gốc trừ chỗ thử nghiệm; AUC một lần chạy; cohort discovery n=247.
**Kết quả:** (1) đổi seed saga/tol/seed toàn cục: LR Rad-LN 0.680–0.688 (mục tiêu 0.627) → không phải nhiễu số học; (2) 19 biến thể của bước chọn đặc trưng: không biến thể nào tái tạo cả PC 0.6393 và LN 0.6269; (3) hoán vị thứ tự bệnh nhân: PC sd 0.021, LN sd 0.041 (LN 0.564–0.707), giá trị của tác giả nằm trong phân bố → thứ tự bệnh nhân khác đủ giải thích; (4) pandas ≥1.5 không chạy được code gốc (`pd.core.common.SettingWithCopyWarning`) → tác giả dùng pandas ≤1.4.x.
**Verdict:** KEEP (chẩn đoán); chưa tái hiện được 0.798; nghi nhất là thứ tự bệnh nhân/phân hoạch khác nhau do trạng thái chạy notebook hoặc nền tảng; chưa xác định cơ chế.

### 2.62. Vòng điều tra môi trường: tái hiện output của tác giả đến 6 chữ số một phần (2026-10-12/13)

**Mã/kết quả:** scratchpad `skver.py`, `lrrows.py`, `dyamrows.py`, `dyamrows2.py`, `paramgrid.py`, `run_cells.py`; `experiments/results/{matrix_*,m2_*,m3_*,lr_*,dy_*,dy2_*,dy3_*,dyth_*,threads_*,paramgrid,notebook_cells_sk0242,final_check_doan_orig}.{log,json}`. Chi tiết: `document/2026-10-12_original-environment/sai-khac-can-dieu-tra.md` D13–D15.
**Config:** `lung_helpers.py` của `main` (khớp MD5 Synapse), dữ liệu gốc (khớp MD5), ô notebook 1–13, 16, 17; epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, `l1_strength` 0.1, KFold 10 `random_state=0`, cohort discovery n=247, một lần chạy; DyAM, không NLP/OvO. Các môi trường thử: scikit-learn 0.24.0–1.3.2, scipy 1.7.3–1.10.1, numpy 1.19.5–1.24.4, torch 1.8.1–1.13.1, pandas 1.2.5–1.4.2 (cài `--target`, đặt qua `PYTHONPATH`, in phiên bản thực nạp).
**Kết quả:** môi trường tốt nhất = scikit-learn 0.24.2 + numpy 1.20.3 + scipy 1.8.1 + pandas 1.4.2 + torch 1.11.0 + Python 3.9.13 (đã cố định trong env `doan-orig`). Khớp 6 chữ số: LR Rad-PC 0.639341, LR Rad-LN 0.626938, DyAM Rad 0.7014, DyAM Rad+IHC-A 0.680753, DyAM IHC-A 0.605631, DyAM Gen 0.675850. Còn lệch: mọi hàng gộp Genomics với nguồn khác (IHC-A+Gen −0.0002, Rad+Gen −0.0015, Rad+IHC-A+Gen +0.0055, DyAM Rad+IHC-A+Gen+PDL1 0.787881 so với 0.798169, −0.0103). Torch, pandas, scipy, số luồng, 13 cấu hình tham số huấn luyện không đổi các hàng còn lệch.
**Verdict:** KEEP (chẩn đoán); chưa tái hiện được 0.798169; scikit-learn trong `requirements.txt` (1.1.1) sai, bản thực là ≤0.24.2. Còn nghi: trạng thái `lung_helpers.py` lúc tác giả chạy hoặc sai số theo nền tảng.

### 2.63. Tái hiện output của tác giả trên Linux (2026-10-13)

**Mã/kết quả:** `experiments/nlp_verify/linux_repro/{run_cells_linux.py,setup_env3.sh,run_linux.sh,chk_env.sh}` · `experiments/results/notebook_cells_linux_py38.{log,json}`. Chi tiết: `document/2026-10-12_original-environment/sai-khac-can-dieu-tra.md` D17.
**Config:** WSL2 Ubuntu-22.04 (kernel 6.6.87.2), Python 3.8.15, torch 1.8.1+cpu, scikit-learn 0.24.2, numpy 1.20.3 (OpenBLAS), scipy 1.6.3, pandas 1.2.5, lifelines 0.26.0; `lung_helpers.py`/notebook của `main` (MD5 khớp Synapse); dữ liệu gốc (MD5 khớp); ô 1–13, 16, 17; epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, `l1_strength` 0.1, KFold 10 `random_state=0`, cohort discovery n=247, một lần chạy; DyAM, không NLP/OvO. Môi trường đặt hoàn toàn trên ổ D:.
**Kết quả:** DyAM Rad+IHC-A+Gen+PDL1 0.798256 (tác giả 0.798169, lệch +0.000087 = một cặp bệnh nhân); Rad+IHC-G+Gen+PDL1 0.783435 (0.783348); Rad+Gen 0.759285 và Rad+IHC-A+Gen 0.754228 khớp 6 chữ số; DyAM Rad, IHC-A, Gen, LR Rad-PC, LR Rad-LN khớp 6 chữ số; 11/25 hàng khớp tuyệt đối. Còn lệch ≤0.0003 ở IHC-A+Gen, Rad+IHC-A, PDL1+Gen (cùng giá trị trên Windows). Windows với cùng thư viện: 0.7879.
**Verdict:** KEEP; con số ~0.80 của bài gốc được tái hiện trên Linux với scikit-learn 0.24.2. Lệch giữa Windows và Linux do nền tảng số học. `requirements.txt` của tác giả ghi sai scikit-learn.

### 2.64. 21 tổ hợp của bài gốc trong môi trường tham chiếu: DyAM, OvO, NLP+DyAM, NLP+OvO; 1 seed và 5 seed (2026-10-13)

**env: ref-linux-py38** (Linux WSL2 Ubuntu-22.04, Python 3.8.15, torch 1.8.1+cpu, scikit-learn 0.24.2, numpy 1.20.3, scipy 1.6.3, pandas 1.2.5).
**Mã/kết quả:** `experiments/reference_env/run_all21_ref.py` (`--part=i/4`, chạy qua `run_ref.sh`) · `experiments/results/ref_all21_part{0..3}.json` (+ `.log`) · bảng đầy đủ: `document/2026-10-13_reference-training-config/ref-all21-results.md` (sinh bởi `make_ref_table.py`).
**Config:** epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False, `attention_gate_enabled=False` cho đơn nguồn, bộ lọc radiomics `ctx.rad_filters` (l1_strength 0.1), KFold 10 shuffle `random_state=0`, cohort discovery n=247; DyAM = `allcombo.run_combo.train`, OvO = `train_ovo_safe` (cùng logic hàm gốc, thêm bỏ qua fold hết đặc trưng); NLP = `cnl_nlp` (MiniLM 384-d, `no_scale`) thay Labs hoặc thêm vào base; 1 seed = seed 42 thứ tự tự nhiên; 5 seed = [42, 7, 123, 2024, 31337] hoán vị nhãn.
**Kết quả (trung bình 21 tổ hợp | trung bình 8 tổ hợp ≥5 nguồn | cao nhất):**
- 1 seed: DyAM 0.7224 | 0.7765 | 0.7983; OvO 0.7214 | 0.7724 | 0.8005; NLP+DyAM 0.7371 | 0.7880 | 0.8030; NLP+OvO 0.7401 | 0.7942 | 0.8243 (đều ở Rad+IHC-A/G+Gen+PDL1).
- 5 seed: DyAM 0.7067 | 0.7431 | 0.7635; OvO 0.7111 | 0.7543 | 0.7714; NLP+DyAM 0.7163 | 0.7569 | 0.7857; NLP+OvO 0.7204 | 0.7703 | 0.7848.
- OvO − DyAM: 1 seed trung bình −0.0010 (hơn 6, kém 11, ngang 4), 8 tổ hợp lớn −0.0042 (3/8 hơn); 5 seed +0.0044 (hơn 10, kém 7, ngang 4), 8 tổ hợp lớn +0.0112 (7/8 hơn). NLP+OvO − NLP+DyAM: 1 seed +0.0029 (12/7/2), 8 lớn +0.0062 (6/8); 5 seed +0.0041 (8/9/4), 8 lớn +0.0133 (5/8). NLP − không NLP: DyAM 1 seed +0.0148, 5 seed +0.0095; OvO 1 seed +0.0186, 5 seed +0.0093 (8/8 tổ hợp lớn hơn ở cả hai).
- So với tác giả (DyAM, 1 seed): khớp 6 chữ số ở IHC-A, Gen, Rad, Rad+Gen, Rad+IHC-A+Gen; lệch ≤0.000087 (một cặp) ở PDL1+Gen, Rad+IHC-G+Gen, Rad+IHC-A+Gen+PDL1 (0.798256 vs 0.798169), Rad+IHC-G+Gen+PDL1; lệch 0.0002 ở Rad+IHC-A, IHC-A+Gen.
- Thứ hạng giữa OvO và DyAM đảo giữa 1 seed và 5 seed (1 seed không dùng được để so mô hình, như §2.58).
**Verdict:** KEEP làm bảng tham chiếu trong môi trường chuẩn; số liệu không so trực tiếp với các bảng sinh trên Windows (§2.x trước 2026-10-13, bảng bài báo).

### 2.65. Chạy lại mọi số liệu của bài trong môi trường tham chiếu (2026-10-13)

**env: ref-linux-py38** (cùng §2.64). Người dùng yêu cầu chạy lại tất cả số liệu chưa cập nhật theo môi trường mới.
**Mã/kết quả (tất cả ở `experiments/results/ref/`, chạy qua `experiments/reference_env/run_paper_ref.py` + `run_chain.sh`, 3 chuỗi song song, OMP/OPENBLAS threads 2):**
- Bảng 21 tổ hợp, Bảng S3 (1 phân hoạch): từ §2.64 (`ref_all21_part*.json`).
- Bảng 20-seed DyAM+NLP vs OvO+NLP (6 tổ hợp k≥5): `ovo_proof/nlp20_{dyam,ovo}__*.csv`.
- Bỏ từng nguồn (BM1, OvO+NLP, 20 seed): `ovo_proof/loso_ovo_*.csv`. Vector sẵn-có (OvO+NLP+miss, 20 seed): `ovo_proof/nlp20miss_ovo__*.csv`.
- Giữ-ngoài theo số chụp (T1) và đường cong học (T2): `order_ovo_t1.json`, `order_ovo_t2.json`.
- Bảng 2–5 (11 biến thể lâm sàng × 5 seed, AUC/C-index/Cox HR/KM): `verify_nlp_5seed.json`; điểm nguy cơ `survival_scores/*.csv` (11 file); tdAUC: `tdauc_5seed.json` (`compute_tdauc.py --ref`, tính ở venv_np2 Windows từ điểm đã lưu, vì sksurv không có trong ref-env).
- Bio_ClinicalBERT: embedding tạo 1 lần ở Windows (doan-env, transformers 4.40.2), cache `D:/repro_linux/cache/bioclin_embed_v2.parquet` (`make_bioclin_cache.py`); NLP MiniLM dùng cache sẵn.
**Config:** epochs 125, lr 0.01, alpha 0.001, beta 0.0, cross_modality False, attention_gate_enabled=False đơn nguồn, bộ lọc radiomics **`ctx.rad_filters` gốc** (không `clean_rad_filters`), KFold 10 random_state 0, n=247; 5 seed [42,7,123,2024,31337] hoán vị nhãn; 20 seed = 5 seed + [11,22,33,44,55,66,77,88,99,101,202,303,404,505,606]; NLP no_scale; `train` / `train_ovo_safe`.
**Lưu ý lệch:** lần chạy 20-seed đầu dùng `build_l1` (làm sạch cột suy biến, kiểu Windows) nên sai cấu hình ref → giữ riêng ở `results/ref_cleanfilter/` (BM1 DyAM+NLP 20-seed 0.773 so với 0.781 khi dùng bộ lọc gốc), KHÔNG dùng trong bài; chạy lại với `ctx.rad_filters` (patch `build_l1` trong `run_paper_ref.py`, cờ `--clean` giữ hành vi cũ). Hai job chết vì thiếu RAM khi chạy 9 job song song → chạy lại 3 chuỗi.
**Kết quả chính (mới | trước đó trên Windows):**
- 5 seed BM1: DyAM 0.763±0.022, OvO 0.771±0.021, DyAM+NLP 0.786±0.017, OvO+NLP 0.785±0.022 | 0.751, 0.767, 0.773, 0.782. OvO > DyAM 10/21 (kém 7, ngang 4) | 12/5/4; k≥5: 7/8, +0.011 | giữ nguyên; NLP+OvO ≥ NLP+DyAM 5/8 (+0.013) | 7/8.
- 20 seed (6 tổ hợp): OvO+NLP hơn ở 5/6 tổ hợp (một hòa), +0.013/phân hoạch, ensemble +0.006 (4/6); BM1 0.787±0.015 vs 0.781±0.013, ensemble 0.796 [0.730, 0.854] vs 0.795 [0.731, 0.854]; tổ hợp 7 nguồn +0.033 (20/20), ensemble +0.024 [+0.001, +0.048] (khoảng duy nhất loại 0) | BM1 0.780 vs 0.771.
- Bỏ từng nguồn (đầy đủ 0.787): pathology −0.028, radiomics −0.028, PD-L1 −0.017, NLP −0.012, genomics −0.012 (18–20/20 phân hoạch) | −0.033/−0.015/−0.014.
- Vector sẵn-có: +0.015, +0.016 ở Rad+IHC-A+Gen, Rad+IHC-G+Gen (19/20, 20/20); +0.001..+0.005 ở 3 tổ hợp; −0.001 ở bảy nguồn TMB | +0.017 ở hai tổ hợp không PD-L1.
- T1 (124 bệnh nhân sau): BM1 OvO+NLP 0.703 [0.587, 0.819], DyAM+NLP 0.700 [0.587, 0.812]; bảy nguồn 0.707 vs 0.678 | 0.743 vs 0.708. T2: OvO+NLP 0.707→0.785, DyAM+NLP 0.677→0.786 (89→222) | 0.709→0.782, 0.679→0.780.
- Bảng NLP (DyAM 5 seed): IHC-A không LS 0.741, Labs 0.732, NLP raw 0.749, PCA16 0.744, Labs+NLP 0.735, Labs+PCA16 0.736, BioClin 0.732; IHC-G 0.765/0.752/0.772/0.768. NLP raw − Labs +0.017/+0.020; NLP raw − không LS +0.007/+0.007 (trước: −0.006/+0.001). Cox HR IHC-A 4.12 (không), 4.13 (labs), 4.52 (NLP raw); IHC-G 4.02/4.54/4.63. C-index 0.608–0.623 (labs cao nhất ở cả hai nhánh). KM: 55/55 > 3.84, 53/55 > 7.88 (nhỏ nhất 7.0). tdAUC 0.63–0.69, ở 18 tháng nhóm không LS cao nhất (0.662, 0.655).
**Bài đã cập nhật:** `paper/make_results_tables.py --ref` (bảng 2–5, 21 tổ hợp, 20-seed, LOSO, miss; bản Windows cũ sao lưu ở `paper/tables/_windows_2026-10-12/`), results/discussion/methods/conclusion/main.tex/supplementary.tex (S2–S5, S3), Hình 3 (đọc `ref/verify_nlp_5seed.json`; `check_figures.py` 0 lỗi), PDF (main + supplementary), Word (`port_word.py` với dữ liệu ref).
**Chưa chạy lại (giữ số Windows, đã nêu rõ):** Hình 4 (KM) và Hình 5 (C-index/tdAUC/Cox forest/IBS) từ phân hoạch đơn gốc; Hình S3 (bootstrap), S4 (scatter 20 cấu hình), S5; ước lượng thổi phồng do chọn mô hình 0.011–0.024 (§2.40); các mục kiểm chứng ngoài (rad_valid/path_valid, MSK-IMPACT, Rizvi/Hellmann, MMIST, Curie) không nằm trong bài chính.
**Verdict:** KEEP — toàn bộ bảng chính của bài nay cùng một môi trường; kết luận định tính giữ (OvO ≥ DyAM ở k≥5, NLP > Labs) nhưng NLP+OvO không còn vượt NLP+DyAM ở BM1 (5 seed) và khoảng cách ở bảy nguồn thu hẹp về 20-seed.

### 2.65b. Dựng lại Hình 4, 5, S3, S4 từ dữ liệu thật (2026-10-13)
env: ref-linux-py38 cho điểm nguy cơ; vẽ hình ở Windows (`paper/make_real_figs.py`, matplotlib + lifelines). **Phát hiện:** `paper/drafts/make_fig4_km.py` vẽ KM từ thời gian sống MÔ PHỎNG (hàm mũ hiệu chỉnh khớp chi-bình-phương đích), `make_fig3_fig5.py` dùng số gõ cứng của phân hoạch Windows cũ (kể cả IBS), `make_figS3_bootstrap.py` là xấp xỉ chuẩn giải tích, `make_figS4_ovo_scatter.py` đọc Excel cũ. Đã thay bằng: Hình 4 = KM thật (seed 42, chia tại TRUNG VỊ điểm nguy cơ, đúng như `km_chi2`; χ² 17.9/17.4/14.0/18.3; n=247, 209 sự kiện); Hình 5 = mean±sd 5 phân hoạch (C-index, tdAUC, Cox HR, log-rank χ²; panel IBS bỏ vì chưa tính lại, `fig5d_ibs.pdf` -> `fig5d_km_chi2.pdf`); Hình S3 = bootstrap bệnh nhân thật NLP raw − Labs (IHC-G) Δ=+0.0195, khoảng 95% [−0.0015, +0.0403]; Hình S4 = 21 cấu hình một phân hoạch từ `ref_all21`. Sửa chữ: bài ghi "chia tại risk score = 0" nhưng mã chia tại trung vị -> đã sửa thành trung vị (Methods, Bảng 5, chú thích hình). Đã bỏ mọi câu về IBS. Verdict: KEEP; các hình cũ (mô phỏng) KHÔNG dùng nữa.

### 2.65c. Số đặc trưng radiomics giữ lại theo fold, ref-linux-py38 (2026-10-13)
env: ref-linux-py38; `experiments/reference_env/count_rad_features_ref.py` -> `experiments/results/ref/rad_feature_counts.json`. Cấu hình: ctx.rad_filters (l1_strength 0.1, robustness 0.15, outlier 6), KFold(10, shuffle, random_state=0), tổ hợp BM1. Trung vị [min-max] qua 10 fold: PC 35.5 [24-45] (163 bn), PL 7.5 [1-13] (21 bn), LN 25 [21-41] (67 bn); cột thô 1688. Trước (Windows): 35 [28-45], 9 [3-20], 25 [17-32]. Đã cập nhật bảng Supp và sửa Methods: Bio_ClinicalBERT so với MiniLM ΔAUC −0.017 -> −0.012. Verdict: KEEP.

### 2.65d. Mức thổi phồng do chọn mô hình, tính lại từ các cặp ref (2026-10-13)
env: ref-linux-py38 (điểm đã lưu; phân tích bootstrap chạy ở Windows). `experiments/archloop/analyze_selection_bias_ref.py` -> `experiments/results/ref/selection_bias.json`. SE hiệu AUC ensemble BM1 (20 seed, bootstrap bệnh nhân 3000) cho 6 cặp gần nhau: bỏ rad 0.0222, path 0.0153, gen 0.0176, pdl1 0.0108, nlp 0.0070 (so với OvO+NLP đầy đủ) và có/không vector sẵn-có 0.0123; trung vị 0.0138 (trước, Windows: 0.0105). K = 52. E[max] khi không có hiệu ứng thật: ρ=0.8 0.0139, ρ=0.5 0.0221, ρ=0 0.0313 (trước 0.011/0.017/0.024). Bài đã đổi 0.011-0.024 -> 0.014-0.031. Hạn chế: K=52 là số ứng viên lịch sử, không chạy lại được; cặp loso gồm thay đổi lớn nên SE có thể hơi cao. Verdict: KEEP.

### 2.66. Bản bài MỘT PHÂN HOẠCH (draft cho người dùng xem, 2026-10-13)
env: ref-linux-py38. `experiments/reference_env/run_single_partition_ref.py` -> `experiments/results/ref/single/{single_partition.json,scores/*.csv}`; tdAUC: `experiments/survival/compute_tdauc_single.py` -> `tdauc_single.json`. Config: epochs 125, lr 0.01, alpha 0.001, beta 0, KFold 10 random_state 0, seed 42, THỨ TỰ TỰ NHIÊN (giao thức bài gốc; KHÔNG chọn phân hoạch tốt nhất), ctx.rad_filters gốc, `train`/`train_ovo_safe`. Kết quả: BM1 DyAM 0.7834 [0.714,0.853], OvO 0.8005 [0.735,0.866], DyAM+NLP 0.8030 [0.738,0.868], OvO+NLP 0.8243 [0.765,0.883]; 11 biến thể lâm sàng AUC IHC-A 0.772/0.768/0.786/0.784/0.758/0.767/0.770, IHC-G 0.797/0.792/0.812/0.805. Bản bài ở `paper_1seed/` (PDF `paper_1seed/main.pdf`, supplementary), bản nhiều seed `paper/` giữ nguyên. Chưa xuất Word cho bản này. Verdict: draft; số liệu một phân hoạch nhạy với phân hoạch, đã ghi rõ ở Limitations.

### 2.67. Sống còn một phân hoạch: Linux (ref-linux-py38) so với Windows (doan-orig) (2026-10-13)

**Câu hỏi:** sự sai khác sống còn giữa hai môi trường còn không khi cùng thư viện chính? **Script:** `experiments/reference_env/run_single_partition_win.py` (bản sao của `run_single_partition_ref.py`, bỏ phần BM1, ghi vào `experiments/results/win_orig/single/`), log `experiments/results/win_single.log`; Linux lấy từ `results/ref/single/single_partition.json`.
**Config:** như §0 (epochs 125, lr 0.01, alpha 0.001, beta 0, KFold(10, random_state=0), seed 42 thứ tự tự nhiên, n=247, `ctx.rad_filters`, DyAM `train`, 11 biến thể lâm sàng). Windows: env `doan-orig` (Python 3.9.13, sklearn 0.24.2, numpy 1.20.3, scipy 1.8.1, pandas 1.4.2, torch 1.11.0+cpu, lifelines 0.27.1, OMP 4, cache riêng `D:\repro_win\cache`). Linux: env ref-linux-py38. Embedding NLP dùng chung cache.
**Kết quả (|Linux − Windows| trên 11 biến thể):** AUC max 0.0041 / mean 0.0013; C-index max 0.0077 / mean 0.0024; Cox HR max 0.55 / mean 0.18 (IHC-A NLP raw 5.81 vs 5.35; IHC-G NLP-PCA16 5.57 vs 5.02; nhiều ô khác chênh <0.2); KM χ² chênh tới 3 (IHC-A NLP raw 21.8 vs 19.0). Thứ tự định tính giữ nguyên ở AUC; Cox HR/KM là chỉ số nhạy nhất.
**Verdict:** KEEP làm bằng chứng — với sklearn 0.24.2 cố định, AUC/C-index hai nền tảng gần như trùng; khác biệt sống còn chủ yếu ở Cox HR (min-max scale điểm) và KM, không phải ở điểm nguy cơ.

### 2.68. Sống còn một phân hoạch: Linux (ref) so với Windows `doan-env` (sklearn 1.1.1) (2026-10-13)

**Script:** `experiments/reference_env/run_single_partition_win2.py` (bản sao `_win.py`, ghi `results/win_doanenv/single/`), log `results/win2_single.log`. **Config:** như §2.67; env `doan-env` (Windows, sklearn 1.1.1, numpy 1.22.4, scipy 1.8.1, pandas 1.4.2, torch 2.5.1+cu121, lifelines 0.27.1; cache riêng `D:\repro_win\cache2`). **Kết quả:** xem bảng so sánh trong phản hồi cùng ngày; AUC lệch tới ~0.03 so với Linux, NLP không còn cao hơn nhãn "không lâm sàng" ở nhánh IHC-A/IHC-G (Cox HR NLP 4.35–4.75 < không LS 5.04–5.65). **Verdict:** KEEP làm bằng chứng — phiên bản sklearn 1.1.1 làm đổi chiều hiệu ứng NLP; không dùng cho bài.

### 2.69. NLP có cải thiện sống còn không — 4 cách kiểm chứng (2026-10-13)

**Script:** `experiments/reference_env/survival_nlp_checks_ref.py` · **Kết quả:** `experiments/results/ref/survival_nlp_checks.json`, log `ref/survival_checks.log`. **env: ref-linux-py38** (lifelines 0.26.0). Đọc điểm nguy cơ 5 seed đã lưu (`ref/survival_scores`, DyAM config chuẩn §0; không train lại) cho A–C; D là Cox phạt OOF (5 x KFold(10)) trên cohort khám phá n=247, không qua DyAM. HR dùng z-score điểm (HR/1 SD, hiệu chỉnh age, ecog, albumin, dnlr, liver_mets), không dùng min-max.
**A (từng seed):** NLP − không LS: ΔC ≈ 0.000 (IHC-A) và 0.0005–0.0012 (IHC-G), 2–3/5 seed dương; Δlog HR +0.011–0.013 (IHC-A, 3–4/5 dương), +0.023 (IHC-G, 3–4/5). NLP − Labs: ΔC −0.005 đến −0.006 (0–1/5 dương); Δlog HR +0.037–0.040 (5/5 dương).
**B (bootstrap bệnh nhân ghép cặp, ensemble 5 seed, 1000 lần):** NLP − không LS: ΔC −0.003 đến 0 (khoảng chứa 0, tỉ lệ dương 0.32–0.45); Δlog HR ≈ +0.00 (IHC-A, khoảng [−0.05, 0.05]) và +0.02 (IHC-G, tỉ lệ dương 0.81–0.83). NLP − Labs: ΔC −0.005 đến −0.007; Δlog HR +0.042–0.047 (tỉ lệ dương 0.94–0.97, khoảng chạm 0).
**C (giá trị tăng thêm trong Cox):** thêm điểm NLP vào điểm không LS: LR 0.03–0.06 (p 0.80–0.86, IHC-A), 1.2–1.3 (p 0.25–0.28, IHC-G); tương quan hai điểm 0.96–0.98 → NLP không mang thông tin sống còn bổ sung.
**D (trực tiếp, không DyAM, C-index OOF):** Labs số 0.648; NLP-PCA16 0.572; NLP raw (PCA32) 0.558 → riêng NLP dự đoán PFS kém Labs số rõ rệt.
**Verdict:** NEGATIVE cho tuyên bố "NLP cải thiện sống còn": không tăng C-index, không tăng thông tin tiên lượng thêm; chỉ còn Cox HR/1 SD nhỉnh hơn Labs (và hơi hơn không LS ở IHC-G) với độ lớn nhỏ. KEEP làm bằng chứng.

### 2.70. Kiểm chứng sống còn NLP trên điểm Windows (2026-10-13)

**Script:** như §2.69 (`survival_nlp_checks_ref.py`, tham số `CHK_OUT=win_checks CHK_SC=survival_scores`), chạy bằng env `doan-orig` (Windows); điểm nguy cơ 5 seed là `experiments/results/survival_scores` (sinh trên Windows 2026-10-12, KHÔNG phải ref-env). **Kết quả:** `experiments/results/win_checks/survival_nlp_checks.json`, log `results/win_checks.log`. Config train như §0; HR dùng z-score. **So với Linux (§2.69):** B ΔC NLP−không LS −0.007 (IHC-A), +0.001–0.002 (IHC-G); Δlog HR +0.04 (tỉ lệ dương 0.89–0.97) thay vì ≈0/+0.02; C thêm NLP vào không LS: IHC-A LR 3.5–3.8 (p 0.05–0.06), IHC-G LR 6.0–6.3 (p 0.012–0.014) thay vì p 0.25–0.86; D (Cox từ Labs/NLP, không DyAM) trùng hệt Linux (Labs 0.648, PCA16 0.572, raw 0.559). **Verdict:** KEEP làm bằng chứng — tín hiệu NLP tăng thông tin tiên lượng chỉ xuất hiện ở điểm Windows, không ở ref-env; C-index vẫn không tăng ở cả hai.

### 2.71. Sống còn của OvO + NLP ở BM1 (Rad+IHC-G+Gen+PDL1), 5 seed (2026-10-13)

**Script:** `experiments/reference_env/survival_ovo_nlp_ref.py` · **Kết quả:** `experiments/results/ref/survival_ovo_nlp.json`, điểm `ref/bm1_surv_scores/*.csv`, log `ref/survival_ovo_nlp.log`. **env: ref-linux-py38.** Config train như §0 (epochs 125, lr 0.01, alpha 0.001, beta 0, KFold(10, random_state=0), seeds [42,7,123,2024,31337] hoán vị nhãn, `make_filter` bộ lọc paper, DyAM `train` / OvO `train_ovo_safe`, no_scale cho NLP), n=247. HR = log HR/1 SD z-score, hiệu chỉnh age, ecog, albumin, dnlr, liver_mets; KM chi2 tại trung vị.
**Trung bình 5 seed (C-index / log HR / KM χ²):** DyAM 0.6161 / 0.425 / 12.3; OvO 0.6204 / 0.460 / 14.4; DyAM+NLP 0.6217 / 0.468 / 16.3; OvO+NLP 0.6226 / 0.495 / 17.9.
**Ghép cặp (ΔC, Δlog HR, số seed dương/5):** OvO+NLP − DyAM+NLP: +0.0009, +0.027 (5/5); OvO+NLP − OvO: +0.0022, +0.035 (5/5; LRT p 0.031); OvO+NLP − DyAM: +0.0064 (5/5), +0.070 (5/5; bootstrap [0.013, 0.106]; LRT p 0.0027); OvO − DyAM: +0.0043, +0.035; DyAM+NLP − DyAM: +0.0055, +0.043. Bootstrap ΔC của mọi cặp chứa 0.
**Verdict:** OvO+NLP có xu hướng cao nhất theo mọi chỉ số, nhưng so với DyAM+NLP chênh lệch nhỏ (ΔC +0.001, LRT p 0.16); phần lớn lợi thế so với DyAM gốc. Dùng được như kết quả phụ, không phải bằng chứng riêng của OvO.

### 2.72. Sống còn OvO + NLP ở BM1 trên Windows (2026-10-13)

**Script:** `survival_ovo_nlp_ref.py` (biến `CHK_OUT`), như §2.71. **env: KHÔNG phải ref-linux-py38** — Windows `doan-env` (sklearn 1.1.1, numpy 1.22.4, torch 2.5.1; `results/win_ovo_env/`, log `win_ovo_env.log`) và Windows `doan-orig` (sklearn 0.24.2, numpy 1.20.3, torch 1.11.0; `results/win_ovo_orig/`, log `win_ovo_orig.log`), OMP 2 luồng, cache riêng; không so trực tiếp với số ref. Config train như §0, 5 seed, BM1.
**Kết quả (C-index / log HR trung bình 5 seed):** doan-env: DyAM 0.6125/0.413, OvO 0.6179/0.457, DyAM+NLP 0.6208/0.474, OvO+NLP 0.6208/0.483; doan-orig: 0.6162/0.420, 0.6210/0.465, 0.6216/0.470, 0.6224/0.496. OvO+NLP − DyAM: doan-env ΔC +0.0083 (5/5), Δlog HR +0.070 (5/5, LRT p 0.0018); doan-orig ΔC +0.0061 (4/5), Δlog HR +0.076 (5/5, LRT p 0.0009). OvO+NLP − DyAM+NLP: doan-env Δlog HR +0.009 (3/5, p 0.92); doan-orig +0.026 (5/5, p 0.22).
**Verdict:** KEEP — hướng OvO+NLP > DyAM gốc lặp lại ở cả hai môi trường Windows (khác với kết quả NLP riêng lẻ); OvO+NLP ≈ DyAM+NLP.

### 2.73. Sống còn BM1 theo từng phân hoạch riêng lẻ (2026-10-13)

Không chạy thêm huấn luyện: đọc `per_seed` trong `results/ref/survival_ovo_nlp.json` (§2.71; env ref-linux-py38; 5 seed hoán vị nhãn) và tính C-index/log HR/KM cho phân hoạch tự nhiên seed 42 từ `results/ref/single/scores/bm1_*.csv` (cùng config §0, HR z-score hiệu chỉnh). **Phân hoạch tự nhiên seed 42 (C / log HR / KM χ²):** DyAM 0.646/0.475/17.4; OvO 0.640/0.477/20.3; DyAM+NLP 0.630/0.450/19.2; OvO+NLP 0.633/0.524/23.9. **Hoán vị nhãn 5 seed, OvO+NLP − DyAM:** Δlog HR 0.098, 0.108, 0.087, 0.029, 0.029 (5/5 dương); ΔC 0.004, 0.011, 0.011, 0.002, 0.005. Phân hoạch có log HR OvO+NLP cao nhất là seed 42 (0.529) ở ref. **Verdict:** KEEP — hướng Δlog HR dương giữ ở mọi phân hoạch; C-index ở phân hoạch tự nhiên thì NLP thấp hơn DyAM (−0.016 đến −0.013), nên một phân hoạch riêng lẻ có thể đảo chiều C-index.

### 2.74. Cập nhật bài báo: sống còn OvO+NLP ở BM1 (2026-10-13)

Không chạy huấn luyện mới. Đưa kết quả §2.71–§2.73 (env ref-linux-py38; hai môi trường Windows chỉ làm bằng chứng độ ổn định) vào: `paper/sections/results.tex` (mục 3.4, đoạn mới "OvO attention with the NLP encoding at the primary benchmark": HR/1 SD 1.53/1.58/1.60/1.64, log-rank χ² 12.3/14.4/16.3/17.9, C-index 0.616–0.623; OvO+NLP − DyAM Δlog HR +0.070, 5/5 seed, bootstrap [0.013, 0.106]; không nêu p-value), `paper/sections/discussion.tex` (đoạn Survival, nêu giả thuyết số tham số, chưa kiểm chứng); `paper_1seed/sections/results.tex` + `discussion.tex` (phân hoạch tự nhiên: HR 1.69 vs 1.61, χ² 23.9 vs 17.4, C-index không theo thứ tự đó: 0.633 vs 0.646). Dựng lại `paper/main.pdf` và `paper_1seed/main.pdf` (không tham chiếu thiếu); Word `paper/manuscript_word.docx` đã port thủ công đoạn mới (`build_manuscript_docx.py`). `paper_1seed` chưa xuất Word. Verdict: KEEP.

### 2.75. Chỉ số của notebook gốc (F1, Precision, Recall, Accuracy tại ngưỡng điểm z-score > 0) tính lại (2026-10-13)

**Script:** `experiments/reference_env/notebook_metrics_from_scores.py` (không train lại; công thức = `generate_metric_plot_V2` trong `lung_helpers.py`: dự đoán dương nếu `score > 0`, nhãn 1 = không đáp ứng là lớp dương) · **Kết quả:** `experiments/results/ref/notebook_metrics.json`. Nguồn điểm: `ref/bm1_surv_scores` + `ref/single/scores` + `ref/survival_scores` (env ref-linux-py38), `win_ovo_env`/`win_ovo_orig` (BM1 Windows), `results/survival_scores` (biến thể lâm sàng Windows cũ). Config train như §0.
**BM1, phân hoạch tự nhiên (AUC/F1/Prec/Rec/Acc):** DyAM 0.783/0.837/0.890/0.789/0.769 (notebook gốc: F1 0.8367 — khớp); OvO 0.801/0.833/0.890/0.784/0.765; DyAM+NLP 0.803/0.839/0.881/0.800/0.769; OvO+NLP 0.824/0.824/0.883/0.773/0.753. **BM1, 5 seed (F1 / Acc):** DyAM 0.811/0.730; OvO 0.811/0.733; DyAM+NLP 0.813/0.734; OvO+NLP 0.825/0.750 (Windows doan-env 0.825/0.747; doan-orig 0.824/0.747).
**Verdict:** KEEP làm bằng chứng — ở ngưỡng z-score 0, OvO+NLP có F1/Accuracy cao nhất trong 5 seed ở cả ba môi trường (+0.012 F1, +0.016 Acc so với DyAM), nhưng ở phân hoạch tự nhiên F1/Accuracy của OvO+NLP thấp hơn DyAM (0.824 vs 0.837): các chỉ số theo ngưỡng nhạy phân hoạch hơn AUC.

## 2.76 Survival analysis on all 21 paper configurations, single partition (2026-10-14)

- Question: do C-index / Cox HR / log-rank chi2 / tdAUC of the fused risk score support the OvO(+NLP) vs DyAM comparison on all 21 configurations, not only BM1 (replaces the earlier BM1-only and IHC-A/IHC-G-arm tables in `paper_1seed`).
- Scripts: `experiments/survival/run_survival_21combos_single.py` (3 parts, logs `experiments/results/ref_surv21_p{0,1,2}.log`), `experiments/survival/compute_tdauc_21_single.py` (venv_np2, sksurv), earlier `bm1_single_survival_ref.py` / `compute_tdauc_bm1_single.py`. Raw: `experiments/results/ref/single21/{surv_part*.json,tdauc_21.json,scores/*.csv}`; BM1: `ref/single/{bm1_survival_single,tdauc_bm1_single}.json`.
- Config: env ref-linux-py38; epochs 125, lr 0.01, alpha 0.001, beta 0, cross_modality off, KFold(10, shuffle, random_state=0), seed 42 natural patient order, `ctx.rad_filters` via `make_filter`, discovery n=247; models `train_dyam` / `train_ovo_safe`; +NLP (MiniLM raw 384, no_scale) only run for the two Labs configurations (20, 21; NLP replaces labs), other +NLP entries repeat the non-NLP values. Cox adjusted for age, ECOG, albumin, dNLR, liver mets (lung_helpers `compute_multivariate_cox`, HR per unit risk score); KM at median; tdAUC Uno 6/12/18 mo.
- Check: AUC of every run equals the AUC of `ref_all21_part*.json` single_run (0 mismatches over 46 runs).
- Results (DyAM / OvO): BM1 (18) C .646/.640, HR 5.27/5.83, chi2 17.4/20.3; BM2 (21) C .640/.637, HR 4.63/6.13, chi2 17.6/20.9; with NLP (labs replaced) DyAM+NLP C .630 HR 5.17 chi2 19.2, OvO+NLP C .633 HR 6.64 chi2 23.9, tdAUC12 .733. Config 17 DyAM is strongest overall (C .655, HR 8.59, chi2 33.0) and OvO is lower there (C .620, HR 6.29). Across the 21 configurations OvO is not systematically better than DyAM on survival (OvO higher C-index in 6/21, HR in 9/21, chi2 in 8/21, tdAUC12 in 10/21) - not reported as counts in the paper; association grows with the number of fused sources (C .56-.59 single-source vs .59-.66 for 6-7 sources).
- Verdict: KEEP; OvO+NLP best HR/chi2/tdAUC at BM2 and cfg 20, but no uniform survival advantage of OvO over DyAM.

## 2.77 Survival analysis on all 21 paper configurations, 5 seeds x 10-fold (2026-10-14)

- Question: does the OvO(+NLP) survival advantage seen in one partition (§2.76) hold over 5 seeds on all 21 configurations?
- Scripts: `experiments/survival/run_survival_21combos_5seed.py` (3 parts; logs `experiments/results/ref_surv5_p{0,1,2}.log`), `experiments/survival/compute_tdauc_21_5seed.py` (venv_np2). Raw: `experiments/results/ref/multi21/{surv_part*.json,tdauc_21_5seed.json,scores/*.csv}`.
- Config: env ref-linux-py38; standard 5-seed harness (SEEDS 42,7,123,2024,31337, label permutation), epochs 125, lr 0.01, alpha 0.001, beta 0, KFold(10), `ctx.rad_filters` via `make_filter`, n=247; DyAM `train_dyam`, OvO `train_ovo_safe`; +NLP (MiniLM raw 384, no_scale) only for the Labs configurations 20, 21. Per seed: AUC, Harrell C, log HR per 1 SD (adjusted age, ECOG, albumin, dNLR, liver mets), KM chi2 (median), tdAUC (Uno). Note: configs 9, 10 seed 42 has 21 patients without score (211 scored); tdAUC computed on scored patients.
- Results (mean of 5 seeds, DyAM -> OvO): BM1 (18) C .616->.620, logHR .425->.460, chi2 12.3->14.4, td12 .681->.709; BM2 (21) C .626->.627, logHR .416->.452, td12 .675->.699; with NLP (labs replaced): cfg 21 DyAM+NLP C .622 logHR .468 td12 .685, OvO+NLP C .623 logHR .495 chi2 17.9 td12 .708; cfg 20 DyAM+NLP C .606 logHR .450, OvO+NLP C .617 logHR .497 td12 .704. OvO logHR/td12 above DyAM in most configurations with >=5 sources (14-21) and in configs 8, 5; below in some small ones (7 td, 9-13); C-index differences are within about +-0.01.
- Verdict: KEEP; OvO(+NLP) has the higher hazard ratio / tdAUC in the large configurations, C-index unchanged; direction consistent with the single partition; no claim of uniform superiority.

### 2.78. Cap nhat bai (2026-10-14)
- `paper_1seed` (mot phan hoach): muc 3.4 viet lai theo 21 to hop (Bang 7-8, Hinh 4-5; bien the ma hoa lam sang sang phu luc), nguon §2.76; PDF main + supplementary dung lai, chua xuat Word.
- `paper` (5 seed): them doan "Survival association across the 21 configurations" va Bang 10 (`paper/tables/table_surv21_5seed.tex`, sinh boi `paper/make_surv21_5seed_table.py`, nguon §2.77), da port sang `build_manuscript_docx.py`, PDF + `manuscript_word.docx` dung lai.

### 2.79. `paper/` chuyen sang chi dung ket qua 5 seed (2026-10-14)
- Bang 21 to hop (Bang 3): cot +NLP chi khac o to hop 20, 21 (NLP thay Labs); cac to hop con lai lap lai so khong NLP, khong in dam (`paper/make_results_tables.py`, nguon `ref_all21_part*.json` 5 seed).
- Bo Bang single-partition (tab:ovo) va Hinh/Bang S3-S4 phu luc (mot phan hoach); viet lai muc 3.2 theo tung to hop, khong tong hop dem.
- Bang NLP20 / leave-one-source-out / availability vector: tinh lai tu 5 seed dau (42, 7, 123, 2024, 31337) cua cac file 20 seed (khong train lai; cung seed, cung diem); bo dong "Mean of the six".
- Hinh KM: ve lai o BM1 tu diem ensemble 5 seed (`paper/make_km_bm1_ensemble.py`); chi so chi2 ensemble (DyAM 9.0, OvO 13.4, DyAM+NLP 20.6, OvO+NLP 18.5) khac trung binh tung seed (12.3, 14.4, 16.3, 17.9) o Bang 9.
- Chua doi: hold-out theo thu tu CT (mot lan chia, khong phai seed) va cau kiem tra tai hien 0.798/0.788 (Linux/Windows). PDF + Word da dung lai.

### 2.80. Cap nhat thong tin tac gia va viet lai Discussion/Abstract/Conclusion/Introduction theo giong thuan loi (2026-10-14)
- Khong train them; chi sua van ban `paper/` (5 seed) va `paper_1seed/` (1 phan hoach); moi so lieu lay tu muc Results san co.
- Thong tin tac gia: Vu Huy Hoang, tac gia duy nhat, SOICT - Dai hoc Bach khoa Ha Noi, email/dien thoai, link GitHub; bo Funding, IRB, Informed consent (ca doan dao duc o muc 2.1).
- Discussion: viet lai thanh bon ket luan (hop nhat nguon; NLP > labs o ca 6 phep so sanh; OvO tot hon voi nhieu nguon, 1/4 tham so; tinh thuc tien), giu Limitations, hold-out, uoc luong lac quan.
- Word: lien ket noi bo [n] -> muc tham khao, DOI -> https://doi.org; `paper/` Word nay render Abstract/Discussion/Conclusion tu LaTeX (khong con port tay cac muc nay; bak: `build_manuscript_docx.py.bak2`).
- Sua loi `\times` bi hong thanh TAB trong abstract/conclusion cua `paper/`.
- PDF + Word dung lai, 0 loi.

### 2.81. Kiểm chứng NGOÀI path_valid: DyAM / OvO / DyAM+NLP / OvO+NLP (2026-10-14) · env: ref-linux-py38 · hướng KHÔNG giữ, không xác nhận được
**Câu hỏi:** lợi thế OvO > DyAM và NLP > Labs thô (CV nội bộ, §2.64/§2.65) có giữ ở cohort ngoài không. Kế hoạch ghi trước ở `document/2026-10-04_arch-search-loop/ledger.md` (2026-10-14); chạy MỘT lần.
**Mã / đầu ra:** `experiments/extval_ref/{make_cache,run_extval_ref,predict_new}.py` · `experiments/results/ref/extval_ref.json` (+`.scores.npz`, `extval_ref.log`) · model đã train: `D:/repro_linux/extval_models/*.pt` (42 file), bộ nhớ đệm `D:/repro_linux/extval_cache.pkl`.
**Cấu hình:** env ref-linux-py38; huấn luyện trên TOÀN BỘ discovery n=247, dự đoán path_valid n=71 (21 đáp ứng, 50 không) một lần; epochs 125, lr 0.01, alpha 0.001, beta 0, `cross_modality_enabled` False, `attention_gate_enabled` False cho đơn nguồn; model `MultiModalDynamicModel` (DyAM) / `MultiModalDynamicModelOvO`; không l1_filter (không có radiomics). 1 seed = seed 42 thứ tự tự nhiên; 5 seed = 42,7,123,2024,31337 + hoán vị hàng. CV nội bộ cùng bộ nguồn: KFold(10, shuffle, rs=0).
**Lệch chuẩn:** 6 biến lâm sàng (NLP6/Labs6) thay vì 13; chỉ nguồn có ở ngoài (GLCM 52/71, PD-L1 71/71, lâm sàng 6); không phải 21 tổ hợp; embedding NLP6 tính ngoài ref env (MiniLM) rồi nạp pickle; 3 ô lâm sàng thiếu ở ngoài điền trung vị discovery; rad_valid không chạy.
**AUC ngoài [CI bootstrap bệnh nhân] | 5 seed mean±sd | ensemble 5 seed || CV nội bộ 1 seed | 5 seed:**

| Bộ nguồn | Mô hình | ngoài 1 seed | ngoài 5 seed | ens | CV 1s | CV 5s |
|---|---|---|---|---|---|---|
| A1 GLCM+PD-L1 | DyAM | 0.717 [0.586,0.834] | 0.717±0.000 | 0.717 | 0.738 | 0.715±0.015 |
| A1 | OvO | 0.718 [0.596,0.834] | 0.718±0.000 | 0.719 | 0.759 | 0.725±0.019 |
| A2 +Labs6 thô | DyAM | 0.694 [0.557,0.819] | 0.694±0.000 | 0.694 | 0.700 | 0.692±0.012 |
| A2 | OvO | 0.730 [0.599,0.850] | 0.730±0.000 | 0.730 | 0.725 | 0.703±0.016 |
| A3 +NLP6 | DyAM+NLP | 0.710 [0.577,0.827] | 0.708±0.000 | 0.709 | 0.759 | 0.732±0.021 |
| A3 | OvO+NLP | 0.711 [0.587,0.825] | 0.711±0.001 | 0.710 | 0.748 | 0.712±0.021 |
| P PD-L1 đơn | DyAM | 0.739 [0.617,0.847] | 0.739±0.000 | 0.739 | 0.710 | 0.723±0.017 |

**Δ ghép cặp (bootstrap 5000, ensemble):** OvO−DyAM A1 +0.002, A2 +0.036 [−0.039,+0.111] p=0.34, A3 +0.002; NLP6−Labs6 DyAM +0.014 p=0.75, OvO −0.020 p=0.63; OvO+NLP6−DyAM+Labs6 +0.016 [−0.067,+0.103] p=0.72; NLP6 − không lâm sàng −0.008 (DyAM), −0.008 (OvO). Mọi CI phủ 0.
**Đọc:** (i) không có phương pháp nào vượt PD-L1 đơn nguồn (0.739); thêm lâm sàng (thô hay NLP6) không giúp ở ngoài. (ii) OvO−DyAM ≈ 0 ở A1/A3, +0.036 ở A2 nhưng không có ý nghĩa. (iii) NLP6 vs Labs6: dấu đảo giữa DyAM (+) và OvO (−), không nhất quán. (iv) sd giữa 5 seed ≈ 0 vì khi train trên toàn bộ discovery chỉ còn seed khởi tạo trọng số ảnh hưởng (hoán vị hàng không đổi gì ở full-batch); khác CV nội bộ, nơi seed đổi cách chia fold (sd ≈ 0.015–0.02). Vì vậy "5 seed" ở ngoài gần như lặp lại 1 seed. (v) nội bộ→ngoài sụt 0.01–0.05 ở A1/A3.
**Giới hạn:** NLP6 không phải NLP13 (6 biến này ở discovery chỉ ~0.496, §2.36) nên KHÔNG kiểm chứng phát hiện NLP-clinical; n đáp ứng = 21, CI ≈ ±0.12, chênh +0.01…+0.03 không xác nhận hay bác bỏ được; 19/71 bệnh nhân không có GLCM.
**Verdict:** negative/không kết luận — lợi thế OvO và NLP của CV nội bộ KHÔNG tái hiện ở ngoài (OvO≈DyAM, NLP6≈Labs6, mọi p>0.3). GIỮ mã và model đã lưu để test cohort mới (`predict_new.py`).


### 2.81. Phân tầng fold: có làm giảm dao động AUC giữa seed không? (2026-10-14, chẩn đoán)
- **Câu hỏi:** sd của AUC giữa 20 seed có giảm khi fold được phân tầng theo nhãn, hoặc theo nhãn x mẫu nguồn có mặt (giả thuyết: số nguồn mỗi bệnh nhân khác nhau làm fold nhạy)?
- **Mã / đầu ra:** `experiments/reference_env/run_fold_strat_ref.py <random|label|pattern>`; `experiments/results/ref/foldstrat/{scheme}__{dyam|ovo}.csv`, `{scheme}.log`, `fold_stats.json`.
- **Cấu hình:** env ref-linux-py38; BM1 Rad+IHC-G+Gen+PDL1; DyAM (`allcombo.run_combo.train`) và OvO (`run_ovo_singlerun.train_ovo_safe`); epochs 125, lr 0.01, alpha 0.001, beta 0, cross_modality False, 10 fold, 20 seed (SEEDS20, hoán vị thứ tự bệnh nhân), l1 `ctx.rad_filters`, discovery n=247. LỆCH CHUẨN: bộ chia fold là StratifiedKFold (random_state 0) với hai cách phân tầng; cách `random` là giao thức chuẩn. Ba seed (một trong mỗi cách DyAM) có một fold bị bỏ vì modality hết đặc trưng sau L1 (222-223/247 bệnh nhân có điểm), AUC tính trên bệnh nhân có điểm. Lần chạy đầu crash vì NaN này (không ghi số), đã sửa rồi chạy lại.
- **Độ lệch giữa các fold của thành phần fold (trung bình 20 seed):** tỉ lệ nhãn dương random 0.085, label 0.016, pattern 0.026; số nguồn trung bình random 0.196, label 0.195, pattern 0.127.
- **Kết quả (AUC, 20 seed):**

| cách chia | DyAM mean ± sd (khoảng) | OvO mean ± sd (khoảng) | OvO−DyAM (OvO hơn) |
|---|---|---|---|
| random | 0.7628 ± 0.0157 (0.731–0.784) | 0.7748 ± 0.0178 (0.738–0.800) | +0.0120 (18/20) |
| label | 0.7612 ± 0.0116 (0.745–0.780) | 0.7700 ± 0.0127 (0.740–0.795) | +0.0087 (17/20) |
| pattern | 0.7592 ± 0.0151 (0.725–0.781) | 0.7686 ± 0.0128 (0.742–0.787) | +0.0094 (18/20) |

- **Đọc (theo quy tắc định trước trong ledger):** sd(pattern) không thấp hơn rõ sd(label) (DyAM 0.0151 so với 0.0116; OvO 0.0128 so với 0.0127), nên mẫu nguồn có mặt không phải nguồn dao động chính; phân tầng theo nhãn giảm sd khoảng 0.004-0.005 nhưng phần lớn dao động (sd ≈ 0.012-0.013) vẫn còn, phù hợp với cỡ mẫu nhỏ (62 dương) và lọc L1 mỗi fold. Hướng OvO > DyAM giữ ở cả ba cách chia (+0.009 đến +0.012, 17-18/20 seed). Mean AUC chênh ≤ 0.004 giữa các cách chia.
- **Verdict:** chẩn đoán, không đổi số trong bài; giả thuyết "số nguồn gây dao động" không được ủng hộ (chưa tách khỏi số chiều).

### 2.82. Cap nhat bai: bang song con 5 seed tach thanh 2 bang (2026-10-14)
Khong chay huan luyen moi. `paper/make_surv21_5seed_table.py` (nguon §2.77, env ref-linux-py38) nay sinh hai bang giong `paper_1seed`: Bang 9 (C-index + log HR) va Bang 10 (log-rank chi2 trung binh 5 seed + tdAUC 12 thang), co cot k; bo `resizebox`, `tabcolsep` 2pt. Them cau log-rank vao muc 3.4 (OvO cao hon DyAM o cau hinh 14-19 va BM2, thap hon o cau hinh 20: 20.5 so 14.3). Word `paper/word_export/build_manuscript_docx.py` da cap nhat (Bang 9, 10) va dung lai; `paper_1seed`: giam `tabcolsep` Bang 7-8 cho vua khuon, URL GitHub dung `\url`. Verdict: cap nhat bai, khong doi so lieu.

### 2.83. AUC theo từng fold của phân hoạch một seed: mean ± sd qua 10 fold (2026-10-14)
- **Câu hỏi:** thầy yêu cầu báo mean ± std cho bản 1 seed; mean ± sd của 10 AUC theo fold là bao nhiêu?
- **Script / đầu ra:** `experiments/reference_env/perfold_single_ref.py` -> `experiments/results/ref/single/perfold_auc.json` (50 chuỗi điểm: BM1 4 mô hình + 46 điểm của 21 cấu hình, nguồn `ref/single/scores`, `ref/single21/scores`). KHÔNG huấn luyện lại; fold dựng lại bằng `KFold(10, shuffle=True, random_state=0)` trên thứ tự tự nhiên; đã kiểm bằng `check_fold_recon.py` (chạy lại BM1 DyAM seed 42, cột `fold` thật khớp 100%, AUC 0.7834).
- **Config:** env ref-linux-py38; seed 42, thứ tự tự nhiên, 10 fold, cấu hình chuẩn §0; sd là độ lệch mẫu (ddof 1) của 10 AUC từng fold (mỗi fold ~25 bệnh nhân, ~6 ca đáp ứng).
- **Kết quả (AUC gộp -> mean ± sd 10 fold):** BM1 DyAM 0.7834 -> 0.763 ± 0.119; OvO 0.8005 -> 0.782 ± 0.114; DyAM+NLP 0.8030 -> 0.789 ± 0.078; OvO+NLP 0.8243 -> 0.804 ± 0.088. BM2 DyAM 0.7813 -> 0.766 ± 0.109; OvO 0.7935 -> 0.777 ± 0.102. Cấu hình 17 DyAM 0.7983 -> 0.779 ± 0.072.
- **Verdict:** KEEP làm số bổ sung; sd lớn (0.07-0.12) vì fold nhỏ, không đo cùng thứ với CI DeLong hay sd giữa seed.

### 2.84. Cap nhat Word `paper_1seed`: gop sub-figure thanh 1 hinh (2026-10-15)
Khong chay huan luyen. `paper_1seed/word_export/build_manuscript_docx.py`: moi hinh nhieu panel (Hinh 4 AUC, Hinh 5 KM, Hinh 6 song con) nay la MOT anh ghep (2 cot, chu A-D, `figures/converted/composite_fig{4,5,6}.png`) voi MOT chu thich, thay cho tung panel rieng. PDF khong doi (da la mot hinh voi subfloat). Chua lam cho `paper/` (5 seed). Word dung lai `manuscript_word_1seed.docx`.

### 2.85. Cap nhat hinh `paper_1seed`: tang co chu Hinh 5, 6 (2026-10-15)
Khong chay huan luyen. `paper_1seed/make_figs_single.py`: panel KM (Hinh 5) dung title 13 / nhan 12 / tick 11 / legend 11 / bang risk 10 (truoc 10/10/10/8/7), legend chuyen xuong duoi o chi2 de khong che duong cong; panel song con (Hinh 6) nhan 12, tick 10-11, legend 10 (truoc 7-10), nhan truc y rut gon ("Time-dependent AUC (12 mo)", "Cox HR per unit risk score"). Hinh phu luc S6 (KM) giu co chu cu. PDF (22 trang, 0 loi) va Word `manuscript_word_1seed.docx` dung lai (anh ghep `composite_fig5/6.png`).
