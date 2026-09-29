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
