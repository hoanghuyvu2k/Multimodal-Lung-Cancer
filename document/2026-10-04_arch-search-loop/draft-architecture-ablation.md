> **QUYẾT ĐỊNH 2026-10-10:** người dùng chọn tập trung vào OvO và bỏ trọng số đều khỏi nội dung bài (Q1 = B). Các bảng A, C, D, E trong file này đang dùng trọng số đều làm mốc/nền và cần viết lại trên nền OvO trước khi đưa vào bài; xem ghi chú ở `document/2026-10-07_decisions-before-writing/quyet-dinh.md`.

# Bản nháp: phần so sánh kiến trúc và ablation

> Lập 2026-10-04. File này để DUYỆT — chưa có gì được đưa vào `paper/`.
> Mọi con số lấy từ `result/training-results.md` (§2.14, §2.22, §2.24, §2.27–§2.30); nguồn ghi ở từng bảng.
> Phần tiếng Anh là văn bản đề xuất cho bài báo; phần tiếng Việt là ghi chú cho người duyệt.

> **CẬP NHẬT 2026-10-05 (§2.32–§2.33):** thêm chuyên gia Phikon (embedding mô hình nền từ tiêu bản gốc) nâng BM1 lên
> 0.8095 ± 0.010 / ensemble 0.8182. NHƯNG đối chứng cho thấy ~73% mức tăng tái lập được bằng một chuyên gia không có ảnh, chỉ biết
> "bệnh nhân có tiêu bản hay không" (BM1: 0.8087); phần do nội dung ảnh chỉ +0.004 và không có ý nghĩa ở mức bệnh nhân. Vì vậy các
> bảng C, D, E bên dưới được GIỮ NGUYÊN không gồm thành phần này. Nếu muốn dùng con số 0.81, phải trình bày nó là mở rộng của
> availability vector (quy trình chăm sóc), không phải đặc trưng học sâu.

> **CẬP NHẬT 2026-10-06 (§2.40–§2.41):** (i) `miss_pattern`: bỏ nó KHÔNG mất AUC ở các tổ hợp có PD-L1 (BM1 −0.001, p = 0.47) nhưng mất 0.017–0.018 ở hai tổ hợp k = 5 KHÔNG có PD-L1 (20/20 và 19/20 seed);
> theo quy tắc định trước kết luận là GIỮ, báo cáo cả hai phiên bản và nêu rõ điều kiện. Nó vẫn là tín hiệu quy trình chăm sóc (khi 50% bệnh nhân bị ẩn pathology, mô hình có nó mất thêm 0.006, p = 0.019).
> (ii) Mức tăng so với trọng số đều (+0.020) nằm trong dải thổi phồng kỳ vọng do chọn trong ~52 ứng viên (0.011–0.024).

## 0. Ba việc cần bạn quyết trước khi đưa vào bài

1. **Backbone trình bày: OvO hay uniform averaging?** Số liệu cho thấy hai cái ngang nhau (uniform nhỉnh hơn
   0.0015 mỗi seed, không có ý nghĩa ở mức bệnh nhân). Bản nháp dưới đây viết theo phương án trung thực nhất:
   coi quy tắc hợp nhất là một trục ablation và báo cáo cả ba (DyAM, OvO, uniform). Nếu bạn muốn giữ OvO làm
   mô hình đề xuất thì đổi được, nhưng không viết được câu "OvO cho AUC cao hơn" — chỉ viết được "ổn định hơn".
2. **Có dùng `miss_pattern` trong mô hình chính không?** Nó đóng góp thật, nhưng là thông tin về quy trình
   chăm sóc chứ không phải sinh học khối u, và nhiều khả năng không khái quát sang cohort khác. Bản nháp giữ
   nó trong bảng ablation và nêu rõ ở Limitations.
3. **Con số tiêu đề.** Đề xuất: **0.796 ± 0.011 (một mô hình, 20 seed)**, kèm 0.804 cho ensemble. Không
   dùng các số 5-seed làm tiêu đề vì 5 seed chuẩn lạc quan khoảng 0.005 (§2.22).

## 1. Những phát biểu số liệu ủng hộ được — và không ủng hộ được

| Phát biểu | Ủng hộ? | Căn cứ |
|---|---|---|
| Mô hình đầy đủ hơn DyAM gốc ở mọi tổ hợp nhiều nguồn, ở mọi phân hoạch | Có | 8/8 tổ hợp k ≥ 5, 17/21 nói chung (§2.31); +0.035 mỗi seed, 20/20 seed trên 4 tổ hợp (§2.27) |
| Mô hình đầy đủ hơn DyAM ở MỌI tổ hợp | **Không** | thua ở chỉ dấu đơn lẻ: TMB, PDL1, Gen, Rad (§2.31) |
| Mức hơn đó có ý nghĩa thống kê ở mức bệnh nhân | **Không** | Δ ensemble gộp +0.023, CI [−0.009, +0.054], p = 0.16 |
| Huấn luyện chung giúp từng nguồn học tốt hơn huấn luyện riêng | Có | chuyên gia độc lập thua 15/15 lượt, −0.013 (§2.28) |
| OvO ổn định hơn DyAM, lợi thế tăng theo số nguồn | Có | sd 0.0157 vs 0.0177; k = 7: 19/20 seed (§2.24) |
| OvO chính xác hơn DyAM trên bệnh nhân mới | **Không** | Δ gộp +0.002, p = 0.57 |
| Attention học được cho kết quả tốt hơn trọng số đều | **Không** | uniform ≥ OvO ≥ DyAM; 0/41 biến thể kiến trúc thắng |
| Mã hoá lâm sàng bằng NLP cải thiện hợp nhất | Có | +0.010 ở BM1 so với không có lâm sàng (§2.14); hơn Labs thô ở 19/21 tổ hợp (5 seed) |
| Nhiệm vụ phụ trên pathology giúp | Yếu | +0.012 ở tổ hợp IHC-A, p = 0.07; ~70% là điều chuẩn chung |

## 2. Văn bản đề xuất (tiếng Anh)

### 2.1 Fusion-rule comparison

> To isolate the contribution of the fusion mechanism, we compared three rules for combining the
> modality-specific risk scores under an identical protocol: the original cooperative attention (DyAM),
> the competitive one-versus-others attention (OvO), and unweighted averaging over the available modalities.
> All three share the same modality-specific linear experts, the same feature filtering and the same
> cross-validation partitions; only the weighting of experts differs. Each configuration was trained with
> 20 random partitions of 10-fold cross-validation on the six combinations that include five or more data
> sources.
>
> Averaged over the six combinations, single-model AUC was 0.743 ± 0.018 for DyAM, 0.749 ± 0.016 for OvO
> and 0.751 ± 0.015 for unweighted averaging (Table A). OvO exceeded DyAM in 14 of 20 partitions (mean
> difference +0.006, Wilcoxon p = 0.008), and the advantage grew with the number of sources, reaching
> +0.018 (19 of 20 partitions) for the seven-source combination. OvO was also less sensitive to the
> partition (between-partition s.d. 0.016 versus 0.018). However, when the uncertainty due to patient
> sampling was taken into account, the pooled difference between OvO and DyAM was not significant
> (+0.002, 95% CI −0.005 to +0.010, paired bootstrap p = 0.57), and neither attention mechanism
> outperformed unweighted averaging. We therefore interpret the benefit of competitive attention as
> improved stability rather than improved discrimination in a cohort of this size.

**Table A. Fusion rule, six multi-source combinations, 20 partitions** (nguồn: §2.24)

| Combination | k | DyAM | OvO | Uniform |
|---|---|---|---|---|
| Rad+IHC-A+Gen | 5 | 0.728 ± 0.021 | 0.730 ± 0.017 | 0.729 ± 0.017 |
| Rad+IHC-G+Gen | 5 | 0.754 ± 0.016 | 0.746 ± 0.016 | 0.746 ± 0.015 |
| Rad+IHC-A+MutAmp | 6 | 0.730 ± 0.020 | 0.741 ± 0.015 | 0.744 ± 0.016 |
| Rad+IHC-A+Gen+PDL1 | 6 | 0.741 ± 0.015 | 0.749 ± 0.016 | 0.751 ± 0.016 |
| Rad+IHC-G+Gen+PDL1 | 6 | 0.760 ± 0.019 | 0.767 ± 0.015 | 0.769 ± 0.014 |
| Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.744 ± 0.016 | 0.762 ± 0.014 | 0.766 ± 0.015 |
| **Mean** | | 0.743 | 0.749 | 0.751 |

### 2.2 Joint training benefits the individual experts

> A recurring concern in multimodal learning is that jointly trained models under-optimise the weaker
> modalities. We tested this directly by comparing each expert's discrimination inside the jointly trained
> model with that of the same expert trained alone on the same partitions. No modality was suppressed by
> joint training (differences between −0.024 and −0.001 for radiomics, genomics and PD-L1), and the
> pathology expert improved markedly (AUC 0.680 jointly versus 0.617 alone). Consistently, a variant in
> which experts were trained independently and only combined at inference was worse than joint training in
> every one of 15 runs (mean −0.013). Methods designed to rebalance modalities during training (gradient
> modulation, auxiliary unimodal supervision, alternating updates) had no measurable effect (|Δ| < 0.002).
> Combining sources therefore does more than pool information: optimising the experts against a shared
> objective regularises the high-dimensional ones.

**Table B. Expert AUC inside the joint model vs trained alone** (BM1, 3 partitions; nguồn: §2.28)

| Expert | n | Joint | Alone | Δ |
|---|---|---|---|---|
| Radiomics, lung parenchyma | 163 | 0.631 | 0.639 | −0.007 |
| Radiomics, pleura | 21 | 0.553 | 0.556 | −0.003 |
| Radiomics, lymph node | 67 | 0.647 | 0.671 | −0.024 |
| Pathology (GLCM) | 105 | 0.680 | 0.617 | **+0.062** |
| Genomics | 247 | 0.666 | 0.674 | −0.008 |
| PD-L1 | 201 | 0.714 | 0.715 | −0.001 |

### 2.3 Ablation of the final model

> Starting from the published DyAM configuration on the best-performing combination
> (radiomics + pathology + genomics + PD-L1), we added components one at a time (Table C). Replacing learned
> attention by unweighted averaging changed single-model AUC from 0.760 to 0.769. Encoding the thirteen
> clinical variables as a sentence embedding raised it to 0.784, and adding a binary vector describing
> which modalities were available to 0.788. Using the PD-L1 score from the clinical pathology report for the 45 patients
> without a re-scored slide, which increases the coverage of this modality from 201 to 246 of 247 patients,
> raised it to 0.796 ± 0.011. Averaging the predictions of models trained on different partitions gave 0.804;
> this gain saturates at about ten models.
>
> The full model exceeded the original DyAM in all 20 partitions on each of the four combinations evaluated
> (mean single-model difference +0.035). With 247 patients this difference did not reach significance at the
> patient level (pooled ensemble difference +0.023, 95% CI −0.009 to +0.054, paired bootstrap p = 0.16),
> in line with the sample-size estimate in Supplementary Note S3.

**Table C. Ablation, Rad+IHC-G+Gen+PDL1, 20 partitions** (nguồn: §2.22, §2.24, §2.27, §2.31)

| Configuration | Single model | Ensemble of 20 |
|---|---|---|
| DyAM (original) | 0.760 ± 0.019 | 0.780 |
| Unweighted averaging | 0.769 ± 0.014 | 0.784 |
| + sentence-embedded clinical variables | 0.784 ± 0.013 | 0.795 |
| + availability vector | 0.788 ± 0.011 | 0.798 |
| + clinical PD-L1 score where the re-scored slide is missing | **0.796 ± 0.011** | **0.804** |

**Table D. Full model vs DyAM, four combinations, 20 partitions** (nguồn: §2.27)

| Combination | DyAM | Full model | Ensemble: DyAM / full |
|---|---|---|---|
| Rad+IHC-A+MutAmp | 0.730 | 0.760 ± 0.010 | 0.747 / 0.768 |
| Rad+IHC-A+Gen+PDL1 | 0.741 | 0.775 ± 0.011 | 0.765 / 0.784 |
| Rad+IHC-G+Gen+PDL1 | 0.760 | 0.796 ± 0.011 | 0.780 / 0.804 |
| Rad+IHC-A+Gen+TMB+PDL1 | 0.744 | 0.785 ± 0.009 | 0.765 / 0.792 |

### 2.6 All 21 combinations (đặt ngay sau 2.3 khi đưa vào bài)

> Across all 21 modality combinations of the original study (five partitions each), the full model exceeded
> DyAM in 17. The advantage depended on the number of sources: it was present in all eight combinations with
> five or more sources (mean +0.035, range +0.029 to +0.040) and in eight of nine combinations with two to
> four sources (mean +0.032), but not for single biomarkers, where adding the clinical embedding and the
> availability vector diluted an already compact predictor (TMB −0.006, genomics −0.010) or changed the
> evaluated population (PD-L1). The model is therefore an improvement specifically for multi-source
> integration, which is the setting it was designed for.

**Table E. Full model vs DyAM, 21 combinations, 5 partitions** (nguồn: §2.31). Cột n là số bệnh nhân DyAM chấm điểm; mô hình
đầy đủ luôn chấm cả 247, nên ở các dòng n < 247 hai AUC không cùng tập bệnh nhân.

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

### 2.7 Leave-one-source-out, secondary endpoints, robustness (bổ sung 2026-10-06; §2.37–§2.39)

> To quantify each source's contribution within the final model, we removed one source at a time (20 partitions, Rad+IHC-G+Gen+PDL1). Removing genomics reduced single-model AUC by 0.029
> (in all 20 partitions), radiomics by 0.013, PD-L1 by 0.014 and pathology by 0.011 when its availability indicator was retained but by 0.033 when the indicator was removed as well, indicating that about
> two thirds of the pathology contribution reflects whether a slide existed rather than its content. Removing the sentence-encoded clinical variables reduced AUC by only 0.003 and removing the
> availability vector by 0.001, because availability is already encoded by averaging over the experts present. The advantage over DyAM in response AUC did not carry over to progression-free survival
> (C-index 0.62 for all models) and was small for a durable-clinical-benefit endpoint. Replacing the sentence encoder (mpnet, bge-small) or averaging four sentence templates changed AUC by less than 0.002.
> Patient-level 95% confidence intervals of the ensemble AUC were 0.740–0.861 for the final model and 0.711–0.843 for DyAM (overlapping), and a genomics-only check in 307 external MSK-IMPACT patients
> gave an overall-survival C-index of 0.54 (95% CI 0.49–0.58).

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

### 2.4 Negative results (đề xuất đưa vào Supplementary)

> We evaluated 41 further modifications of the fusion architecture or the training procedure against
> unweighted averaging under a pre-specified protocol (screening on three combinations and five partitions;
> confirmation on six combinations and 20 partitions with patient-level paired tests). None met the
> acceptance criteria. They included alternative fusion operators (median, sum, fusion before the
> non-linearity, score standardisation, domain-balanced hierarchical fusion), non-linear experts,
> gated and reliability-weighted fusion, patient-similarity graphs and graph neural networks, multi-task
> learning with survival, cooperative learning, mixup, feature dropout, self-distillation, modality
> rebalancing, and gradient-boosted trees. The most harmful variants were those that changed how the
> saturated expert scores are combined (fusion before the non-linearity −0.034, score standardisation
> −0.035, sum −0.037, median −0.099). The only architectural change with a consistent positive
> effect was an auxiliary task in which a small hidden layer of the pathology expert also predicts the
> clinical PD-L1 score; it improved the four combinations using low-dimensional pathology features by
> +0.012 (19 of 20 partitions; pooled patient-level p = 0.07) but about 70% of this gain was reproduced
> with a randomly permuted auxiliary target, indicating a regularisation effect rather than transfer of
> biological information. Five further baseline variables (sex, smoking status, histological subtype,
> number of lesions, biopsy site) did not improve the model either as an additional expert (−0.015) or
> when appended to the clinical sentence (+0.000).

### 2.5 Limitations (câu cần thêm)

> The availability vector encodes which examinations a patient received. Availability alone was predictive
> (a single indicator of whether a pathology slide existed reached an AUC of 0.67), which reflects the care
> pathway rather than tumour biology and is unlikely to transfer to cohorts assembled differently.
> Reported improvements over the original model are consistent across partitions but are not statistically
> significant at the patient level, and the comparison involved many candidate models evaluated on the same
> 247 patients. CT acquisition parameters and accession order were also associated with response
> (univariable AUC up to 0.61), suggesting temporal or technical drift within the cohort; these variables
> were deliberately not used as predictors.

## 3. Ghi chú cho người duyệt

- **Tên gọi trong bảng** đang dùng nhãn của bài (Rad, IHC-A, IHC-G, Gen, MutAmp, TMB, PDL1). "Partition" = seed.
- **Bảng C** đã tách riêng bước NLP và bước availability vector ở 20 seed (§2.31): NLP +0.015, availability +0.004.
- **Bảng E (mục 2.6)** là mô hình đầy đủ trên cả 21 tổ hợp, 5 seed — đáp ứng "Full evaluation rule". Chưa có kiểm định mức bệnh
  nhân cho 21 tổ hợp; chỉ có cho 4 tổ hợp ở Bảng D.
- **Mâu thuẫn với bài hiện tại:** bài đang trình bày OvO là cải tiến về độ chính xác. Mục 2.1 thay bằng
  phát biểu về độ ổn định. Điều này liên quan tới SK-1…SK-6 trong
  `document/2026-09-29_verify-paper-numbers/sai-khac-can-quyet-dinh.md`, vẫn đang chờ bạn quyết.
- **Số 0.760 của DyAM ở BM1** là số 20 seed chạy lại trong môi trường hiện tại; Bảng 4 của bài ghi số khác
  (5 seed, scipy cũ). Nếu dùng bản nháp thì toàn bộ bảng so sánh nên chuyển sang cùng một bộ 20 seed.
