# Tài liệu Dự án: MultiModalDynamicModel (DyAM)

> **Tóm tắt:** Hệ thống dự đoán đáp ứng điều trị ung thư phổi đa phương thức, kết hợp Radiomics (CT), Genomics (Gen), Pathology (Mô bệnh học) và Clinical (Lâm sàng). Mô hình trung tâm là `MultiModalDynamicModel` và biến thể `OvO`.

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Dữ liệu sử dụng](#2-dữ-liệu-sử-dụng)
3. [Kiến trúc mô hình MultiModalDynamicModel](#3-kiến-trúc-mô-hình-multimodaldynamicmodel)
4. [AttentionMatrix – Cơ chế Attention gốc](#4-attentionmatrix--cơ-chế-attention-gốc)
5. [Biến thể OvO (One-Versus-Others)](#5-biến-thể-ovo-one-versus-others)
6. [Quy trình huấn luyện](#6-quy-trình-huấn-luyện)
7. [Cải tiến đề xuất](#7-cải-tiến-đề-xuất)
8. [Tài nguyên học tập](#8-tài-nguyên-học-tập)

---

## 1. Tổng quan dự án

### 1.1. Bài toán

Phân loại nhãn 0/1 trên bệnh nhân ung thư phổi:
- `SD`, `POD`, ... → `label = 1`
- `PR`, `CR` → `label = 0`

Đánh giá bằng AUC/ROC và khoảng tin cậy 95% (phương pháp DeLong).

### 1.2. Dữ liệu đầu vào

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet` | ~3996×1690 | Radiomics CT phổi |
| `genomic_data_v3.parquet` | 247×11 | Đặc trưng gen |
| `lung_pathology_pdl1_glcm_v3.parquet` | 105×150 | Mô bệnh học PD-L1 + GLCM |
| `pdl1_score.parquet` | 201×1 | PD-L1 TPS score |
| `final_cohort_listing.csv` | 368×2 | Danh sách bệnh nhân |

### 1.3. Các Modality và ý nghĩa y học

| Tên modality | Nhóm | Ý nghĩa y học |
|---|---|---|
| `rad_lesion_pc` | Radiomics – khối u nguyên phát | Hình dạng, kích thước, texture của khối u chính trên CT |
| `rad_lesion_pl` | Radiomics – màng phổi | Tổn thương/thâm nhiễm màng phổi; tiên lượng xấu |
| `rad_lesion_ln` | Radiomics – hạch lympho | Kích thước, texture hạch trung thất – đánh giá di căn hạch |
| `rad_lesion_lu` | Radiomics – lesion tổng hợp | Đại diện lesion nổi bật nhất (volume lớn nhất) |
| `path_ihc_pdl1` | Pathology – IHC PD-L1 (IHC-A) | Mức biểu hiện PD-L1 tại mô u – biomarker chính cho thuốc ức chế PD-1 |
| `path_ihc_glcm` | Pathology – texture IHC-G | Đặc trưng GLCM của tín hiệu PD-L1 – pattern vi mô phức tạp |
| `gen_driver_tmb` | Genomics – TMB | Tumor Mutational Burden – TMB cao → đáp ứng miễn dịch tốt hơn |
| `gen_driver_mut_amp` | Genomics – driver mutations | EGFR, ALK, KRAS – phân tầng nhóm oncogene-driven |
| `cnl_pdl1_score` | Clinical – PD-L1 TPS | Tỉ lệ % tế bào u bắt màu PD-L1 (0–100%) |
| `cnl_dem_labs` | Clinical – demographics & labs | Tuổi, pack_years, ECOG, albumin, dNLR, di căn, therapy line, ... |

### 1.4. Kết quả tốt nhất (từ bài báo)

| Cấu hình | AUC | 95% CI |
|---------|-----|--------|
| DyAM Rad+Gen | 0.759 | [0.693, 0.826] |
| DyAM Rad+IHC-A+Gen+PDL1 | **0.798** | [0.735, 0.861] |
| **OvO** Rad+IHC-G+Gen+PDL1 | **0.8003** | [0.739, 0.862] |

> Ghi chú nguồn số: hai dòng DyAM (0.759, 0.798) là output lưu sẵn trong `Figures-Finalized.ipynb` và `figures-finalized.html` của tác giả (hai file này cho cùng số; bản phát hành Synapse 2022-08-18). Chạy lại `lung_helpers.py` nguyên bản của nhánh `main` (trùng MD5 với bản trên Synapse) cùng dữ liệu gốc (trùng MD5) trong môi trường ghim của bài gốc (Python 3.9.13, torch 1.11.0, scikit-learn 1.1.1, scipy 1.8.1, numpy 1.22.4) trên Windows cho 0.738 và 0.764, nhưng trên Linux (Python 3.8.15, torch 1.8.1, scikit-learn 0.24.2, numpy 1.20.3, scipy 1.6.3, pandas 1.2.5) cho 0.759 và 0.798, khớp output lưu của tác giả (lệch tối đa một cặp bệnh nhân, 0.000087). Hệ điều hành là biến số quyết định; `requirements.txt` ghi sai scikit-learn (1.1.1, thực tế 0.24.2). Dòng OvO là kết quả của dự án này. Chi tiết: `document/2026-10-12_original-environment/original-environment.md`.

---

## 2. Dữ liệu sử dụng

### 2.1. Radiomics CT phổi

File Excel/Parquet chứa **1690 cột**: 2 cột metadata + 1688 cột features.

**Metadata:**
- `job_tag`: Loại radiomics (`filtered-radiomics` hoặc `pertubation-radiomics`)
- `lesion_index`: Chỉ số tổn thương của bệnh nhân

**Cấu trúc tên cột:** `[filter_type]_[feature_class]_[feature_name]`

Ví dụ: `wavelet-LLH_glcm_Contrast` = Wavelet Low-Low-High → GLCM → Contrast

**Các loại filter (18 loại):** `original`, `exponential`, `gradient`, `lbp-2D`, `lbp-3D-m1/m2/k`, `logarithm`, `square`, `squareroot`, `wavelet-LLL/LLH/LHL/.../HHH`

**Các lớp đặc trưng (7 lớp):**

| Lớp | Mô tả |
|-----|-------|
| Shape | Hình dạng 3D tổn thương (chỉ trong `original`) |
| First Order | Thống kê phân phối cường độ pixel |
| GLCM | Gray Level Co-occurrence Matrix |
| GLDM | Gray Level Dependence Matrix |
| GLRLM | Gray Level Run Length Matrix |
| GLSZM | Gray Level Size Zone Matrix |
| NGTDM | Neighboring Gray Tone Difference Matrix |

### 2.2. Quy trình tạo file Radiomics

```
CT DICOM → Resample (1.0mm) → Window/Level (1350/250) → Bin Width (20)
         → Perturbation → Feature Extraction (18 filters × 7 classes)
         → Feature Selection:
             Step 1: Loại bỏ features không ổn định (inter-lesion variance > 0.15)
             Step 2: Loại bỏ features có nhiều outlier (cutoff = 6×)
             Step 3: L1 Elastic Net (l1_ratio=0.5, C=0.1)
         → Reduced Feature Set (~45 features)
```

---

## 3. Kiến trúc mô hình MultiModalDynamicModel

### 3.1. Tổng quan

`MultiModalDynamicModel` là mô hình PyTorch cho phân loại nhị phân đa phương thức. Mô hình:
- Học trọng số rủi ro riêng cho từng modality
- Học attention weights để tổng hợp các modality
- Hỗ trợ modality thiếu qua `mask`

### 3.2. Các tham số khởi tạo

```python
model = MultiModalDynamicModel(
    epochs=100,                    # Số epoch
    alpha=1.0,                     # Hệ số L2 regularization
    beta=1.0,                      # Hệ số phạt attention norm
    lr=0.01,                       # Learning rate Adam
    hidden_factor=2,               # Chưa dùng (legacy)
    class_weight='balanced',       # Gợi ý cân bằng lớp
    cross_modality_enabled=False,  # Attention chéo giữa modality
    no_scale=[],                   # Danh sách modality không chuẩn hóa
    attention_gate_enabled=True    # Bật nhân attention khi tổng hợp
)
```

### 3.3. Luồng dữ liệu tổng quát

```
Input: [X₁, X₂, ..., Xₙ]  +  Mask: [batch, n_modalities]
         ↓
   RobustScaler (chuẩn hóa độc lập từng modality)
         ↓
   AttentionMatrix.forward()
   ┌─────────────────────────────────────────────┐
   │ Nhánh 1: Risk Scores                        │
   │   risk_i = Linear_i(X_i)                   │
   │   risk_weights = tanh(risk_i) × mask        │
   │                                             │
   │ Nhánh 2: Attention Weights                  │
   │   attn_ij = Linear_ij(X_i) / n_feat_i      │
   │   attn_matrix = softplus(attn_ij) × mask    │
   │   attn_scores = sum(attn_matrix, dim=1)     │
   │   attn_weight = L1_normalize(attn_scores)   │
   │                                             │
   │ Kết hợp:                                    │
   │   total_risk = Σ(risk_weights × attn_weight)│
   └─────────────────────────────────────────────┘
         ↓
   response_zscore → Output: score đã chuẩn hóa
```

### 3.4. Chuẩn hóa dữ liệu (RobustScaler)

**Công thức:** `X_scaled = (X - median) / IQR`

**Lý do dùng RobustScaler thay vì StandardScaler:**
- Dữ liệu y học có nhiều outlier (khối u rất lớn/nhỏ)
- Median và IQR không bị ảnh hưởng bởi outlier
- StandardScaler dùng mean/std sẽ bị kéo lệch bởi giá trị cực đoan

Các modality trong `no_scale` (ví dụ: NLP embeddings đã normalized) sẽ bỏ qua bước này.

### 3.5. Loss function

```
Total Loss = (BCE_loss + α·L2_reg + β·attn_norm) / n_batch
```

- **BCE_loss**: `BCEWithLogitsLoss` với `pos_weight = sum(negative) / sum(positive)` – tự động cân bằng lớp
- **L2_reg**: `Σ||W||₂` – tổng chuẩn L2 của tất cả trọng số
- **attn_norm**: `||attn_matrix||₂` – khuyến khích attention phân tán

### 3.6. Response Z-score Normalization

Sau huấn luyện, output được chuẩn hóa:

```python
# Training: tìm ngưỡng tối ưu trên ROC curve
threshold = find_optimal_cutoff(target, input)  # max(TPR - FPR)
output = (input - threshold) / std(input)

# Inference: dùng threshold và std từ training
output = (input - mu) / std
```

Score > 0 → dự đoán SD/PD (xấu); Score < 0 → dự đoán PR/CR (tốt).

### 3.7. Dự đoán và phân tích

```python
# Dự đoán điểm
scores = model.predict_proba(valid_inputs, valid_mask)

# Phân tích chi tiết
score, risks, attentions, shares = model.get_summary_scores(inputs, mask)
# risks: [batch, n_modalities] – risk score từng modality
# attentions: [batch, n_modalities] – attention weights (tổng = 1)
# shares: attention × n_modalities_available
```

### 3.8. So sánh với các phương pháp khác

| Đặc điểm | MultiModalDynamicModel | Logistic Regression | Random Forest |
|----------|------------------------|---------------------|---------------|
| Xử lý multimodal | ✅ Tự nhiên | ❌ Phải concat | ✅ Có |
| Missing modality | ✅ Tự động (mask) | ❌ Phải impute | ⚠️ Một phần |
| Attention | ✅ Có | ❌ Không | ❌ Không |
| Interpretability | ✅ Cao (risk, attention, share) | ⚠️ Trung bình | ⚠️ Trung bình |

---

## 4. AttentionMatrix – Cơ chế Attention gốc

### 4.1. Kiến trúc lớp

**Lớp Risk Linear (`l_risk_linears`):** N lớp `Linear(n_features_i, 1)` – mỗi modality một lớp riêng.

**Lớp Attention Linear (`l_attn_linears`):** N×N lớp – mỗi cặp modality (i→j) một lớp.

### 4.2. Forward Pass chi tiết

**Bước 1 – Tính Risk Scores:**
```
M_i → Linear_Risk_i → R_i [batch, 1]
risk_weights = tanh(cat(R_i)) × linear_mask   [batch, N]
```

**Bước 2 – Tính Attention Matrix:**
```
M_i → Linear_Attn_ij / n_features_i → A_ij [batch, 1]
attn_matrix = softplus(reshape(A_ij))         [batch, N, N]
attn_matrix = attn_matrix × matrix_mask
  (cross_modality_enabled=False → chỉ giữ đường chéo)
```

**Bước 3 – Tổng hợp Attention Weights:**
```
attn_scores = linear_mask × attn_matrix.sum(dim=1)   [batch, N]
attn_weight = L1_normalize(attn_scores)               [batch, N]  (tổng = 1)
```

**Bước 4 – Tổng hợp Risk cuối:**
```
total_risk = Σ(risk_weights × attn_weight)  [batch]
```

### 4.3. Ví dụ với số liệu cụ thể (3 modality, 1 bệnh nhân, Radiomics thiếu)

**Input:** mask = [1, 1, 0] (Clinical, Pathology có; Radiomics thiếu)

**Nhánh Risk:**
- tanh([0.8, -0.5, 1.2]) × [1,1,0] = **[0.66, -0.46, 0.0]**

**Nhánh Attention:**
```
attn_matrix sau mask:
 [[2.1, 1.5, 0.0],
  [1.8, 2.5, 0.0],
  [0.0, 0.0, 0.0]]

attn_scores (sum dim=1) = [3.9, 4.0, 0.0]
attn_weight (L1 norm)   = [0.49, 0.51, 0.0]
```

**Total Risk:**
```
total = 0.66×0.49 + (-0.46)×0.51 + 0.0×0.0 = 0.3234 - 0.2346 = 0.0888
```

### 4.4. Tensor shapes tóm tắt

| Tensor | Shape | Mô tả |
|--------|-------|-------|
| `X_i` | `[batch, n_features_i]` | Dữ liệu modality i |
| `mask` | `[batch, N]` | Mask modality |
| `risk_weights` | `[batch, N]` | Risk sau tanh + mask |
| `attn_matrix` (nội bộ) | `[batch, N, N]` | Ma trận attention 3D |
| `attn_weight` (mixing_matrix) | `[batch, N]` | Attention weights chuẩn hóa |
| `total_risk` (output) | `[batch]` | Risk tổng hợp |

---

## 5. Biến thể OvO (One-Versus-Others)

### 5.1. Ý tưởng cốt lõi

Thay vì attention "hợp tác" (mỗi modality chia sẻ miếng bánh 100%), OvO dùng attention **"cạnh tranh"**: mỗi modality phải chứng minh vượt trội hơn trung bình của các modality khác.

**Công thức OvO:**
```
score_i = Linear_i(X_i) / n_features_i
mean_others_i = mean([score_j | j ≠ i])
ovo_i = sigmoid(score_i - mean_others_i)
attn_weight = L1_normalize([ovo_1, ..., ovo_n] × mask)
```

**Lý do chọn sigmoid thay vì softmax:**
- Mỗi `ovo_i` được tính **độc lập** – phù hợp ý nghĩa OvO
- Sigmoid = "Xác suất modality i tốt hơn các modality khác"
- Softmax tạo xác suất **tương đối** (phụ thuộc lẫn nhau), làm mất tính OvO
- Gradient ổn định hơn trong khoảng (0, 1)

### 5.2. So sánh Attention gốc vs OvO

| Đặc điểm | Attention Matrix (gốc) | AttentionMatrixOvO |
|----------|------------------------|---------------------|
| Cơ chế | Hợp tác | Cạnh tranh |
| Công thức | `a_i = score_i / Σ score_j` | `a_i = sigmoid(score_i - mean_others)` |
| Attention layers | N×N | N |
| Cross-modality | Có (tùy chọn) | Không cần |
| Giải thích | Tầm quan trọng tuyệt đối | Sự vượt trội tương đối |

### 5.3. Ví dụ so sánh (3 modality, scores thô = [0.9, 0.6, 0.2], Radiomics thiếu)

**Attention gốc:**
```
attn_scores (masked) = [0.9, 0.6, 0.0]
attn_weight = [60.0%, 40.0%, 0.0%]
```

**OvO:**
```
ovo_0 = sigmoid(0.9 - (0.6+0.2)/2) = sigmoid(0.5) ≈ 0.622
ovo_1 = sigmoid(0.6 - (0.9+0.2)/2) = sigmoid(0.05) ≈ 0.512
ovo_2 = 0 (thiếu)
attn_weight = [54.8%, 45.2%, 0.0%]
```

OvO "trừng phạt" modality dẫn đầu một chút và "khen thưởng" modality trung bình → phân bổ công bằng hơn.

### 5.4. Kết quả thực nghiệm (20 test cases, 10-fold CV)

| Metric | Giá trị |
|--------|---------|
| OvO tốt hơn | 7/20 cases (35%) |
| Original tốt hơn | 9/20 cases (45%) |
| Bằng nhau | 4/20 cases (20%) |
| Cải thiện tốt nhất (OvO) | +3.27% (PDL1+Gen) |
| Giảm nhiều nhất (OvO) | -4.94% (Rad+IHC-A+Gen) |

**Nhận xét:**
- OvO phù hợp với cấu hình **2-3 modality** (cạnh tranh giúp chọn lọc tốt)
- Original phù hợp với cấu hình **3+ modality phức tạp** (hợp tác bổ sung thông tin)

### 5.5. Hướng dẫn sử dụng

```python
# Mô hình gốc
from lung_helpers import train
summary_df, coef_df = train(
    modality_list_in=data, modality_mask=mask,
    outcomes=labels, l1_dfs_filter=dfs_rad_filters,
    model_params=model_params, folds=10
)

# Mô hình OvO
from lung_helpers import train_ovo
summary_df_ovo, coef_df_ovo = train_ovo(
    modality_list_in=data, modality_mask=mask,
    outcomes=labels, l1_dfs_filter=dfs_rad_filters,
    model_params=model_params, folds=10
)
```

**Lưu ý:** Công thức `sigmoid(score_i - mean_others)` là **custom design** dựa trên ý tưởng OvO, không được lấy từ một bài báo cụ thể. Nếu cần academic citation, cần tham chiếu:
- OvO classification: Hastie & Tibshirani (1998)
- Attention: Vaswani et al. (2017)

---

## 6. Quy trình huấn luyện

### 6.1. Hàm `train()` – KFold Cross-Validation

```python
def train(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params, folds=10):
```

**Tham số:**
- `modality_list_in`: List DataFrame, mỗi cái một modality
- `modality_mask`: DataFrame boolean [n_patients, n_modalities]
- `outcomes`: DataFrame với cột `'label'` (0/1)
- `l1_dfs_filter`: Dict `{pos: {'l1_selection_df': df, 'kwargs': {...}}}`
- `model_params`: Dict tham số cho `MultiModalDynamicModel`
- `folds`: Số fold hoặc `'LOO'` (Leave-One-Out)

**Luồng xử lý từng fold:**
1. Copy data (deep copy để bảo vệ dữ liệu gốc)
2. Áp dụng L1 Feature Selection (chỉ trên training set – tránh data leakage)
3. Kiểm tra modality rỗng sau filter → skip nếu có
4. Huấn luyện `MultiModalDynamicModel`
5. Dự đoán trên validation set
6. Thu thập risk, attention, share scores cho từng bệnh nhân

**Giá trị trả về:**
- `summary_df`: DataFrame với columns `label, score, fold, risk_*, attn_*, share_*, score_norm, error`
- `df_coef_agg`: Hệ số mô hình tổng hợp từ tất cả folds

#### Leave-One-Out (LOO)

Khi `folds='LOO'`, số fold = số bệnh nhân. Mỗi fold để lại 1 bệnh nhân làm test.

**Nên dùng khi:** dataset nhỏ (<50-100 mẫu), cần đánh giá chính xác nhất, nghiên cứu y học với bệnh nhân ít.

### 6.2. Hàm `train_subsample()` – ShuffleSplit

Tương tự `train()` nhưng dùng `ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)`.

**Khác biệt chính:**
- 20 lần chia ngẫu nhiên (cố định, không do `folds` kiểm soát)
- Các lần chia có thể overlap → AUC có thể cao hơn một chút
- Trả về `list` các tuple `(auc, ci, valid_scores, valid_labels)`

```python
results = train_subsample(modality_list_in, modality_mask, outcomes,
                          l1_dfs_filter, model_params, folds=10)
aucs = [r[0] for r in results]
print(f"Mean AUC: {np.mean(aucs):.3f}")
```

### 6.3. L1 Feature Selection (chi tiết)

Filter áp dụng trên tập Radiomics qua 3 bước:
1. **Robustness filter** (`robustness_cutoff=0.15`): loại features không ổn định giữa perturbations
2. **Outlier filter** (`outlier_cutoff=6`): loại features có quá nhiều outlier
3. **L1 Elastic Net** (`l1_strength=0.1`): chỉ giữ features có hệ số ≠ 0

Kết quả: từ ~1688 features → ~45 features ổn định và dự báo tốt.

### 6.4. Ví dụ sử dụng hoàn chỉnh

```python
from lung_helpers import train, auc_roc_ci

model_params = {
    'epochs': 100, 'lr': 0.01,
    'alpha': 1.0, 'beta': 1.0,
    'cross_modality_enabled': False,
    'attention_gate_enabled': True
}

l1_dfs_filter = {
    0: {  # Radiomics modality
        'l1_selection_df': df_radiomics_full,
        'kwargs': {'robustness_cutoff': 0.15, 'outlier_cutoff': 6, 'l1_strength': 0.1}
    }
}

summary_df, coef_df = train(
    modality_list_in=[df_radiomics, df_genomics, df_clinical],
    modality_mask=modality_mask,
    outcomes=df_outcomes,
    l1_dfs_filter=l1_dfs_filter,
    model_params=model_params,
    folds=10
)

auc, ci = auc_roc_ci(summary_df['label'].values, summary_df['score'].values, 0.95)
print(f"AUC = {auc:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")
```

---

## 7. Cải tiến đề xuất

*Lấy cảm hứng từ bài báo Nature: "Multimodal deep learning for cancer prognosis prediction with clinical information prompts integration"*

### 7.1. Hướng 1: Ngôn ngữ hóa dữ liệu Clinical – NLP Embeddings ✅ Đã triển khai

> Tài liệu đầy đủ: [document/nlp-clinical.md](document/nlp-clinical.md)

**Vấn đề:** Clinical data xử lý qua `RobustScaler` → `[65, 20, 0.8, ...]` biến thành `[-0.12, ...]`, mất ngữ cảnh y khoa. Tuổi 65 trở thành con số vô nghĩa với mô hình.

**Giải pháp đã implement trong [`clinical_nlp_embedding.py`](clinical_nlp_embedding.py):**

**Bước 1 – Tạo text prompt từ 13 chỉ số lâm sàng:**
```python
# Đầu vào: [65, 30, 1, 4.2, 2.8, 1, 0, 1.25, 2, 1, 1, 1, 1]
# Đầu ra:
"The patient is a 65-year-old. Smoking history in pack-years is 30.
 ECOG performance status is 1. Blood albumin concentration is 4.20.
 Derived neutrophil-to-lymphocyte ratio (dNLR) is 2.80.
 The patient is diagnosed with adenocarcinoma at the lung site.
 Metastatic status: with brain metastasis and without liver metastasis.
 Currently undergoing therapy line 2, utilizing combination therapy.
 The patient received prior PD-L1 immunotherapy."
```

**Bước 2 – Trích xuất embedding:**
```python
from clinical_nlp_embedding import ClinicalTextEmbedder
embedder = ClinicalTextEmbedder('sentence-transformers/all-MiniLM-L6-v2')
df_embeddings = embedder.embed_dataframe(df_labs)
# Output: DataFrame (n_patients, 384) – columns: nlp_0...nlp_383
```

**Bước 3 – Tích hợp vào pipeline (no_scale bắt buộc):**
```python
from clinical_nlp_embedding import prepare_nlp_clinical_modality

prepare_nlp_clinical_modality(modality_dict, df_labs, modality_MASK)
nlp_idx = modality_names.index('cnl_nlp_embedding')

model_params_nlp = {**model_params, 'no_scale': [nlp_idx]}  # QUAN TRỌNG
summary_df, coef_df = train(..., model_params=model_params_nlp)
```

> **Lưu ý:** Luôn truyền `no_scale=[nlp_idx]`. NLP embeddings được normalize theo cosine distance – áp `RobustScaler` sẽ phá vỡ cấu trúc ngữ nghĩa.

**Kết quả chuyển đổi chiều:** 13 → **384 chiều**

**Cài đặt:** `pip install sentence-transformers`

**Script so sánh:** [`compare_nlp_clinical.py`](compare_nlp_clinical.py) / [`compare_nlp_clinical_full.py`](compare_nlp_clinical_full.py)

### 7.2. Hướng 2: Feature-Level Cross-Attention

**Vấn đề hiện tại:** `AttentionMatrix` hoạt động ở mức macro (modality-level), chỉ hỏi "modality nào quan trọng hơn?" thay vì "feature cụ thể nào trong Radiomics liên quan đến feature Clinical nào?".

**Giải pháp:** Bidirectional Cross-Attention ở mức đặc trưng – cho phép Clinical Embedding tương tác trực tiếp với Radiomics Features trước khi tính risk score.

*Ví dụ:* Nếu văn bản lâm sàng ghi "tiền sử hút thuốc nặng", attention sẽ tự động chú ý đến các vùng khói trên CT.

### 7.3. Hướng 3: Cosine Similarity Loss

**Vấn đề:** Deep learning có xu hướng "lười" – các nhánh Radiomics và Genomics có thể focus vào cùng một đặc điểm, gây redundancy.

**Giải pháp:** Thêm `Cosine Similarity Loss` để phạt nếu attention matrix của các modality học lặp lại nhau, buộc mô hình khai thác **thông tin bổ sung (complementary)** giữa các modality.

### 7.4. Hướng 4: Foundation Models

**Vấn đề:** Đang dùng static features (~45 Radiomics, ~8 Genomics) → dễ overfitting với dataset nhỏ.

**Giải pháp khi scale-up:**
- **Genomics:** `scFoundation` – biến trình tự gen thành embeddings
- **Radiomics:** `UNI` hoặc `RetCCL` – rút gọn ảnh CT thành vector
- Lợi ích: kế thừa kiến thức từ hàng triệu ảnh y tế toàn cầu

**Khuyến nghị thực tế:** Bắt đầu từ **Hướng 1** (NLP Clinical) vì ít thay đổi code nhất và có thể đánh giá cải thiện ngay.

---

## 8. Tài nguyên học tập

### 8.1. Lộ trình học (Learning Path)

| Bước | Chủ đề | Thời gian |
|------|--------|-----------|
| 1 | Attention Mechanism cơ bản | 1-2 tuần |
| 2 | Attention cho Tabular Data | 1 tuần |
| 3 | Multi-Modal Fusion đơn giản | 2 tuần |
| 4 | Attention-based Fusion | 2 tuần |
| 5 | Missing Modality Handling | 1 tuần |
| 6 | Áp dụng vào dự án | 2-3 tuần |

### 8.2. Papers tham khảo

- **Transformer Attention:** Vaswani et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Bahdanau Attention:** Bahdanau et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*
- **OvO Classification:** Hastie & Tibshirani (1998). *Classification by pairwise coupling*
- **Multimodal Cancer:** *Multimodal deep learning for cancer prognosis prediction with clinical information prompts integration* (Nature)
- **Radiomics:** IBSI Standard – https://theibsi.github.io/
- **PyRadiomics:** https://pyradiomics.readthedocs.io/

### 8.3. Notebooks Kaggle gợi ý

| Notebook | Từ khóa | Độ khó | Học gì |
|---------|---------|--------|--------|
| Attention Mechanism Explained Simply | `attention mechanism explained kaggle` | ⭐ | Attention từ đầu |
| Tabular Data with Neural Attention | `tabular neural attention kaggle` | ⭐⭐ | Attention cho tabular |
| PetFinder Simple Multimodal Baseline | `petfinder multimodal baseline` | ⭐⭐ | Image + Tabular fusion |
| Handling Missing Data in Multimodal | `missing data multimodal kaggle` | ⭐⭐ | Masking mechanism |
| RSNA Pneumonia Detection Multimodal | `rsna pneumonia multimodal kaggle` | ⭐⭐⭐ | Y tế thực tế |

### 8.4. Code mẫu nhanh

```python
# Attention weights cơ bản
attention_scores = torch.matmul(query, key.transpose(-2, -1))
attention_weights = F.softmax(attention_scores, dim=-1)
output = torch.matmul(attention_weights, value)

# Masking modality bị thiếu
mask = torch.tensor([[1, 1, 0], [1, 0, 1]])  # [batch, n_modalities]
masked_features = features * mask.unsqueeze(-1)
```

---

## Phụ lục: Công thức toán học

### Mô hình gốc (AttentionMatrix)

```
Risk:       rᵢ = tanh(Wᵢᵀ · xᵢ)
Attention:  sᵢⱼ = softplus(Wᵢⱼᵀ · xᵢ / |xᵢ|)
Weights:    aⱼ = sⱼ / Σₖ sₖ      (L1 normalize)
Output:     y = Σᵢ (rᵢ × aᵢ)
```

### Mô hình OvO (AttentionMatrixOvO)

```
Risk:       rᵢ = tanh(Wᵢᵀ · xᵢ)
Scores:     sᵢ = Wᵢᵀ · xᵢ / |xᵢ|
Mean:       μᵢ = (1/(N-1)) · Σⱼ≠ᵢ sⱼ
OvO:        ovoᵢ = sigmoid(sᵢ - μᵢ)
Weights:    aᵢ = ovoᵢ / Σⱼ ovoⱼ  (L1 normalize)
Output:     y = Σᵢ (rᵢ × aᵢ)
```

### Loss tổng hợp

```
L = (BCE(y, ŷ) + α·Σ||Wᵢ||₂ + β·||A||_F) / batch_size
```

---

---

## Tài liệu chi tiết (document/)

Tất cả tài liệu chi tiết nằm trong thư mục [`document/`](document/):

### Dữ liệu
| File | Mô tả |
|------|-------|
| [du-lieu-radiomics.md](document/du-lieu-radiomics.md) | Giải thích 1690 cột Radiomics, 18 filter types, 7 feature classes, quy trình PyRadiomics |

### Kiến trúc mô hình
| File | Mô tả |
|------|-------|
| [kien-truc-mo-hinh.md](document/kien-truc-mo-hinh.md) | Tài liệu gốc `MultiModalDynamicModel` – tất cả thành phần |
| [giai-thich-mo-hinh.md](document/giai-thich-mo-hinh.md) | Giải thích step-by-step với ví dụ số liệu cụ thể |
| [dyam-architecture.md](document/dyam-architecture.md) | Sơ đồ Mermaid kiến trúc DyAM + bảng tensor shapes |
| [slides-mo-hinh.md](document/slides-mo-hinh.md) | Bản slide Marp cho trình bày |
| [tong-quan-du-an.md](document/tong-quan-du-an.md) | Slides tổng quan dự án (tiếng Việt) |

### Attention Mechanism
| File | Mô tả |
|------|-------|
| [attention-matrix.md](document/attention-matrix.md) | Chi tiết kiến trúc `AttentionMatrix` gốc + ví dụ số liệu |
| [attention-matrix-ovo.md](document/attention-matrix-ovo.md) | Chi tiết `AttentionMatrixOvO` + bảng so sánh |

### Biến thể OvO
| File | Mô tả |
|------|-------|
| [ovo-attention.md](document/ovo-attention.md) | Tài liệu đầy đủ OvO – kiến trúc, kết quả thực nghiệm 20 test cases |
| [ovo-modalities.md](document/ovo-modalities.md) | Bảng ý nghĩa y học của từng modality |
| [ovo-formula.md](document/ovo-formula.md) | Nguồn gốc công thức `sigmoid(score_i - mean_others)` |
| [ovo-sigmoid-vs-softmax.md](document/ovo-sigmoid-vs-softmax.md) | So sánh chi tiết Sigmoid vs Softmax trong OvO |

### Quy trình huấn luyện
| File | Mô tả |
|------|-------|
| [huong-dan-train.md](document/huong-dan-train.md) | Giải thích đầy đủ hàm `train()` (KFold) với ví dụ số liệu |
| [huong-dan-train-subsample.md](document/huong-dan-train-subsample.md) | Tài liệu hàm `train_subsample()` (ShuffleSplit) |

### NLP Clinical
| File | Mô tả |
|------|-------|
| [nlp-clinical.md](document/nlp-clinical.md) | **Tài liệu triển khai NLP** – API, code nguồn, hướng dẫn tích hợp |
| [nlp-clinical-guide.md](document/nlp-clinical-guide.md) | Giải thích cơ chế hoạt động, thư viện, pipeline |
| [ke-hoach-nlp.md](document/ke-hoach-nlp.md) | Kế hoạch triển khai ban đầu và các câu hỏi mở |

### Cải tiến & Tài nguyên
| File | Mô tả |
|------|-------|
| [cai-tien-mo-hinh.md](document/cai-tien-mo-hinh.md) | 4 hướng cải tiến từ bài báo Nature |
| [tai-nguyen-hoc-tap.md](document/tai-nguyen-hoc-tap.md) | Notebooks Kaggle, papers, lộ trình học |
| [chuyen-doi-pptx.md](document/chuyen-doi-pptx.md) | Hướng dẫn chuyển Markdown → PowerPoint |

---

*Dự án: DyAM – Dynamic Attention Multimodal Model for Lung Cancer Treatment Response Prediction*
