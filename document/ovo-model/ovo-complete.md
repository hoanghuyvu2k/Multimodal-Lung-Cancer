# OvO Attention – Tài Liệu Tổng Hợp

> Gộp từ: `ovo-attention.md`, `attention-matrix-ovo.md`, `ovo-sigmoid-vs-softmax.md`, `ovo-formula.md`, `ovo-modalities.md`

---

## 1. Tổng Quan

**OvO (One-Versus-Others) Attention** là một biến thể của `MultiModalDynamicModel` gốc, thay thế cơ chế **attention hợp tác** (cooperative) bằng cơ chế **attention cạnh tranh** (competitive).

| | `AttentionMatrix` (Hợp tác) | `AttentionMatrixOvO` (Cạnh tranh) |
|---|---|---|
| **Ý tưởng** | Tất cả modalities chia sẻ "miếng bánh" attention | Mỗi modality phải chứng minh mình tốt hơn các modalities khác |
| **Công thức attention** | `aᵢ = scoreᵢ / Σ(scoreⱼ)` | `aᵢ = sigmoid(scoreᵢ - mean(score_others))` |
| **Cross-modality** | Hỗ trợ tùy chọn | Không cần (OvO đã xử lý) |
| **Số attention layers** | N² layers | N layers |

---

## 2. Kiến Trúc

### 2.1. Mô Hình Gốc (AttentionMatrix)

```
Input Modalities: [M₁, M₂, ..., Mₙ]
       ↓
 Risk Scores: rᵢ = tanh(Wᵢᵀ · xᵢ)
       ↓
 Attention Scores: sᵢ = softplus(Wᵢⱼᵀ · xᵢ) / |xᵢ|
       ↓
 Normalize: aᵢ = sᵢ / Σⱼ sⱼ
       ↓
 Final Risk = Σ(rᵢ × aᵢ)
```

### 2.2. Mô Hình OvO (AttentionMatrixOvO)

```
Input Modalities: [M₁, M₂, ..., Mₙ]
       ↓
 Risk Scores: rᵢ = tanh(Wᵢᵀ · xᵢ)
       ↓
 Attention Scores: sᵢ = Wᵢᵀ · xᵢ / |xᵢ|
       ↓
 OvO: ovoᵢ = sigmoid(sᵢ - mean(sⱼ, j≠i))
       ↓
 Normalize: aᵢ = ovoᵢ / Σⱼ ovoⱼ
       ↓
 Final Risk = Σ(rᵢ × aᵢ)
```

**Vị trí trong code:**
- `AttentionMatrixOvO` – `lung_helpers.py:959-1095`
- `MultiModalDynamicModelOvO` – `lung_helpers.py:1315-1370`
- `train_ovo()` – `lung_helpers.py:1927-1997`

---

## 3. Tại Sao Dùng Sigmoid (Không Phải Softmax)?

### 3.1. So Sánh Công Thức

**Sigmoid (cách dùng hiện tại):**
```python
ovo_i = sigmoid(score_i - mean_others)  # độc lập cho từng modality
attn_weight = normalize(ovo_scores, p=1)  # L1 normalize sau
```

**Softmax (giả định):**
```python
ovo_scores = [score_i - mean_others for i in range(N)]
attn_weight = softmax(ovo_scores)  # tổng = 1 ngay từ đầu
```

### 3.2. Lý Do Chọn Sigmoid

| Tiêu chí | Sigmoid | Softmax |
|---|---|---|
| **Tính toán** | Độc lập cho từng modality | Phụ thuộc lẫn nhau |
| **Khoảng giá trị** | (0,1) cho mỗi ovo_i riêng lẻ | (0,1) và tổng = 1 ngay |
| **Ý nghĩa ngữ nghĩa** | "Xác suất độc lập modality i tốt hơn các modalities khác" | "Xác suất tương đối" (mất ý nghĩa OvO) |
| **Gradient** | Ổn định hơn | Có thể vanishing gradient |
| **Nhất quán với gốc** | Giữ L1 normalize như mô hình gốc | Thay thế hoàn toàn |

### 3.3. Ví Dụ So Sánh Số

Giả sử 3 modalities với scores: M1=0.5, M2=0.3, M3=0.2

**Với Sigmoid:**
```
ovo_1 = sigmoid(0.5 - 0.25) = sigmoid(0.25) = 0.562
ovo_2 = sigmoid(0.3 - 0.35) = sigmoid(-0.05) = 0.488
ovo_3 = sigmoid(0.2 - 0.40) = sigmoid(-0.20) = 0.450
attn  = [0.562, 0.488, 0.450] / 1.5 = [0.375, 0.325, 0.300]
```

**Với Softmax:**
```
ovo_scores = [0.25, -0.05, -0.2]
attn = softmax([0.25, -0.05, -0.2]) = [0.421, 0.311, 0.268]
```

Softmax "khuếch đại" sự vượt trội của M1 nhiều hơn, Sigmoid giữ phân phối attention ôn hòa hơn.

### 3.4. Nguồn Gốc Công Thức

Công thức `sigmoid(score_i - mean(score_others))` là **custom design** dựa trên ý tưởng OvO trong classification, **không** lấy trực tiếp từ một bài báo cụ thể nào. Các tài liệu tham khảo có liên quan:
- OvO classification: Hastie & Tibshirani (1998), *Pairwise Coupling*, NIPS
- Attention mechanism: Vaswani et al. (2017), *Attention Is All You Need*, NIPS
- Không có bài báo nào về "OvO Attention" cụ thể

---

## 4. Luồng Dữ Liệu Chi Tiết – Ví Dụ Với Số

Ví dụ: **1 bệnh nhân**, **3 modalities**, Radiomics bị thiếu.

- M0 (Clinical): `[0.5, -0.2]`
- M1 (Pathology): `[0.9, 0.1, -1.2]`
- M2 (Radiomics): không có — `mask = [1, 1, 0]`

### Bước 1 – Tính Risk Scores

```
R0 = tanh(linear_risk_0(M0)) = 0.66
R1 = tanh(linear_risk_1(M1)) = -0.46
R2 = tanh(linear_risk_2(M2)) = 0.83  → bị loại do mask

risk_weights = [0.66, -0.46, 0.0]
```

### Bước 2 – Tính OvO Attention Scores

```
Attention scores thô (từ linear layers):
  S0 = 0.9,  S1 = 0.6,  S2 = 0.2

OvO cho M0: mean_others = (0.6+0.2)/2 = 0.4 → ovo_0 = sigmoid(0.9-0.4) = sigmoid(0.5) ≈ 0.622
OvO cho M1: mean_others = (0.9+0.2)/2 = 0.55 → ovo_1 = sigmoid(0.6-0.55) = sigmoid(0.05) ≈ 0.512
OvO cho M2: mean_others = (0.9+0.6)/2 = 0.75 → ovo_2 = sigmoid(0.2-0.75) = sigmoid(-0.55) ≈ 0.366

Sau mask: ovo_scores = [0.622, 0.512, 0.0]
L1 normalize: sum = 1.134
attn_weight = [0.548, 0.452, 0.0]
```

### Bước 3 – Kết Hợp

```
risk_weights * attn_weight = [0.66×0.548, -0.46×0.452, 0×0] = [0.362, -0.208, 0.0]
Total_Risk = 0.362 - 0.208 = 0.154
```

### So Sánh Attention Gốc vs OvO

| Modality | Attention Gốc (Hợp tác) | Attention OvO (Cạnh tranh) |
|---|---|---|
| Clinical | 0.9/1.5 = **60.0%** | **54.8%** |
| Pathology | 0.6/1.5 = **40.0%** | **45.2%** |
| Radiomics | 0.0 (mask) | 0.0 (mask) |

Mô hình OvO "điều chỉnh" sự phân bổ attention – không để một modality thống trị tuyệt đối, đồng thời "thưởng" cho modality giữ được giá trị gần mức trung bình.

---

## 5. Kết Quả Thực Nghiệm

**Nguồn:** `excel/ovo_comparison_full_results.xlsx` (20 test cases, 10-fold CV)

### 5.1. Tổng Quan

| Metric | Giá trị |
|---|---|
| OvO tốt hơn Original | 7/20 cases (35%) |
| Original tốt hơn OvO | 9/20 cases (45%) |
| Bằng nhau (1 modality) | 4/20 cases (20%) |
| Cải thiện trung bình | -0.32% |
| Cải thiện tốt nhất (OvO) | +3.27% (PDL1+Gen) |
| Giảm nhiều nhất (OvO) | -4.94% (Rad+IHC-A+Gen) |

### 5.2. Top 5 Cấu Hình OvO Tốt Hơn

| Test Case | Original AUC | OvO AUC | Cải thiện |
|---|---|---|---|
| PDL1+Gen | 0.6931 | **0.7157** | +3.27% |
| Rad+Gen | 0.7384 | **0.7548** | +2.23% |
| Rad+IHC-G+Gen+PDL1 | 0.7839 | **0.8003** | +2.10% |
| Rad+IHC-A+Gen+PDL1 | 0.7640 | **0.7765** | +1.63% |
| TMB+PDL1 | 0.7051 | **0.7151** | +1.42% |

### 5.3. Top 5 Cấu Hình Original Tốt Hơn

| Test Case | Original AUC | OvO AUC | Chênh lệch |
|---|---|---|---|
| Rad+IHC-A+Gen | **0.7567** | 0.7193 | -4.94% |
| IHC-G+Gen | **0.7558** | 0.7231 | -4.33% |
| Rad+IHC-A+Gen+PDL1+Labs | **0.7683** | 0.7466 | -2.83% |
| Rad+IHC-G+Gen | **0.7871** | 0.7662 | -2.66% |
| Rad+IHC-G+Gen+PDL1+Labs | **0.7879** | 0.7834 | -0.56% |

### 5.4. Phân Tích Theo Số Lượng Modalities

**1 modality:** Kết quả giống hệt (OvO không có gì để so sánh).

**2 modalities:** OvO tốt hơn 3/6 cases (50%) – cải thiện rõ rệt với PDL1+Gen, Rad+Gen.

**3+ modalities:** Original tốt hơn 6/10 cases (60%) – đặc biệt khi có IHC-G hoặc Labs.

### 5.5. Cấu Hình Tốt Nhất

- **OvO:** `Rad+IHC-G+Gen+PDL1` – AUC = **0.8003** (95% CI: 0.739–0.862) ✅
- **Original:** `Rad+IHC-G+Gen+PDL1+Labs` – AUC = 0.7879 (95% CI: 0.723–0.853)

---

## 6. Ý Nghĩa Y Học Của Các Modalities

| Tên modality | Nhóm dữ liệu | Ý nghĩa y học |
|---|---|---|
| `rad_lesion_pc` | Radiomics – khối u phổi nguyên phát (PC) | Đặc trưng hình dạng, kích thước, texture của khối u chính trên CT. Phản ánh gánh nặng u và mức độ ác tính. |
| `rad_lesion_pl` | Radiomics – tổn thương màng phổi (PL) | Đặc trưng tổn thương ở màng phổi; gợi ý bệnh lan rộng (malignant pleural disease), tiên lượng xấu hơn. |
| `rad_lesion_ln` | Radiomics – hạch lympho (LN) | Đặc trưng hạch trung thất/rốn phổi. Đánh giá di căn hạch – yếu tố quan trọng trong giai đoạn bệnh. |
| `rad_lesion_lu` | Radiomics – lesion tổng hợp (top lesion) | Đại diện cho lesion nổi bật nhất (volume lớn nhất) từ PC/PL/LN. Tóm tắt gánh nặng hình ảnh quan trọng nhất. |
| `path_ihc_pdl1` (IHC-A) | Pathology – IHC PD-L1 | Đặc trưng định lượng/diện tích/cường độ nhuộm IHC PD-L1 trên mô. Biomarker trực tiếp cho đáp ứng thuốc PD-1/PD-L1. |
| `path_ihc_glcm` (IHC-G) | Pathology – IHC texture (GLCM) | Đặc trưng texture (GLCM) của tín hiệu PD-L1 trên mô. Nắm bắt pattern vi mô phức tạp hơn chỉ số TPS thô. |
| `gen_driver_tmb` | Genomics – Tumor Mutational Burden | Số lượng đột biến soma/Mb DNA. TMB cao → nhiều neoantigen → hệ miễn dịch dễ nhận diện u. |
| `gen_driver_mut_amp` | Genomics – driver mutations & amplifications | Tình trạng đột biến/khuếch đại gene driver (EGFR, ALK, KRAS…). Phân tầng "oncogene-driven" vs "wild-type". |
| `cnl_pdl1_score` | Clinical – PD-L1 TPS | Tỉ lệ % tế bào u bắt màu PD-L1 (0–100%). Biomarker chuẩn dùng trong chỉ định thuốc PD-1/PD-L1. |
| `cnl_dem_labs` | Clinical – demographics & labs | Tuổi, pack_years, ECOG, albumin, dNLR, di căn não/gan, tumor burden, line of therapy,… |

---

## 7. Hướng Dẫn Sử Dụng

### 7.1. Mô Hình Gốc

```python
from lung_helpers import train, get_training_data

data, mask, labels = get_training_data(modal_list, modality_dict, modality_MASK, df_outcomes)
summary_df, coef_df = train(data, mask, labels, l1_filter, model_params, folds=10)
```

### 7.2. Mô Hình OvO

```python
from lung_helpers import train_ovo, get_training_data

data, mask, labels = get_training_data(modal_list, modality_dict, modality_MASK, df_outcomes)
summary_df_ovo, coef_df_ovo = train_ovo(data, mask, labels, l1_filter, model_params, folds=10)
# Lưu ý: cross_modality_enabled tự động bị loại bỏ trong train_ovo()
```

### 7.3. So Sánh AUC

```python
from lung_helpers import auc_roc_ci

auc_orig, ci_orig = auc_roc_ci(summary_df['label'].values, summary_df['score'].values, 0.95)
auc_ovo,  ci_ovo  = auc_roc_ci(summary_df_ovo['label'].values, summary_df_ovo['score'].values, 0.95)

print(f"Original: {auc_orig:.4f} (95% CI: {ci_orig[0]:.3f}–{ci_orig[1]:.3f})")
print(f"OvO:      {auc_ovo:.4f}  (95% CI: {ci_ovo[0]:.3f}–{ci_ovo[1]:.3f})")
print(f"Delta:    {auc_ovo - auc_orig:+.4f}")
```

---

## 8. Scripts So Sánh

| Script | Mục đích |
|---|---|
| `compare_ovo_attention_full.py` | KFold 10-fold, tất cả 20 test cases, lưu Excel |
| `compare_ovo_subsample.py` | ShuffleSplit 20 splits, tập trung cấu hình quan trọng |
| `compare_ovo_sigmoid_vs_softmax.py` | So sánh trực tiếp sigmoid vs softmax variant |

```bash
python compare_ovo_attention_full.py    # full comparison
python compare_ovo_subsample.py         # subsample comparison
```

---

## 9. Kết Luận & Khuyến Nghị

**Dùng OvO khi:**
- Có **2–3 modalities** (cải thiện +2–3% AUC)
- Các modalities có thể **thay thế** nhau (substitutable)
- Muốn tự động chọn lọc modality kém chất lượng

**Dùng Original khi:**
- Có **3+ modalities phức tạp**, đặc biệt khi có IHC-G hoặc Labs
- Các modalities **bổ sung** cho nhau (complementary)
- Cần cross-modality attention

**Khuyến nghị chung:** Thử cả hai và so sánh. Cấu hình điểm chuẩn tốt nhất là `Rad+IHC-G+Gen+PDL1` (OvO đạt AUC = 0.8003 ✅).

---

## 10. Phụ Lục: Công Thức Toán Học

### Mô Hình Gốc

```
Risk scores:      rᵢ = tanh(Wᵢᵀ · xᵢ)
Attention scores: sᵢ = softplus(Wᵢⱼᵀ · xᵢ) / |xᵢ|
Attention weights: aᵢ = sᵢ / Σⱼ sⱼ
Final output:     y = Σᵢ (rᵢ × aᵢ)
```

### Mô Hình OvO

```
Risk scores:      rᵢ = tanh(Wᵢᵀ · xᵢ)
Attention scores: sᵢ = Wᵢᵀ · xᵢ / |xᵢ|
Mean others:      μᵢ = (1/(N-1)) · Σⱼ≠ᵢ sⱼ
OvO scores:       ovoᵢ = sigmoid(sᵢ - μᵢ)
Attention weights: aᵢ = ovoᵢ / Σⱼ ovoⱼ
Final output:     y = Σᵢ (rᵢ × aᵢ)
```
