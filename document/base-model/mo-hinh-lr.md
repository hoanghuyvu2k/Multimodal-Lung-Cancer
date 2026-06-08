# Mô hình LR trong dự án DyAM

## Tóm tắt nhanh

Trong dự án này có **hai thứ khác nhau** đều được gọi là "LR":

| Tên | File | Framework | Có Attention không? |
|-----|------|-----------|---------------------|
| `train_LR` | `lung_helpers.py:3254` | sklearn | **Không** — LR thuần túy |
| `MILR` / `train_MILR` | `lung_helpers.py:772` | PyTorch | **Có** — softmax attention qua nhiều tổn thương |

---

## 1. `train_LR` — Logistic Regression cơ bản

### Không có cơ chế attention nào cả

`train_LR` sử dụng `LogisticRegression` của scikit-learn thuần túy:

```python
# lung_helpers.py, line ~3270
clf = LogisticRegression(
    penalty='elasticnet',
    max_iter=2500,
    solver='saga',
    l1_ratio=0.5,
    C=0.1,
    class_weight='balanced'
)
clf.fit(X_train, train_labels)
valid_scores = clf.predict_proba(X_valid)[:, 1] - 0.5
```

**Cơ chế hoạt động:**
- Nhận vào ma trận đặc trưng `X` (bệnh nhân × feature) và nhãn `y`
- Tối ưu hóa hàm mất mát logistic với regularization **ElasticNet** (kết hợp L1 + L2)
- Dự đoán bằng hàm sigmoid thuần túy: `p = sigmoid(w·x + b)`
- Không có bước nào tính trọng số attention cho từng modality hay từng lesion

**Quy trình training:**
1. `RobustScaler` chuẩn hóa features (loại bỏ outlier)
2. 10-fold KFold cross-validation trên discovery cohort
3. Không có feature selection động — ElasticNet tự co hệ số về 0
4. Output: `summary_df` chứa score dự đoán và label của từng bệnh nhân

**ElasticNet là gì?**
- `l1_ratio=0.5` → 50% L1 (LASSO) + 50% L2 (Ridge)
- L1 làm co hệ số về đúng 0 → feature selection tự động
- L2 ổn định khi features tương quan với nhau (ví dụ EGFR ↔ TMB)
- `C=0.1` → regularization mạnh → model đơn giản, chống overfitting

**Khi nào dùng `train_LR`:**
- Baseline để so sánh với DyAM
- Khi chỉ có một modality (LR Rad-PC, LR Rad-LN, LR Gen, LR PDL1-TPS)
- Figure 4B: Vẽ violin của hệ số EGFR và STK11 qua 10 fold

---

## 2. `MILR` — Multi-Instance Logistic Regression (có Attention)

### Tên "LR" nhưng thực chất là mạng neural với attention

`MILR` là một `nn.Module` PyTorch — **không phải** `LogisticRegression` của sklearn. Tên "LR" xuất phát từ lý thuyết **Multi-Instance Learning** (MIL), không phải Logistic Regression thông thường.

```python
# lung_helpers.py, line ~772
class MILR(nn.Module):
    def __init__(self, K, L, H):
        super(MILR, self).__init__()
        self.attn = nn.Linear(self.L, 1, bias=False)   # Attention scorer
        self.rfct = nn.Linear(self.L, self.H, bias=True)  # Risk encoder
        self.intr = nn.Linear(self.H, 1, bias=True)    # Classifier
        self.sm   = nn.Softmax(dim=0)                   # Chuẩn hóa attention

    def forward(self, input):
        # input: (N_lesions, L_features) — nhiều tổn thương của 1 bệnh nhân
        Cterm = self.rfct(input)                          # (N, H) — risk mỗi lesion
        Aterm = self.attn(input)                          # (N, 1) — attention score mỗi lesion
        Eterm = torch.mm(
            self.sm(Aterm).transpose(1, 0),              # (1, N) — attention weights (tổng=1)
            self.tanh(Cterm)                              # (N, H)
        ).view(self.H)                                    # (H,) — tổng hợp có trọng số
        logit = self.intr(Eterm)                          # scalar — dự đoán cuối
        return logit, Aterm, Cterm, Eterm
```

**Cơ chế attention trong MILR:**

```
Bệnh nhân A có 3 tổn thương PC:
  Lesion 1 → score 0.8
  Lesion 2 → score 0.1  
  Lesion 3 → score 0.1

Attention weights (softmax): [0.70, 0.15, 0.15]

Risk tổng hợp = 0.70×risk₁ + 0.15×risk₂ + 0.15×risk₃
              ↓
Dự đoán: sigmoid(W·risk_tổng_hợp + b)
```

**Mục đích:**
- Mỗi bệnh nhân có **nhiều tổn thương** cùng loại (PC, PL, LN)
- Mô hình học tự động tổn thương nào quan trọng nhất (softmax → tổng = 1)
- Đây là **instance-level attention** (chọn lesion quan trọng), khác với **modality-level attention** của DyAM

**Tham số:**
- `K` = số lesion tối đa (padding nếu ít hơn)
- `L` = số feature radiomic mỗi lesion
- `H` = kích thước hidden layer (risk representation)

### `MultiLesionModel` — Wrapper của MILR

```python
# lung_helpers.py, line ~673
class MultiLesionModel:
    def __init__(self, K, L, H):
        self.model = MILR(K, L, H)  # PyTorch module bên trong
        self.scaler = RobustScaler()

    def fit(self, X, y):
        # Training loop PyTorch với BCEWithLogitsLoss
        ...

    def predict_proba(self, X):
        # Trả về xác suất [0, 1]
        ...
```

`MultiLesionModel` bọc `MILR` và cung cấp interface giống sklearn (`fit` / `predict_proba`) để dễ tích hợp vào pipeline.

### `train_MILR` — Training loop

```python
# lung_helpers.py, line ~2990
def train_MILR(data, mask, labels, model_params, folds=10):
    # 10-fold KFold (giống train_LR)
    # Nhưng dùng MultiLesionModel thay vì sklearn LogisticRegression
    ...
```

---

## 3. So sánh ba loại mô hình

| Thuộc tính | `train_LR` | `train_MILR` / MILR | `train` / DyAM |
|------------|------------|---------------------|----------------|
| Framework | sklearn | PyTorch | PyTorch |
| Attention | Không | Có (instance-level) | Có (modality-level) |
| Mục tiêu attention | — | Chọn lesion quan trọng | Chọn modality quan trọng |
| Input | 1 vector/bệnh nhân | Ma trận lesion/bệnh nhân | Nhiều modality |
| Regularization | ElasticNet (L1+L2) | BCEWithLogitsLoss + weight | BCEWithLogitsLoss + L1 |
| Số modality | 1 | 1 (multi-lesion) | Nhiều |
| Diễn giải | Hệ số `coef_` | Attention weight per lesion | Attention weight per modality |

### Loại attention khác nhau như thế nào

```
MILR (instance-level attention):
  Bệnh nhân → [Lesion 1, Lesion 2, Lesion 3]
                    ↓
              Attention weights → Tổng hợp → Dự đoán
  (TRONG cùng 1 modality, chọn lesion nào quan trọng)

DyAM (modality-level attention):
  Bệnh nhân → [Rad-PC risk, IHC-A risk, Gen risk, PDL1 risk]
                    ↓
              Attention weights → Tổng hợp → Dự đoán
  (GIỮA các modality khác nhau, chọn loại dữ liệu nào quan trọng)
```

---

## 4. Kết quả thực tế (từ notebook)

**LR cơ bản (`train_LR`) — discovery cohort, 10-fold KFold:**

| Model | AUC | Ghi chú |
|-------|-----|---------|
| LR Clinical | 0.570 | Baseline thấp nhất |
| LR Rad-PC | 0.641 | Radiomics tổn thương PC |
| LR Rad-LN | 0.681 | Radiomics hạch lympho — tốt hơn PC |
| LR PDL1-TPS | 0.729 | PDL1 — marker lâm sàng mạnh nhất |
| LR Gen (No-TMB) | gần thất bại | F1 ≈ 0.486, model không học được |
| LR Gen (with TMB) | 0.729 | TMB đóng góp quan trọng |

**MILR (`train_MILR`) — cùng discovery cohort:**

| Model | AUC | So sánh |
|-------|-----|---------|
| MILR Rad-PC | ~0.612 | **Thấp hơn** LR Rad-PC (0.641) |

**Nhận xét:** MILR dù có attention nhưng không vượt LR thông thường trên radiomics PC. Điều này cho thấy với bộ dữ liệu nhỏ (n ≈ 280), mạng neural attention thêm tham số nhưng không học được gì hữu ích hơn LR đơn giản.

**DyAM (`train`) — best model:**

| Model | AUC |
|-------|-----|
| DyAM Rad+IHC-G+Gen+PDL1 | 0.788 |
| DyAM (best single) | > LR mọi modality |

---

## 5. Khi nào dùng loại nào

- **`train_LR`:** Baseline nhanh, 1 modality, cần hệ số giải thích được (Figure 4B)
- **`train_MILR`:** Khi muốn xử lý nhiều lesion cùng loại, nhưng không cần cross-modality
- **`train`/DyAM:** Khi có nhiều modality khác nhau, cần học trọng số giữa các loại dữ liệu

Trong các figure chính của paper, `train_LR` được dùng làm **baseline comparator** cho DyAM. `train_MILR` xuất hiện ở cell so sánh các mô hình (cell 16–17) nhưng không phải mô hình đề xuất chính.

---

## 6. Tóm tắt

> **`train_LR`** = Logistic Regression sklearn thuần túy, **không có attention**, dùng ElasticNet để chọn feature, là baseline đơn giản nhất.
>
> **`MILR`/`train_MILR`** = Mạng neural PyTorch **có attention** qua softmax, nhưng chỉ attention ở mức lesion (chọn tổn thương nào quan trọng trong 1 modality), không phải cross-modality.
>
> **DyAM (`AttentionMatrix`/`AttentionMatrixOvO`)** = Mô hình chính, attention ở mức modality (học bệnh nhân này nên dựa vào loại dữ liệu nào — Radiology, IHC, Genomics hay PDL1).
