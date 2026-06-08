# Hướng dẫn Huấn luyện — `train()` và `train_subsample()`

> Tài liệu chi tiết về hai hàm huấn luyện chính trong `lung_helpers.py`.

---

## 1. Hàm `train()` — KFold Cross-Validation

### 1.1. Chữ ký hàm

```python
summary_df, coef_df = train(modality_list_in, modality_mask, outcomes,
                             l1_dfs_filter, model_params, folds=10)
```

### 1.2. Tham số đầu vào

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `modality_list_in` | `List[DataFrame]` | Mỗi DataFrame = 1 modality (hàng = bệnh nhân, cột = đặc trưng) |
| `modality_mask` | `DataFrame` | Boolean — modality nào có sẵn cho từng bệnh nhân |
| `outcomes` | `DataFrame` | Phải có cột `'label'` (0=PR/CR, 1=SD/PD) |
| `l1_dfs_filter` | `dict` | Bộ lọc L1 feature selection per modality |
| `model_params` | `dict` | Tham số cho `MultiModalDynamicModel` |
| `folds` | `int` hoặc `'LOO'` | Số fold KFold (mặc định 10); `'LOO'` = Leave-One-Out |

**Ví dụ `modality_mask`:**
```python
modality_mask
#            rad_lesion_pc  gen_driver_mut  cnl_dem_labs
# R-001              True            True          True
# R-002              True            True         False  # Thiếu clinical
# R-003              True           False          True  # Thiếu genomics
```

**Ví dụ `l1_dfs_filter`:**
```python
l1_dfs_filter = {
    0: {  # Filter cho modality index 0 (radiomics)
        'l1_selection_df': df_radiology,   # DataFrame tham chiếu đầy đủ
        'kwargs': {
            'robustness_cutoff': 0.15,     # Loại đặc trưng không ổn định
            'outlier_cutoff': 6,           # Loại đặc trưng có outlier
        }
    }
}
```

**Ví dụ `model_params`:**
```python
model_params = {
    'epochs': 100,
    'lr': 0.01,
    'alpha': 1.0,                      # L2 regularization
    'beta': 1.0,                       # Attention norm regularization
    'cross_modality_enabled': False,   # Chỉ attention nội-modality
    'attention_gate_enabled': True     # Nhân attention vào risk
}
```

### 1.3. Luồng xử lý

**Khởi tạo:**
```python
kf = KFold(n_splits=folds, random_state=0, shuffle=True)
```
Với `folds='LOO'`: `folds = len(outcomes.index)` — mỗi fold để lại 1 bệnh nhân.

**Mỗi fold:**
1. Copy deep dữ liệu để tránh thay đổi gốc
2. Áp dụng L1 feature selection (dùng `valid_px` để tránh data leakage)
3. Kiểm tra: nếu modality nào hết feature → bỏ fold
4. Chuẩn bị numpy arrays: `train_feature_inputs`, `valid_feature_inputs`, mask, labels
5. Huấn luyện: `clf = MultiModalDynamicModel(**model_params); clf.fit(...)`
6. Dự đoán: `valid_scores = clf.predict_proba(valid_feature_inputs, valid_feature_mask)`
7. Lấy risk/attention/share: `clf.get_summary_scores(...)`
8. Tính AUC với DeLong CI

### 1.4. Giá trị trả về

**`summary_df`** — DataFrame per bệnh nhân:

| Cột | Mô tả |
|-----|-------|
| `label` | Nhãn thực tế (0/1) |
| `score` | Điểm dự đoán (z-score) |
| `fold` | Fold mà bệnh nhân thuộc về |
| `risk_<modality>` | Risk score từng modality |
| `attn_<modality>` | Attention weight từng modality |
| `share_<modality>` | Tỷ lệ đóng góp từng modality |
| `score_norm` | Score normalize về [0,1] |
| `error` | Sai số tuyệt đối |

**`coef_df`** — Tổng hợp hệ số (weights) từ tất cả fold.

### 1.5. Ví dụ số liệu cụ thể (10-fold, 30 bệnh nhân)

```
Fold 1: Train 27 bệnh nhân → Test 3 → AUC = 0.750
Fold 2: Train 27 bệnh nhân → Test 3 → AUC = 0.833
...
Fold 10: Train 27 bệnh nhân → Test 3 → AUC = 0.750
────────────────────────────────────────────────
Tổng hợp: AUC = 0.782 ± 0.137 (95% CI: [0.645, 0.919])
```

**Ví dụ phân tích attention:**
```python
summary_df[['attn_rad_lesion_pc', 'attn_gen_driver_mut', 'attn_cnl_dem_labs']].mean()
# attn_rad_lesion_pc    0.367   # Radiomics: 36.7%
# attn_gen_driver_mut   0.433   # Genomics: 43.3% ← quan trọng nhất
# attn_cnl_dem_labs     0.200   # Clinical: 20.0%
```

### 1.6. Leave-One-Out CV

LOO tạo N fold (N = số bệnh nhân). Mỗi fold: train trên N-1, test trên 1. Phù hợp với dataset nhỏ (< 50-100 bệnh nhân) — tốn thời gian nhưng đánh giá chính xác nhất.

---

## 2. Hàm `train_subsample()` — ShuffleSplit

### 2.1. Chữ ký hàm

```python
results = train_subsample(modality_list_in, modality_mask, outcomes,
                          l1_dfs_filter, model_params, folds=10)
```

Trả về `list` các tuple `(auc, ci, valid_scores, valid_labels)`.

### 2.2. Điểm khác biệt so với `train()`

| Đặc điểm | `train()` | `train_subsample()` |
|----------|-----------|---------------------|
| Phương pháp | KFold CV | ShuffleSplit |
| Số lần chia | `folds` (mặc định 10) | Luôn **20 lần** (hardcoded) |
| Kích thước test | ~1/folds | 10% cố định |
| Random state | 0 | 42 |
| Giá trị trả về | `(summary_df, coef_df)` | `list[(auc, ci, scores, labels)]` |
| Mục đích chính | KFold CV chuẩn | Error bars cho biểu đồ |

### 2.3. Khởi tạo

```python
kf = ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)
```

**Lưu ý quan trọng:** Dù tham số tên là `folds`, số lần chia thực tế luôn là 20 (không phụ thuộc vào `folds`). Tham số `folds` chỉ xử lý LOO.

### 2.4. Mỗi split

Giống `train()`:
1. Copy deep, áp dụng L1 filter (loại trừ valid_px)
2. Kiểm tra modality rỗng → skip nếu có
3. Huấn luyện `MultiModalDynamicModel`
4. Dự đoán, tính AUC + DeLong CI
5. Chỉ lưu kết quả nếu validation set có cả 2 lớp

### 2.5. Giá trị trả về

```python
results = train_subsample(...)
# results[i] = (auc_i, ci_i, scores_i, labels_i)

aucs = [r[0] for r in results]
mean_auc = np.mean(aucs)  # AUC trung bình qua 20 lần

# Dùng để vẽ error bars trong Figure 4D:
# generate_auc_plot_V2(..., ss_aucs=summary_dfs_ss)
```

### 2.6. Ví dụ sử dụng

```python
from lung_helpers import train_subsample

results = train_subsample(
    modality_list_in=data,
    modality_mask=mask,
    outcomes=labels,
    l1_dfs_filter=dfs_rad_filters,
    model_params=model_params,
    folds=10  # Tham số này không ảnh hưởng đến số lần chia (luôn là 20)
)

mean_auc = np.mean([r[0] for r in results])
print(f"Mean AUC (20 splits): {mean_auc:.3f}")
```

---

## 3. Lưu ý chung về cả hai hàm

- **Data leakage prevention**: L1 feature selection loại trừ `valid_px` → scaler fit chỉ trên train set
- **Missing modality**: `modality_mask` xử lý tự động — attention tái normalize khi thiếu modality
- **Fold skip**: nếu bất kỳ modality nào hết feature sau L1 filter, fold bị bỏ qua
- **AUC skip**: fold không tính AUC nếu validation set không có đủ cả 2 lớp (0 và 1)

---

## 4. So sánh các hàm huấn luyện

| Hàm | CV Method | Output | Khi nào dùng |
|-----|-----------|--------|--------------|
| `train()` | KFold 10-fold | `(summary_df, coef_df)` | Báo cáo AUC chính |
| `train_ovo()` | KFold 10-fold | `(summary_df, coef_df)` | OvO attention variant |
| `train_subsample()` | ShuffleSplit 20x | `list[(auc, ci, ...)]` | Error bars biểu đồ |
| `train_eval_all()` | Single split | summary chi tiết | Debug / exploratory |
| `train_LR()` | KFold | — | Baseline Logistic Regression |

---

*Tổng hợp từ: `huong-dan-train.md`, `huong-dan-train-subsample.md`*
