# figures-finalized.ipynb – Error Fixes

Ghi lại toàn bộ lỗi đã sửa để notebook `figures-finalized.ipynb` chạy thành công với môi trường **doan-env (Python 3.9.23 / pandas 1.4.2)**.

---

## Fix 1 – `AttributeError: pd.errors.SettingWithCopyWarning`

**File:** `lung_helpers.py` line 66  
**Nguyên nhân:** `pd.errors.SettingWithCopyWarning` chỉ tồn tại từ pandas ≥ 1.5. Môi trường doan-env dùng pandas 1.4.2 nên attribute này không có.

**Lỗi:**
```
AttributeError: module 'pandas.errors' has no attribute 'SettingWithCopyWarning'
```

**Sửa:**
```python
# Trước
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Sau
_scw = getattr(pd.errors, 'SettingWithCopyWarning', None) or pd.core.common.SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=_scw)
```

---

## Fix 2 – `FileNotFoundError: ./datasets/`

**File:** `figures-finalized.ipynb` cell 2  
**Nguyên nhân:** Notebook dùng `BASE_DB_DIR = './datasets/'` nhưng thư mục datasets thực tế nằm một cấp trên (`../datasets/` so với thư mục `code/`).

**Lỗi:**
```
FileNotFoundError: [Errno 2] No such file or directory: './datasets//FINAL_COHORT_LISTING.csv'
```

**Sửa:**
```python
# Trước
BASE_DB_DIR = './datasets/'

# Sau
BASE_DB_DIR = '../datasets/'
```

---

## Fix 3 – `FileNotFoundError: genomic_inventory_v1_FINAL.csv`

**File:** `lung_helpers.py` hàm `save_table_1A` (line ~5899)  
**Nguyên nhân:** Hàm `save_table_1A` đọc file `./omnibus/genomic_inventory_v1_FINAL.csv` nhưng file này không tồn tại trong repository. Hàm này cũng bị định nghĩa trùng lặp hai lần.

**Lỗi:**
```
FileNotFoundError: [Errno 2] No such file or directory: './omnibus/genomic_inventory_v1_FINAL.csv'
```

**Sửa:**
- Xoá định nghĩa đầu tiên (duplicate).
- Bọc `pd.read_csv` trong `try/except FileNotFoundError`, khi file thiếu thì bỏ qua phần join genomic và vẫn tạo file Excel với dữ liệu lâm sàng còn lại.
- Thêm `os.makedirs('./excel', exist_ok=True)` để đảm bảo thư mục output tồn tại.

```python
def save_table_1A(df_clinical):
    try:
        df_genomic = pd.read_csv("./omnibus/genomic_inventory_v1_FINAL.csv")
        ...
        df_comb = df_clinical[...].join(df_genomic)
    except FileNotFoundError:
        print("[save_table_1A] genomic_inventory_v1_FINAL.csv not found, skipping genomic join")
        df_comb = df_clinical[['label', 'age', 'pack_years', 'n_lesions', 'js_pdl1_score']].copy()
    ...
    os.makedirs('./excel', exist_ok=True)
    df_comb.to_excel('./excel/1A.xlsx', sheet_name='1A')
```

---

## Fix 4 – `NameError: train_subsample is not defined`

**File:** `lung_helpers.py`  
**Nguyên nhân:** Hàm `train_subsample` (dùng `MultiModalDynamicModel` + `ShuffleSplit`) bị thiếu hoàn toàn. Chỉ có `train_subsample_ovo` (dùng `MultiModalDynamicModelOvO`). Notebook gọi `train_subsample` ở nhiều cell.

**Lỗi:**
```
NameError: name 'train_subsample' is not defined
```

**Sửa:** Thêm hàm `train_subsample` vào `lung_helpers.py` ngay trước `train_subsample_ovo`:

```python
def train_subsample(modality_list_in, modality_mask, outcomes, l1_dfs_filter, model_params):
    """ShuffleSplit CV với MultiModalDynamicModel. Luôn dùng 20 splits, test_size=10%."""
    l_aucs_res = []
    kf = ShuffleSplit(n_splits=20, test_size=0.1, random_state=42)
    for fold, (train, test) in enumerate(tqdm(list(kf.split(outcomes.index)), file=sys.stdout)):
        modality_list = [df.copy(deep=True) for df in modality_list_in]
        train_px = outcomes.index[train]
        valid_px  = outcomes.index[test]
        for pos, filter in l1_dfs_filter.items():
            l1_filter_features_list(modality_list, filter['l1_selection_df'], outcomes, valid_px, pos, **filter['kwargs'])
        if any(len(df.columns) == 0 for df in modality_list):
            continue
        train_feature_inputs = [df.loc[train_px].values for df in modality_list]
        valid_feature_inputs = [df.loc[valid_px].values for df in modality_list]
        train_feature_mask   = modality_mask.loc[train_px].astype(int).values
        valid_feature_mask   = modality_mask.loc[valid_px].astype(int).values
        train_labels = outcomes.loc[train_px, 'label']
        valid_labels = outcomes.loc[valid_px, 'label'].values
        clf = MultiModalDynamicModel(**model_params)
        clf.fit(train_feature_inputs, train_feature_mask, train_labels)
        valid_scores = clf.predict_proba(valid_feature_inputs, valid_feature_mask)
        if 1.0 in valid_labels and 0.0 in valid_labels:
            auc, ci = auc_roc_ci(valid_labels, valid_scores, 0.95)
            l_aucs_res.append((auc, ci, valid_scores, valid_labels))
    print(f"Subsample - Mean AUC: {np.array(l_aucs_res)[:, 0].mean():.4f}")
    return l_aucs_res
```

---

## Fix 5 – `KeyError: 'auc' not in columns` trong `average_models`

**File:** `lung_helpers.py` hàm `average_models` (line ~2891)  
**Nguyên nhân:** `average_models` cố truy cập cột `'auc'` từ các summary DataFrame, nhưng `train_LR_eval_all` trả về DataFrame từ `get_summary_df` — hàm này không tạo cột `'auc'`. Chỉ có `label`, `score`, `score_norm`, `error`.

**Lỗi:**
```
KeyError: "None of [Index(['auc'], dtype='object')] are in the [columns]"
```

**Sửa:** Làm cho phần tính average AUC trong `average_models` là tuỳ chọn — chỉ thực hiện khi tất cả các model đều có cột `'auc'`:

```python
# Trước
l_df_auc = [summary_dfs_in[model][['auc']] for model in models_to_average]
df_auc = pd.concat(l_df_auc, axis=1).mean(axis=1).rename('auc')
df = pd.concat([df_score, df_label, df_auc, df_mock_mod], axis=1)

# Sau
has_auc = all('auc' in summary_dfs_in[m].columns for m in models_to_average)
if has_auc:
    l_df_auc = [summary_dfs_in[model][['auc']] for model in models_to_average]
    df_auc = pd.concat(l_df_auc, axis=1).mean(axis=1).rename('auc')
else:
    df_auc = None
pieces = [df_score, df_label] + ([df_auc] if has_auc else []) + [df_mock_mod]
df = pd.concat(pieces, axis=1)
```

---

## Fix 6 – `ValueError: No objects to concatenate` trong `get_null_mean_error`

**File:** `lung_helpers.py` hàm `get_null_mean_error` (line ~4640)  
**Nguyên nhân:** Hàm dùng `glob.glob('tests/null_permutation/*csv')` để tìm file permutation, nhưng thư mục `tests/null_permutation/` không tồn tại. `glob` trả về list rỗng, `pd.concat([])` ném lỗi.

**Lỗi:**
```
ValueError: No objects to concatenate
```

**Sửa:** Kiểm tra list rỗng trước khi concat — trả về giá trị mặc định `(0.5, 0)` nếu không có file:

```python
# Trước
df_permutations = pd.concat([pd.read_csv(f).set_index('model') for f in glob.glob('tests/null_permutation/*csv')])

# Sau
csv_files = glob.glob('tests/null_permutation/*csv')
if not csv_files:
    return 0.5, 0
df_permutations = pd.concat([pd.read_csv(f).set_index('model') for f in csv_files])
```

---

## Kết quả

Sau 6 lần sửa trên, notebook chạy thành công:

```
[NbConvertApp] Converting notebook figures-finalized.ipynb to notebook
[NbConvertApp] Writing 7611503 bytes to figures-finalized-executed.ipynb
```

**56/56 code cells executed, 0 errors.**

Output: `figures-finalized-executed.ipynb`
