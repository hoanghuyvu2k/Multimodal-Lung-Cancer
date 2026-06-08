# NLP Clinical Embedding — Triển khai và Tích hợp

> Tài liệu kỹ thuật đầy đủ về module NLP Clinical: từ ý tưởng, pipeline, code nguồn, đến tích hợp vào notebook.

---

## 1. Mục đích

Thay vì đưa dữ liệu lâm sàng dạng số thô (`[65, 30, 1, 4.2, ...]`) qua `RobustScaler` — làm mất ngữ cảnh y khoa — phương pháp NLP:

1. Biến **13 chỉ số lâm sàng** thành **câu văn tiếng Anh** có ý nghĩa y khoa
2. Trích xuất **384 chiều embedding** giàu thông tin qua `all-MiniLM-L6-v2`
3. Tích hợp vào pipeline `train()` / `train_ovo()` như một modality độc lập

**Chuyển đổi chiều:** 13 chiều (raw) → **384 chiều** (NLP dense vector)

**Tại sao tiếng Anh?** Tối ưu hóa vocab của model HuggingFace được train trên corpus tiếng Anh.

---

## 2. Cài đặt

```bash
pip install sentence-transformers
```

Dependency này **không có trong `requirements.txt`** — cài một lần trước khi chạy.

---

## 3. Module `clinical_nlp_embedding.py`

### 3.1. `df_to_text_prompts(df_clinical_labs)`

Chuyển mỗi hàng bệnh nhân thành câu văn tiếng Anh.

**13 cột đầu vào:**

| Cột | Mô tả |
|-----|-------|
| `age` | Tuổi bệnh nhân |
| `pack_years` | Số năm hút thuốc |
| `ecog` | ECOG performance status (0–4) |
| `albumin` | Nồng độ albumin máu |
| `dnlr` | Derived Neutrophil-to-Lymphocyte Ratio |
| `tumor_burden` | Gánh nặng khối u ban đầu |
| `brain_mets` | Di căn não (0/1) |
| `liver_mets` | Di căn gan (0/1) |
| `therapy_line` | Dòng điều trị |
| `recieves_combo_therapy` | Liệu pháp kết hợp (0/1) |
| `site_lung` | Vị trí phổi (0/1) |
| `recieves_pdl1_therapy` | Đã dùng PD-L1 trước (0/1) |
| `hist_adeno` | Mô học adenocarcinoma (0/1) |

**Ví dụ prompt đầu ra:**
```
"The patient is a 65-year-old. Smoking history in pack-years is 30.
ECOG performance status is 1. Blood albumin concentration is 4.20.
Derived neutrophil-to-lymphocyte ratio (dNLR) is 2.80.
Initial tumor burden stands at 1.25.
The patient is diagnosed with adenocarcinoma at the lung site.
Metastatic status: with brain metastasis and without liver metastasis.
Currently undergoing therapy line 2, utilizing combination therapy.
The patient received prior PD-L1 immunotherapy."
```

### 3.2. `ClinicalTextEmbedder`

```python
embedder = ClinicalTextEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
df_embeddings = embedder.embed_dataframe(df_labs)
# Output: DataFrame shape (n_patients, 384), columns: nlp_0...nlp_383
```

**Model `all-MiniLM-L6-v2`:**
- Kích thước: ~90MB (tải một lần, cache tại `~/.cache/huggingface/`)
- Embedding dimension: **384**
- Tốc độ: ~1000 bệnh nhân/giây trên CPU; ~280 bệnh nhân mất 5–10 giây
- Chất lượng: excellent sentence similarity, tối ưu cho inference

### 3.3. `prepare_nlp_clinical_modality(df_dict, df_labs, modality_mask, name)`

Hàm tích hợp end-to-end:

```python
from clinical_nlp_embedding import prepare_nlp_clinical_modality

prepare_nlp_clinical_modality(
    df_dict=modality_dict,         # cập nhật in-place
    df_labs=df_clinical_labs,
    modality_mask=modality_MASK,   # cập nhật in-place
    name='cnl_nlp_embedding'
)
```

Sau lời gọi này:
- `modality_dict['cnl_nlp_embedding']` = DataFrame (n_patients, 384)
- `modality_MASK['cnl_nlp_embedding']` = True/False cho từng bệnh nhân

---

## 4. Code nguồn đầy đủ

```python
# clinical_nlp_embedding.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def df_to_text_prompts(df_clinical_labs):
    prompts = []
    df_clean = df_clinical_labs.fillna(0)

    for idx, row in df_clean.iterrows():
        age          = int(row.get('age', 0))
        pack_years   = row.get('pack_years', 0)
        ecog         = row.get('ecog', 0)
        albumin      = row.get('albumin', 0)
        dnlr         = row.get('dnlr', 0)
        tumor_burden = row.get('tumor_burden', 0)
        brain_mets   = "with" if row.get('brain_mets') == 1 else "without"
        liver_mets   = "with" if row.get('liver_mets') == 1 else "without"
        therapy_line = int(row.get('therapy_line', 1))
        combo        = "combination therapy" if row.get('recieves_combo_therapy') == 1 else "monotherapy"
        site         = "lung" if row.get('site_lung') == 1 else "extra-pulmonary"
        pdl1         = "received" if row.get('recieves_pdl1_therapy') == 1 else "did not receive"
        hist         = "adenocarcinoma" if row.get('hist_adeno') == 1 else "non-adenocarcinoma"

        prompt = (
            f"The patient is a {age}-year-old. Smoking history in pack-years is {pack_years}. "
            f"ECOG performance status is {ecog}. Blood albumin concentration is {albumin:.2f}. "
            f"Derived neutrophil-to-lymphocyte ratio (dNLR) is {dnlr:.2f}. "
            f"Initial tumor burden stands at {tumor_burden:.2f}. "
            f"The patient is diagnosed with {hist} at the {site} site. "
            f"Metastatic status: {brain_mets} brain metastasis and {liver_mets} liver metastasis. "
            f"Currently undergoing therapy line {therapy_line}, utilizing {combo}. "
            f"The patient {pdl1} prior PD-L1 immunotherapy."
        )
        prompts.append(prompt)
    return prompts


class ClinicalTextEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"[NLP] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[NLP] Ready. Embedding dim = {self.dim}")

    def embed_dataframe(self, df_clinical_labs):
        print(f"[NLP] Encoding {len(df_clinical_labs)} patients...")
        prompts = df_to_text_prompts(df_clinical_labs)
        embeddings = self.model.encode(prompts, show_progress_bar=False)
        df_emb = pd.DataFrame(embeddings, index=df_clinical_labs.index)
        df_emb.columns = [f"nlp_{c}" for c in df_emb.columns]
        return df_emb


def prepare_nlp_clinical_modality(df_dict, df_labs, modality_mask, name='cnl_nlp_embedding'):
    embedder = ClinicalTextEmbedder()
    df_emb = embedder.embed_dataframe(df_labs)
    if name not in modality_mask.columns:
        modality_mask[name] = False
    modality_mask.loc[df_emb.index, name] = True
    df_dict[name] = df_emb
    print(f"[NLP] Modality '{name}' injected. Shape: {df_emb.shape}")
    return df_emb
```

---

## 5. Tích hợp vào pipeline train

```python
from clinical_nlp_embedding import prepare_nlp_clinical_modality
from lung_helpers import train, get_training_data

# 1. Chuẩn bị modality NLP (cập nhật modality_dict và modality_MASK in-place)
prepare_nlp_clinical_modality(modality_dict, df_labs, modality_MASK)

# 2. Tìm index của NLP modality — PHẢI tính động, không hardcode
nlp_idx = list(modality_MASK.columns).index('cnl_nlp_embedding')

# 3. model_params với no_scale — bắt buộc để bảo toàn cosine distance
model_params_nlp = {**model_params, 'no_scale': [nlp_idx]}

# 4. Lấy data và train
modality_names = ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln',
                  'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_nlp_embedding']
data, mask, labels = get_training_data(modality_names, modality_dict, modality_MASK, df_outcomes)

summary_df, coef_df = train(data, mask, labels, dfs_rad_filters, model_params_nlp, folds=10)
```

> **Tại sao phải có `no_scale`?** Embeddings từ sentence-transformers đã được chuẩn hóa về unit sphere (chuẩn L2=1). `RobustScaler` sẽ trừ median và chia IQR theo từng chiều → phá vỡ cấu trúc ngữ nghĩa cosine. Với 13 cột số thô (`cnl_dem_labs`), scaler là cần thiết; với 384-dim NLP thì ngược lại.

> **Tại sao không dùng `train_LR`?** `train_LR` dùng `RobustScaler` hardcoded bên trong, không có tham số `no_scale`. NLP embeddings 384d không thể đưa vào LR trực tiếp.

---

## 6. Tích hợp vào `Figures-Finalized-NLP.ipynb`

File `Figures-Finalized-NLP.ipynb` là bản copy của `Figures-Finalized.ipynb` với **7 thay đổi**:

| Cell | Vị trí | Loại | Mục đích |
|------|---------|------|----------|
| 14 | Sau model_params | Mới | NLP Setup — khởi tạo modality, tính `no_scale` index |
| 20 | Sau DyAM training | Mới | Train 2 model DyAM+NLP (10-fold KFold) |
| 23 | Sau subsample | Mới | NLP Subsample với ShuffleSplit |
| 52 | Model lists | Sửa | Thêm `NLP_models` và `NLP_colors` |
| 53 | Metrics table | Sửa | Thêm NLP vào bảng EF3 |
| 63 | KM plots | Sửa | Thêm 4 KM plots cho NLP models |
| 68 | Cuối notebook | Mới | Bảng so sánh AUC Raw vs NLP |

### Cell 14 — NLP Setup

```python
import importlib, clinical_nlp_embedding
importlib.reload(clinical_nlp_embedding)
from clinical_nlp_embedding import prepare_nlp_clinical_modality

prepare_nlp_clinical_modality(modality_dict, df_labs, modality_MASK, 'cnl_nlp_embedding')
nlp_modality_idx = list(modality_MASK.columns).index('cnl_nlp_embedding')
model_params_nlp = {**model_params, 'no_scale': [nlp_modality_idx]}
```

### Cell 20 — Training

```python
data, mask, labels = get_training_data(
    ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln',
     'path_ihc_pdl1', 'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_nlp_embedding'],
    modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1+NLP'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params_nlp)

data, mask, labels = get_training_data(
    ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln',
     'path_ihc_glcm', 'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_nlp_embedding'],
    modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-G+Gen+PDL1+NLP'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params_nlp)
```

### Cell 52 — Model Lists

```python
NLP_colors = sns.color_palette("Greens", 4)
NLP_models = ['DyAM Rad+IHC-A+Gen+PDL1+NLP', 'DyAM Rad+IHC-G+Gen+PDL1+NLP']

model_dict = {
    ...
    'DyAM Rad+IHC-A+Gen+PDL1+NLP': {'color': NLP_colors[3]},
    'DyAM Rad+IHC-G+Gen+PDL1+NLP': {'color': NLP_colors[2]},
}

annot_list = [
    ...
    {'name': 'DyAM+NLP\nClinical',
     'left': 'DyAM Rad+IHC-A+Gen+PDL1+NLP',
     'right': 'DyAM Rad+IHC-G+Gen+PDL1+NLP'},
]
```

### Cell 68 — AUC Comparison

```python
compare_pairs = [
    ('DyAM Rad+IHC-A+Gen+PDL1+Labs', 'DyAM Rad+IHC-A+Gen+PDL1+NLP'),
    ('DyAM Rad+IHC-G+Gen+PDL1+Labs', 'DyAM Rad+IHC-G+Gen+PDL1+NLP'),
]
for raw_name, nlp_name in compare_pairs:
    for name in [raw_name, nlp_name]:
        if name in summary_dfs:
            auc, ci = auc_roc_ci(summary_dfs[name]['label'].values,
                                 summary_dfs[name]['score'].values, 0.95)
            print(f"{name:<42} {auc:.3f}   [{ci[0]:.3f}-{ci[1]:.3f}]")
```

---

## 7. Luồng dữ liệu tổng thể

```
df_labs (13 cột số/bệnh nhân)
        ↓
prepare_nlp_clinical_modality()          [clinical_nlp_embedding.py]
  df_to_text_prompts()                   → câu văn tiếng Anh
  ClinicalTextEmbedder.encode()          → 384-dim vector (cosine normalized)
        ↓
modality_dict['cnl_nlp_embedding']       DataFrame n×384
modality_MASK['cnl_nlp_embedding']       bool series
        ↓
get_training_data([..., 'cnl_nlp_embedding'])
        ↓
train(..., model_params_nlp)             [lung_helpers.py]
  no_scale=[idx] → bỏ qua RobustScaler cho NLP
  AttentionMatrix học trọng số NLP vs Rad/IHC/Gen/PDL1
        ↓
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1+NLP']
  → score, attn_cnl_nlp_embedding, label
        ↓
Figure 4D (AUC bar) | EF3 (metrics table) | 5B/6B (KM plots) | Cell 68 (comparison)
```

---

## 8. Thứ tự chạy notebook

1. Cell 14 (NLP Setup) phải chạy **trước** cell 20 (NLP Training)
2. Nếu kernel restart: phải chạy lại từ đầu
3. Lần đầu: download model ~90MB từ HuggingFace Hub (cần internet)
4. Các lần sau: model được cache tự động

---

*Tổng hợp từ: `ke-hoach-nlp.md`, `nlp-clinical-guide.md`, `nlp-clinical.md`, `figures-finalized-nlp-changes.md`*
