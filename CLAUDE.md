# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running experiments

All scripts must be run from the `code/` directory. Data files are expected at `../datasets/` (one level up).

```bash
# Run a specific comparison experiment
python compare_ovo_attention_full.py   # OvO vs original attention (all modality combos, 10-fold CV)
python compare_nlp_clinical_full.py    # NLP embedding vs raw clinical (full run)
python compare_4way_full.py            # 4-way: Original | OvO | NLP | NLP+OvO
python compare_ovo_subsample.py        # OvO vs original with ShuffleSplit
python debug_dyam_step_by_step.py      # Step-by-step debugging of a single forward pass

# NLP requires extra dependency (not in requirements.txt)
pip install sentence-transformers
```

There are no test suites, linting configs, or build steps. This is a research codebase.

All scripts use `importlib.reload(lung_helpers)` at the top — this is intentional for hot-reload during development, not a bug.

## Architecture

### Core module: `lung_helpers.py`

The single large library (~5300 lines) containing everything. Importing it also sets global matplotlib/seaborn style and suppresses FutureWarnings. Key sections:

**Model classes (PyTorch `nn.Module`):**
- `AttentionMatrix` – cooperative attention: N×N linear layers, softplus activation, L1-normalized weights
- `AttentionMatrixOvO` – competitive attention: N linear layers, `sigmoid(score_i - mean_others)`, L1-normalized
- `AttentionMatrixOvO_Softmax` – same but uses softmax instead of sigmoid
- `MultiModalDynamicModel` – wraps `AttentionMatrix`; handles RobustScaler per modality, BCEWithLogitsLoss with auto class-weighting, Adam optimizer, z-score output normalization
- `MultiModalDynamicModelOvO` – wraps `AttentionMatrixOvO`
- `MultiModalDynamicModelGMU` – wraps `GatedMultimodalUnitFusion` (alternative architecture)
- `MaskedMultiModalLoader` – PyTorch `Dataset` for multimodal batching with mask
- `MILR`, `MultiLesionModel` – older/alternative model variants

**Training functions** (all return `(summary_df, coef_df)` except `train_subsample`):
- `train(...)` – KFold CV, returns per-patient `summary_df` with risk/attention/share scores
- `train_ovo(...)` – same but uses `MultiModalDynamicModelOvO`
- `train_gmu(...)` – same but uses `MultiModalDynamicModelGMU`
- `train_subsample(...)` – ShuffleSplit (always 20 splits, 10% test), returns list of `(auc, ci, scores, labels)`
- `train_eval_all(...)` – single train/val split, no CV
- `train_LR(...)`, `train_XGBoost(...)`, `train_RandomForest(...)` – baseline comparators

**Data preparation functions** (build `modality_dict` + `modality_MASK` in-place):
- `get_clinical_table_v2(path, main_index_col, cohort)` – loads and filters clinical CSV
- `decorate_with_site_index(df_radiology)` – splits radiomics by lesion type (PC/PL/LN)
- `prepare_rad_modality_by_size(df_dict, df, modality_mask, sites, name, ...)` – selects top-N lesions by size
- `prepare_rad_modality(df_dict, df, modality_mask, sites, l_idx, name)` – selects specific lesion index
- `prepare_other_modalities(df_dict, df, modality_mask, name)` – for genomics/pathology/clinical
- `get_training_data(modal_list, modal_dict, df_mask, df_outcomes)` – aligns all modalities to common patient index, returns `(list_of_dfs, mask_df, outcomes_df)`

**Evaluation:**
- `auc_roc_ci(y_true, y_pred, alpha)` – AUC + DeLong confidence intervals

### NLP Clinical module: `clinical_nlp_embedding.py`

Standalone module. Converts 13 numeric clinical columns into 384-dim sentence embeddings.
- `df_to_text_prompts(df)` – builds English sentences from clinical row values
- `ClinicalTextEmbedder` – wraps `sentence-transformers/all-MiniLM-L6-v2`
- `prepare_nlp_clinical_modality(df_dict, df_labs, modality_mask, name)` – end-to-end integration

**Critical:** Always pass `no_scale=[idx]` for NLP modalities in `model_params`. The embeddings are cosine-normalized; RobustScaler will corrupt them.

## Typical experiment pattern

```python
import lung_helpers, importlib
importlib.reload(lung_helpers)
from lung_helpers import *

BASE_DB_DIR = '../datasets'

# 1. Load cohort and outcomes
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort'] == 'discovery']
df_clinical = get_clinical_table_v2(path=f"{BASE_DB_DIR}/clinical.csv",
                                    main_index_col='dmp_pt_id', cohort=df_cohort_disc)
df_outcomes = df_clinical[['label']].copy()

# 2. Load raw data
df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_...parquet")
df_genomic   = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")

# 3. Build modality_dict and modality_MASK
modality_dict = {}
modality_MASK = pd.DataFrame(index=df_outcomes.index)

df_rad_by_site = decorate_with_site_index(df_radiology)
prepare_rad_modality_by_size(modality_dict, df_rad_by_site, modality_MASK,
                              sites=['PC'], name='rad_lesion_pc')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, name='gen_driver_mut_amp')

# 4. Align and train
modal_list = ['rad_lesion_pc', 'gen_driver_mut_amp']
data, mask, labels = get_training_data(modal_list, modality_dict, modality_MASK, df_outcomes)

model_params = {'epochs': 100, 'lr': 0.01, 'alpha': 1.0, 'beta': 1.0,
                'cross_modality_enabled': False, 'attention_gate_enabled': True}
l1_filter = {0: {'l1_selection_df': df_radiology, 'kwargs': {'robustness_cutoff': 0.15, 'outlier_cutoff': 6}}}

summary_df, coef_df = train(data, mask, labels, l1_filter, model_params, folds=10)
auc, ci = auc_roc_ci(summary_df['label'].values, summary_df['score'].values, 0.95)
```

## Key data conventions

- Patient index column: `main_index` in all datasets
- Radiomics `job_tag`: `'filtered-radiomics'` (original) vs `'pertubation-radiomics'` (perturbations)
- Labels: `label=0` → PR/CR (response), `label=1` → SD/POD (no response)
- Parquet files may have case-variant filenames; scripts use try/except to handle both
- Output Excel files are written to `excel/` subdirectory (create if missing)

## Documentation

Full project documentation is in `README.md` (combined) with detailed references in `document/`:
- `document/kien-truc-mo-hinh.md` – model architecture detail
- `document/huong-dan-train.md` – `train()` walkthrough with concrete numbers
- `document/ovo-attention.md` – OvO variant, experimental results across 20 test cases
- `document/nlp-clinical.md` – NLP clinical embedding implementation guide
- `document/du-lieu-radiomics.md` – radiomics feature naming conventions
