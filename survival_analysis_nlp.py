"""
survival_analysis_nlp.py
Survival analysis cho NLP clinical embedding variants.
Thực hiện 5 bước theo survival-analysis-plan.md Prompt 4.

Run từ code/ directory:
    python survival_analysis_nlp.py
"""

import importlib
import warnings
warnings.filterwarnings("ignore")

import lung_helpers
importlib.reload(lung_helpers)
from lung_helpers import *

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 0. Config ─────────────────────────────────────────────────────────────────

BASE_DB_DIR = '../datasets'
OUT_DIR = 'vector_figs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load cohort & clinical ─────────────────────────────────────────────────

print("[1/6] Loading cohort and clinical data...")
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/FINAL_COHORT_LISTING.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort'] == 'discovery']

CLINICAL_CSV = (
    f"{BASE_DB_DIR}/"
    "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv"
)
df_clinical = get_clinical_table_v2(
    path=CLINICAL_CSV,
    main_index_col='dmp_pt_id',
    cohort=df_cohort_disc,
)
df_outcomes = df_clinical[['label']].copy(deep=True)

clinical_predictors = [
    'age', 'pack_years', 'ecog', 'albumin', 'dnlr',
    'brain_mets', 'liver_mets', 'tumor_burden', 'therapy_line',
    'recieves_combo_therapy', 'site_lung', 'recieves_pdl1_therapy', 'hist_adeno',
]
df_labs = df_clinical[clinical_predictors]

# ── 2. Load raw data ──────────────────────────────────────────────────────────

print("[2/6] Loading raw modality data...")
df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
df_tmb     = df_genomic[['TMB']]
df_nontmb  = df_genomic.loc[:, ~df_genomic.columns.str.contains("TMB")]
df_pdl1    = pd.read_parquet(f"{BASE_DB_DIR}/PDL1_SCORE.parquet")

df_radiology = pd.read_parquet(
    f"{BASE_DB_DIR}/LUNG_RADIOMICS_spacing1.0_MirpOn_Window1350.250_allImageTypes_bw20.parquet"
)
df_radiology_by_site = decorate_with_site_index(df_radiology)

df_texture = pd.read_parquet(
    f"{BASE_DB_DIR}/NEW_LUNG_glcm_Autocorrelation_v2_20x_stain1_PDL1.parquet"
)
df_glcm = pd.read_parquet(f"{BASE_DB_DIR}/LUNG_PATHOLOGY_PDL1_GLCM_V3.parquet")

# ── 3. Build modality_dict ────────────────────────────────────────────────────

print("[3/6] Building modality_dict...")
modality_MASK = df_clinical[[]]
modality_dict = {}

modality_PC_full = prepare_rad_modality_by_size(
    modality_dict, df_radiology_by_site, modality_MASK, 'PC', 'rad_lesion_pc')
modality_PL_full = prepare_rad_modality_by_size(
    modality_dict, df_radiology_by_site, modality_MASK, 'PL', 'rad_lesion_pl')
modality_LN_full = prepare_rad_modality_by_size(
    modality_dict, df_radiology_by_site, modality_MASK, 'LN', 'rad_lesion_ln')

prepare_other_modalities(modality_dict, df_texture, modality_MASK, 'path_ihc_pdl1')
prepare_other_modalities(modality_dict, df_glcm,    modality_MASK, 'path_ihc_glcm')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, 'gen_driver_mut_amp')
prepare_other_modalities(modality_dict, df_tmb,     modality_MASK, 'gen_driver_tmb')
prepare_other_modalities(modality_dict, df_nontmb,  modality_MASK, 'gen_driver_non_tmb')
prepare_other_modalities(modality_dict, df_pdl1,    modality_MASK, 'cnl_pdl1_score')
prepare_other_modalities(modality_dict, df_labs,    modality_MASK, 'cnl_dem_labs')

modality_dict['cnl_pdl1_score'] = -modality_dict['cnl_pdl1_score'] / 100.0
modality_MASK = modality_MASK.fillna(False)

# NLP embeddings
import clinical_nlp_embedding
importlib.reload(clinical_nlp_embedding)
from clinical_nlp_embedding import prepare_nlp_clinical_modality
from sklearn.decomposition import PCA

prepare_nlp_clinical_modality(modality_dict, df_labs, modality_MASK, 'cnl_nlp_embedding')

_pca = PCA(n_components=16, random_state=42)
_pca_vals = _pca.fit_transform(modality_dict['cnl_nlp_embedding'].values)
modality_dict['cnl_nlp_pca16'] = pd.DataFrame(
    _pca_vals, index=modality_dict['cnl_nlp_embedding'].index
)
modality_MASK['cnl_nlp_pca16'] = modality_MASK['cnl_nlp_embedding']

# ── 4. Train 8 model variants ─────────────────────────────────────────────────

print("[4/6] Training 8 DyAM model variants (10-fold CV each)...")

importlib.reload(lung_helpers)
from lung_helpers import *

model_params = {
    'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0,
    'cross_modality_enabled': False,
}
dfs_rad_filters = {
    0: {'l1_selection_df': modality_PC_full, 'kwargs': {'l1_strength': 0.1}},
    1: {'l1_selection_df': modality_PL_full, 'kwargs': {'l1_strength': 0.1}},
    2: {'l1_selection_df': modality_LN_full, 'kwargs': {'l1_strength': 0.1}},
}

BASE_MODS = ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln',
             'gen_driver_mut_amp', 'cnl_pdl1_score']

summary_dfs = {}

# IHC-A arm (path_ihc_pdl1 = texture features)
print("  Training IHC-A variants...")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_pdl1'], modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params)
print("    DyAM Rad+IHC-A+Gen+PDL1 done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_pdl1', 'cnl_dem_labs'], modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1+Labs'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params)
print("    DyAM Rad+IHC-A+Gen+PDL1+Labs done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_pdl1', 'cnl_nlp_embedding'], modality_dict, modality_MASK, df_outcomes)
nlp_idx = list(mask.columns).index('cnl_nlp_embedding')
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1+NLP'], _ = train(
    data, mask, labels, dfs_rad_filters, {**model_params, 'no_scale': [nlp_idx]})
print("    DyAM Rad+IHC-A+Gen+PDL1+NLP done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_pdl1', 'cnl_nlp_pca16'], modality_dict, modality_MASK, df_outcomes)
pca_idx = list(mask.columns).index('cnl_nlp_pca16')
summary_dfs['DyAM Rad+IHC-A+Gen+PDL1+NLP-PCA16'], _ = train(
    data, mask, labels, dfs_rad_filters, {**model_params, 'no_scale': [pca_idx]})
print("    DyAM Rad+IHC-A+Gen+PDL1+NLP-PCA16 done")

# IHC-G arm (path_ihc_glcm = GLCM features)
print("  Training IHC-G variants...")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_glcm'], modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-G+Gen+PDL1'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params)
print("    DyAM Rad+IHC-G+Gen+PDL1 done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_glcm', 'cnl_dem_labs'], modality_dict, modality_MASK, df_outcomes)
summary_dfs['DyAM Rad+IHC-G+Gen+PDL1+Labs'], _ = train(
    data, mask, labels, dfs_rad_filters, model_params)
print("    DyAM Rad+IHC-G+Gen+PDL1+Labs done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_glcm', 'cnl_nlp_embedding'], modality_dict, modality_MASK, df_outcomes)
nlp_idx = list(mask.columns).index('cnl_nlp_embedding')
summary_dfs['DyAM Rad+IHC-G+Gen+PDL1+NLP'], _ = train(
    data, mask, labels, dfs_rad_filters, {**model_params, 'no_scale': [nlp_idx]})
print("    DyAM Rad+IHC-G+Gen+PDL1+NLP done")

data, mask, labels = get_training_data(
    BASE_MODS + ['path_ihc_glcm', 'cnl_nlp_pca16'], modality_dict, modality_MASK, df_outcomes)
pca_idx = list(mask.columns).index('cnl_nlp_pca16')
summary_dfs['DyAM Rad+IHC-G+Gen+PDL1+NLP-PCA16'], _ = train(
    data, mask, labels, dfs_rad_filters, {**model_params, 'no_scale': [pca_idx]})
print("    DyAM Rad+IHC-G+Gen+PDL1+NLP-PCA16 done")

# ── 5. Survival analysis ──────────────────────────────────────────────────────

print("[5/6] Running survival analyses...")

MODELS = {
    'No Clinical (IHC-A)':   'DyAM Rad+IHC-A+Gen+PDL1',
    '+ Labs (IHC-A)':        'DyAM Rad+IHC-A+Gen+PDL1+Labs',
    '+ NLP raw (IHC-A)':     'DyAM Rad+IHC-A+Gen+PDL1+NLP',
    '+ NLP-PCA16 (IHC-A)':   'DyAM Rad+IHC-A+Gen+PDL1+NLP-PCA16',
    'No Clinical (IHC-G)':   'DyAM Rad+IHC-G+Gen+PDL1',
    '+ Labs (IHC-G)':        'DyAM Rad+IHC-G+Gen+PDL1+Labs',
    '+ NLP raw (IHC-G)':     'DyAM Rad+IHC-G+Gen+PDL1+NLP',
    '+ NLP-PCA16 (IHC-G)':   'DyAM Rad+IHC-G+Gen+PDL1+NLP-PCA16',
}

# ── Bước 1: C-index + bootstrap 95% CI ────────────────────────────────────────

print("  Step 1: C-index with bootstrap CI...")
cindex_results = {}
for label, key in MODELS.items():
    ci, lo, hi = compute_cindex_bootstrap(summary_dfs[key], df_clinical)
    cindex_results[label] = {'c_index': ci, 'ci_lower': lo, 'ci_upper': hi}
    print(f"    {label:<30} C={ci:.3f} [{lo:.3f}–{hi:.3f}]")

# SA-1: C-index bar chart
fig, ax = plt.subplots(figsize=(10, 5))
labels_list  = list(cindex_results.keys())
c_vals = [cindex_results[k]['c_index'] for k in labels_list]
c_lo   = [cindex_results[k]['c_index'] - cindex_results[k]['ci_lower'] for k in labels_list]
c_hi   = [cindex_results[k]['ci_upper'] - cindex_results[k]['c_index'] for k in labels_list]
colors = ['#4878CF'] * 4 + ['#6ACC65'] * 4
x = np.arange(len(labels_list))
bars = ax.bar(x, c_vals, color=colors, alpha=0.8, width=0.6)
ax.errorbar(x, c_vals, yerr=[c_lo, c_hi], fmt='none', color='black', capsize=4, linewidth=1.2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Random (C=0.5)')
ax.set_xticks(x)
ax.set_xticklabels(labels_list, rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Harrell's C-index")
ax.set_ylim(0.45, 0.95)
ax.set_title("SA-1: C-index Comparison (Bootstrap 95% CI)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/SA1_cindex_comparison.svg", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/SA1_cindex_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved SA1_cindex_comparison")

# ── Bước 2: Paired Wilcoxon test (per-fold C-index) ───────────────────────────

print("  Step 2: Paired Wilcoxon test (per-fold C-index)...")
wilcoxon_results = {}
for ihc in ['IHC-A', 'IHC-G']:
    key_base = f'DyAM Rad+{ihc}+Gen+PDL1'
    for variant in ['Labs', 'NLP', 'NLP-PCA16']:
        key_b = f'DyAM Rad+{ihc}+Gen+PDL1+{variant}'
        stat, p, ci_a, ci_b = paired_cindex_test(
            summary_dfs[key_base], summary_dfs[key_b], df_clinical
        )
        delta = np.mean(ci_b) - np.mean(ci_a)
        tag = f'{ihc} vs +{variant}'
        wilcoxon_results[tag] = {'delta_c': delta, 'p_value': p, 'stat': stat}
        sig = '*' if p < 0.05 else ''
        print(f"    {tag:<30} ΔC={delta:+.4f}  p={p:.4f} {sig}")

# SA-4: Per-fold C-index boxplot (NoClinical vs NLP-PCA16 per arm)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for ax, ihc in zip(axes, ['IHC-A', 'IHC-G']):
    key_base = f'DyAM Rad+{ihc}+Gen+PDL1'
    fold_base = compute_cindex_per_fold(summary_dfs[key_base], df_clinical)['per_fold']
    fold_nlp  = compute_cindex_per_fold(summary_dfs[f'{key_base}+NLP-PCA16'], df_clinical)['per_fold']
    fold_data = [fold_base, fold_nlp]
    bp = ax.boxplot(fold_data, patch_artist=True, widths=0.5)
    colors_bp = ['#4878CF', '#E87E4D'] if ihc == 'IHC-A' else ['#6ACC65', '#E87E4D']
    for patch, c in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, (a_val, b_val) in enumerate(zip(fold_base, fold_nlp)):
        ax.plot([1, 2], [a_val, b_val], color='gray', alpha=0.4, linewidth=0.8)
    p_val = wilcoxon_results[f'{ihc} vs +NLP-PCA16']['p_value']
    ax.set_xticklabels(['No Clinical', '+NLP-PCA16'])
    ax.set_title(f"{ihc}  (Wilcoxon p={p_val:.3f})")
    ax.set_ylabel("C-index per fold")
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.7)
plt.suptitle("SA-4: Per-fold C-index (No Clinical vs NLP-PCA16)", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/SA4_perfold_cindex_boxplot.svg", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/SA4_perfold_cindex_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved SA4_perfold_cindex_boxplot")

# ── Bước 3: Time-dependent AUC ────────────────────────────────────────────────

print("  Step 3: Time-dependent AUC at 6m/12m/18m...")
tdauc_results = {}
for label, key in MODELS.items():
    try:
        tdauc_results[label] = compute_tdauc(summary_dfs[key], df_clinical, times=[6, 12, 18])
    except Exception as e:
        print(f"    Warning ({label}): {e}")

# Print table
print(f"\n    {'Model':<30} {'6m':>6} {'12m':>6} {'18m':>6} {'Mean':>6}")
print("    " + "-" * 56)
for label, r in tdauc_results.items():
    print(f"    {label:<30} "
          f"{r.get(6, float('nan')):>6.3f} "
          f"{r.get(12, float('nan')):>6.3f} "
          f"{r.get(18, float('nan')):>6.3f} "
          f"{r.get('mean_auc', float('nan')):>6.3f}")

# SA-2: Time-dependent AUC line chart (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, ihc, color_base in zip(axes, ['IHC-A', 'IHC-G'], ['blues', 'greens']):
    arm_models = {k: v for k, v in tdauc_results.items() if ihc in k}
    for name, result in arm_models.items():
        pts = [t for t in [6, 12, 18] if t in result]
        ys  = [result[t] for t in pts]
        ax.plot(pts, ys, marker='o', label=name.replace(f' ({ihc})', ''))
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Time-Dependent AUC')
    ax.set_xticks([6, 12, 18])
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7)
    ax.set_title(f"SA-2: Time-Dep AUC — {ihc} arm")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/SA2_tdauc_comparison.svg", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/SA2_tdauc_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved SA2_tdauc_comparison")

# ── Bước 4: Multivariate Cox ──────────────────────────────────────────────────

print("  Step 4: Multivariate Cox PH...")
cox_results = {}
for label, key in MODELS.items():
    try:
        cph = compute_multivariate_cox(summary_dfs[key], df_clinical)
        cox_results[label] = cph
        row = cph.summary.loc['score']
        p_str = f"{row['p']:.3f}" if row['p'] >= 0.001 else "<0.001"
        print(f"    {label:<30} HR={row['exp(coef)']:.2f} "
              f"[{row['exp(coef) lower 95%']:.2f}–{row['exp(coef) upper 95%']:.2f}] p={p_str}")
    except Exception as e:
        print(f"    Warning ({label}): {e}")

# SA-3: Forest plot
if cox_results:
    ax3 = generate_forest_plot(cox_results, panel="SA-3: Multivariate Cox HR (score adjusted for age/ECOG/albumin/dNLR/liver_mets)")
    plt.savefig(f"{OUT_DIR}/SA3_forest_plot_multivariate.svg", bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/SA3_forest_plot_multivariate.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved SA3_forest_plot_multivariate")

# ── Bước 5: Brier Score / IBS ─────────────────────────────────────────────────

print("  Step 5: Integrated Brier Score (IBS)...")
brier_results = {}
for label, key in MODELS.items():
    try:
        times_arr, brier_vals, ibs = compute_brier_score_curve(summary_dfs[key], df_clinical)
        brier_results[label] = (times_arr, brier_vals, ibs)
        print(f"    {label:<30} IBS={ibs:.4f}")
    except Exception as e:
        print(f"    Warning ({label}): {e}")

# SA-5: Brier score over time
if brier_results:
    ax5 = generate_brier_score_plot(brier_results, panel="SA-5: Brier Score Over Time")
    plt.savefig(f"{OUT_DIR}/SA5_brier_score_curve.svg", bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/SA5_brier_score_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved SA5_brier_score_curve")

# ── 6. Summary table ─────────────────────────────────────────────────────────

print("\n[6/6] Summary table")
print()

rows = []
for label, key in MODELS.items():
    auc_binary, auc_ci = auc_roc_ci(
        summary_dfs[key]['label'].values,
        summary_dfs[key]['score'].values,
        0.95
    )
    ci_r = cindex_results.get(label, {})
    tdauc_r = tdauc_results.get(label, {})
    cox_row = cox_results[label].summary.loc['score'] if label in cox_results else None
    brier_r = brier_results.get(label, (None, None, None))

    rows.append({
        'Model': label,
        'AUC (binary)': f"{auc_binary:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}]",
        'C-index': f"{ci_r.get('c_index', float('nan')):.3f} [{ci_r.get('ci_lower', float('nan')):.3f}–{ci_r.get('ci_upper', float('nan')):.3f}]",
        'tdAUC-6m': f"{tdauc_r.get(6, float('nan')):.3f}",
        'tdAUC-12m': f"{tdauc_r.get(12, float('nan')):.3f}",
        'tdAUC-18m': f"{tdauc_r.get(18, float('nan')):.3f}",
        'Cox HR': f"{cox_row['exp(coef)']:.2f}" if cox_row is not None else 'N/A',
        'Cox p': f"{cox_row['p']:.3f}" if cox_row is not None else 'N/A',
        'IBS': f"{brier_r[2]:.4f}" if brier_r[2] is not None else 'N/A',
    })

df_summary = pd.DataFrame(rows).set_index('Model')
print(df_summary.to_string())

os.makedirs('excel', exist_ok=True)
df_summary.to_excel('excel/survival_analysis_summary.xlsx')
print("\nSaved excel/survival_analysis_summary.xlsx")

print("\n=== Wilcoxon paired C-index test (NoClinical vs NLP-PCA16) ===")
for ihc in ['IHC-A', 'IHC-G']:
    r = wilcoxon_results.get(f'{ihc} vs +NLP-PCA16', {})
    sig = ' *SIGNIFICANT*' if r.get('p_value', 1) < 0.05 else ''
    print(f"  {ihc}: ΔC={r.get('delta_c', 0):+.4f}  p={r.get('p_value', 1):.4f}{sig}")

print("\nDone. Figures saved to:", OUT_DIR)

# ── 7. External Validation ────────────────────────────────────────────────────
# PATH-VAL (n=71, 52 có pathology): IHC-A / IHC-G ± NLP-PCA16
# RAD-VAL  (n=50, 46 có radiomics): Rad-only ± NLP-PCA16
# Mỗi model: train trên toàn bộ discovery → predict trên val → C-index
# Note: val cohort không có genomics/PDL1 → chỉ validate modality thực sự có sẵn

print("\n" + "="*60)
print("[7/6] External Validation")
print("="*60)

# ── 7.1 Load validation cohorts ───────────────────────────────────────────────

df_cohort_rad_val  = df_cohort[df_cohort['cohort'] == 'rad_valid']
df_cohort_path_val = df_cohort[df_cohort['cohort'] == 'path_valid']

df_clinical_rad_val  = get_clinical_table_v2(
    path=CLINICAL_CSV, main_index_col='did_acc',        cohort=df_cohort_rad_val)
df_clinical_path_val = get_clinical_table_v2(
    path=CLINICAL_CSV, main_index_col='pdl1_image_id',  cohort=df_cohort_path_val)

df_outcomes_rad_val  = df_clinical_rad_val[['label']].copy()
df_outcomes_path_val = df_clinical_path_val[['label']].copy()

# Radiology validation data
df_radiology_valid = pd.read_parquet(
    f"{BASE_DB_DIR}/LUNG_RADIOMICS_spacing1.0_MirpOn_Window1350.250_allImageTypes_bw20_VALIDATION.parquet"
)
df_radiology_valid_by_site = decorate_with_site_index(df_radiology_valid)

# Pathology validation data
df_texture_valid = pd.read_parquet(
    f"{BASE_DB_DIR}/NEW_LUNG_glcm_Autocorrelation_v2_20x_stain1_PDL1_VALIDATION.parquet"
)
df_glcm_valid = pd.read_parquet(
    f"{BASE_DB_DIR}/LUNG_PATHOLOGY_PDL1_GLCM_V3_VALIDATION.parquet"
)

# ── 7.2 Build modality_dict cho từng val cohort ───────────────────────────────

# PATH-VAL modalities
modality_mask_path_val = df_clinical_path_val[[]]
modality_dict_path_val = {}
prepare_other_modalities(modality_dict_path_val, df_texture_valid, modality_mask_path_val, 'path_ihc_pdl1')
prepare_other_modalities(modality_dict_path_val, df_glcm_valid,    modality_mask_path_val, 'path_ihc_glcm')
modality_mask_path_val = modality_mask_path_val.fillna(False)

# NLP cho PATH-VAL (dùng cùng PCA đã fit trên discovery — KHÔNG fit lại)
df_labs_path_val = df_clinical_path_val[[c for c in clinical_predictors if c in df_clinical_path_val.columns]]
prepare_nlp_clinical_modality(modality_dict_path_val, df_labs_path_val, modality_mask_path_val, 'cnl_nlp_embedding')
_pca16_path_val = _pca.transform(modality_dict_path_val['cnl_nlp_embedding'].values)
modality_dict_path_val['cnl_nlp_pca16'] = pd.DataFrame(
    _pca16_path_val, index=modality_dict_path_val['cnl_nlp_embedding'].index)
modality_mask_path_val['cnl_nlp_pca16'] = modality_mask_path_val['cnl_nlp_embedding']

# RAD-VAL modalities
modality_mask_rad_val = df_clinical_rad_val[[]]
modality_dict_rad_val = {}
modality_PC_full_val = prepare_rad_modality_by_size(
    modality_dict_rad_val, df_radiology_valid_by_site, modality_mask_rad_val, 'PC', 'rad_lesion_pc')
modality_PL_full_val = prepare_rad_modality_by_size(
    modality_dict_rad_val, df_radiology_valid_by_site, modality_mask_rad_val, 'PL', 'rad_lesion_pl')
modality_LN_full_val = prepare_rad_modality_by_size(
    modality_dict_rad_val, df_radiology_valid_by_site, modality_mask_rad_val, 'LN', 'rad_lesion_ln')
modality_mask_rad_val = modality_mask_rad_val.fillna(False)

# NLP cho RAD-VAL
df_labs_rad_val = df_clinical_rad_val[[c for c in clinical_predictors if c in df_clinical_rad_val.columns]]
prepare_nlp_clinical_modality(modality_dict_rad_val, df_labs_rad_val, modality_mask_rad_val, 'cnl_nlp_embedding')
_pca16_rad_val = _pca.transform(modality_dict_rad_val['cnl_nlp_embedding'].values)
modality_dict_rad_val['cnl_nlp_pca16'] = pd.DataFrame(
    _pca16_rad_val, index=modality_dict_rad_val['cnl_nlp_embedding'].index)
modality_mask_rad_val['cnl_nlp_pca16'] = modality_mask_rad_val['cnl_nlp_embedding']

# ── 7.4 PATH-VAL: IHC-A / IHC-G ± NLP-PCA16 ─────────────────────────────────

print("\n--- PATH-VAL (n=71, 52 có pathology) ---")

# anchor_mod = imaging modality xác định tập bệnh nhân chuẩn cho so sánh fair.
# Mọi model trong cùng nhóm đều bị restrict về đúng tập bệnh nhân có anchor_mod.
path_val_models = {
    'IHC-A only (Path-Val)':        (['path_ihc_pdl1'],                    'path_ihc_pdl1'),
    'IHC-A + NLP-PCA16 (Path-Val)': (['path_ihc_pdl1', 'cnl_nlp_pca16'],  'path_ihc_pdl1'),
    'IHC-G only (Path-Val)':        (['path_ihc_glcm'],                    'path_ihc_glcm'),
    'IHC-G + NLP-PCA16 (Path-Val)': (['path_ihc_glcm', 'cnl_nlp_pca16'],  'path_ihc_glcm'),
}

val_cindex_path = {}
for label, (val_mods, anchor_mod) in path_val_models.items():
    try:
        # Tập bệnh nhân chuẩn: chỉ những ai có anchor imaging modality
        anchor_px = modality_mask_path_val[
            modality_mask_path_val[anchor_mod].astype(bool)
        ].index

        data_d, m_d, l_d = get_training_data(val_mods, modality_dict,         modality_MASK,         df_outcomes)
        data_v, m_v, l_v = get_training_data(val_mods, modality_dict_path_val, modality_mask_path_val, df_outcomes_path_val)

        data_comb = [pd.concat([d, v]) for d, v in zip(data_d, data_v)]
        mask_comb = pd.concat([m_d, m_v])
        lbls_comb = pd.concat([l_d, l_v])

        mp = model_params.copy()
        if 'cnl_nlp_pca16' in val_mods:
            pca_col_idx = list(m_d.columns).index('cnl_nlp_pca16')
            mp = {**model_params, 'no_scale': [pca_col_idx]}

        sdf_val = train_eval_all(
            data_comb, mask_comb, lbls_comb,
            {}, mp,
            train_px=l_d.index,
            valid_px=l_v.index,
        )

        # FIX: restrict về đúng tập bệnh nhân có anchor modality → fair comparison
        sdf_val = sdf_val[sdf_val.index.isin(anchor_px)]

        ci, lo, hi = compute_cindex_bootstrap(sdf_val, df_clinical_path_val)
        auc_v, _ = auc_roc_ci(sdf_val['label'].values, sdf_val['score'].values, 0.95)
        val_cindex_path[label] = {'c_index': ci, 'ci_lower': lo, 'ci_upper': hi,
                                   'auc': auc_v, 'n_val': len(sdf_val)}
        print(f"  {label:<40} C={ci:.3f} [{lo:.3f}–{hi:.3f}]  AUC={auc_v:.3f}  n={len(sdf_val)}")
    except Exception as e:
        print(f"  WARNING ({label}): {e}")

# ── 7.5 RAD-VAL: Rad-only ± NLP-PCA16 ────────────────────────────────────────

print("\n--- RAD-VAL (n=50, 46 có radiomics) ---")

dfs_rad_filters_val = {
    0: {'l1_selection_df': modality_PC_full, 'kwargs': {'l1_strength': 0.1}},
    1: {'l1_selection_df': modality_PL_full, 'kwargs': {'l1_strength': 0.1}},
    2: {'l1_selection_df': modality_LN_full, 'kwargs': {'l1_strength': 0.1}},
}

# anchor = bệnh nhân có ít nhất 1 trong 3 lesion type (PC/PL/LN)
rad_val_models = {
    'Rad only (Rad-Val)':        ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln'],
    'Rad + NLP-PCA16 (Rad-Val)': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'cnl_nlp_pca16'],
}

# Bệnh nhân có ít nhất 1 rad lesion modality
rad_anchor_px = modality_mask_rad_val[
    modality_mask_rad_val[['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln']].any(axis=1)
].index

val_cindex_rad = {}
for label, mods in rad_val_models.items():
    try:
        data_d, m_d, l_d = get_training_data(mods, modality_dict,        modality_MASK,        df_outcomes)
        data_v, m_v, l_v = get_training_data(mods, modality_dict_rad_val, modality_mask_rad_val, df_outcomes_rad_val)

        data_comb = [pd.concat([d, v]) for d, v in zip(data_d, data_v)]
        mask_comb = pd.concat([m_d, m_v])
        lbls_comb = pd.concat([l_d, l_v])

        mp = model_params.copy()
        if 'cnl_nlp_pca16' in mods:
            pca_col_idx = list(m_d.columns).index('cnl_nlp_pca16')
            mp = {**model_params, 'no_scale': [pca_col_idx]}

        sdf_val = train_eval_all(
            data_comb, mask_comb, lbls_comb,
            dfs_rad_filters_val, mp,
            train_px=l_d.index,
            valid_px=l_v.index,
        )

        # FIX: restrict về đúng tập bệnh nhân có radiomics → fair comparison
        sdf_val = sdf_val[sdf_val.index.isin(rad_anchor_px)]

        ci, lo, hi = compute_cindex_bootstrap(sdf_val, df_clinical_rad_val)
        auc_v, _ = auc_roc_ci(sdf_val['label'].values, sdf_val['score'].values, 0.95)
        val_cindex_rad[label] = {'c_index': ci, 'ci_lower': lo, 'ci_upper': hi,
                                  'auc': auc_v, 'n_val': len(sdf_val)}
        print(f"  {label:<40} C={ci:.3f} [{lo:.3f}–{hi:.3f}]  AUC={auc_v:.3f}  n={len(sdf_val)}")
    except Exception as e:
        print(f"  WARNING ({label}): {e}")

# ── 7.6 Figure SA-6: External Validation C-index ─────────────────────────────

all_val = {**val_cindex_path, **val_cindex_rad}
if all_val:
    fig, ax = plt.subplots(figsize=(10, 4))
    val_labels = list(all_val.keys())
    vc  = [all_val[k]['c_index']  for k in val_labels]
    vlo = [all_val[k]['c_index'] - all_val[k]['ci_lower'] for k in val_labels]
    vhi = [all_val[k]['ci_upper'] - all_val[k]['c_index'] for k in val_labels]
    val_colors = ['#4878CF', '#E87E4D', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
    x = np.arange(len(val_labels))
    ax.bar(x, vc, color=val_colors[:len(val_labels)], alpha=0.8, width=0.6)
    ax.errorbar(x, vc, yerr=[vlo, vhi], fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Random (C=0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(val_labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Harrell's C-index (External Validation)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("SA-6: External Validation C-index (Bootstrap 95% CI)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/SA6_external_validation_cindex.svg", bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/SA6_external_validation_cindex.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved SA6_external_validation_cindex")

# ── 7.7 Summary bảng internal vs external ────────────────────────────────────

print("\n=== Internal (CV) vs External Validation C-index ===")
print(f"{'Model':<42} {'Internal C':>10} {'External C':>10} {'ΔC':>7}")
print("-" * 72)

# Internal discovery models (pooled C-index)
internal_map = {
    'IHC-A only':        ('No Clinical (IHC-A)', 'IHC-A only (Path-Val)',        val_cindex_path),
    'IHC-A + NLP-PCA16': ('+ NLP-PCA16 (IHC-A)', 'IHC-A + NLP-PCA16 (Path-Val)', val_cindex_path),
    'IHC-G only':        ('No Clinical (IHC-G)', 'IHC-G only (Path-Val)',        val_cindex_path),
    'IHC-G + NLP-PCA16': ('+ NLP-PCA16 (IHC-G)', 'IHC-G + NLP-PCA16 (Path-Val)', val_cindex_path),
    'Rad only':          (None,                   'Rad only (Rad-Val)',            val_cindex_rad),
    'Rad + NLP-PCA16':   (None,                   'Rad + NLP-PCA16 (Rad-Val)',     val_cindex_rad),
}

for short, (int_key, ext_key, ext_dict) in internal_map.items():
    int_c = cindex_results.get(int_key, {}).get('c_index', float('nan')) if int_key else float('nan')
    ext_c = ext_dict.get(ext_key, {}).get('c_index', float('nan'))
    delta = ext_c - int_c if not (pd.isna(int_c) or pd.isna(ext_c)) else float('nan')
    int_str = f"{int_c:.3f}" if not pd.isna(int_c) else "  N/A"
    ext_str = f"{ext_c:.3f}" if not pd.isna(ext_c) else "  N/A"
    dlt_str = f"{delta:+.3f}" if not pd.isna(delta) else "   N/A"
    print(f"  {short:<40} {int_str:>10} {ext_str:>10} {dlt_str:>7}")

# Save external val to Excel
ext_rows = []
for k, v in {**val_cindex_path, **val_cindex_rad}.items():
    ext_rows.append({
        'Model (External Val)': k,
        'n_val': v['n_val'],
        'C-index': round(v['c_index'], 3),
        'CI lower': round(v['ci_lower'], 3),
        'CI upper': round(v['ci_upper'], 3),
        'AUC (binary)': round(v['auc'], 3),
    })
df_ext = pd.DataFrame(ext_rows).set_index('Model (External Val)')
df_ext.to_excel('excel/external_validation_cindex.xlsx')
print("\nSaved excel/external_validation_cindex.xlsx")
print("\nAll done.")
