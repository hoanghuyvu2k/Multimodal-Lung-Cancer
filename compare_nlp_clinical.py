import sys
import pandas as pd
import numpy as np
from pathlib import Path
import importlib

# Import lung_helpers
import lung_helpers
importlib.reload(lung_helpers)
from lung_helpers import (
    train, auc_roc_ci, 
    get_clinical_table_v2, prepare_rad_modality_by_size, prepare_rad_modality,
    prepare_other_modalities, decorate_with_site_index
)

from clinical_nlp_embedding import prepare_nlp_clinical_modality

print("=" * 80)
print("SO SÁNH MÔ HÌNH ATTENTION GỐC VÀ MÔ HÌNH TÍCH HỢP NLP CLINICAL EMBEDDING")
print("=" * 80)

# Thiết lập đường dẫn dữ liệu
BASE_DB_DIR = '../datasets'

# Load dữ liệu cohort
print("\n[1/6] Đang load dữ liệu cohort...")
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort']=='discovery']
print(f"   - Discovery cohort: {len(df_cohort_disc)} samples")

# Load clinical data
print("\n[2/6] Đang load clinical data...")
df_clinical = get_clinical_table_v2(
    path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv", 
    main_index_col='dmp_pt_id',
    cohort=df_cohort_disc
)
df_outcomes = df_clinical[['label']].copy(deep=True)
print(f"   - Outcomes: {len(df_outcomes)} samples")
print(f"   - Class distribution: {df_outcomes['label'].value_counts().to_dict()}")

# Load các modality
print("\n[3/6] Đang load các modality...")
clinical_predictors = ['age', 'pack_years', 'ecog', 'albumin', 'dnlr', 
                       'brain_mets', 'liver_mets', 'tumor_burden', 'therapy_line', 
                       'recieves_combo_therapy', 'site_lung', 'recieves_pdl1_therapy', 'hist_adeno']

try:
    df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
    df_tmb = df_genomic[['TMB']]
    df_nontmb = df_genomic.loc[:, ~df_genomic.columns.str.contains("TMB")]
except:
    print("   ⚠ Không tìm thấy file genomic_data_v3")
    df_genomic = pd.DataFrame()
    df_tmb = pd.DataFrame()
    df_nontmb = pd.DataFrame()

try:
    df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/pdl1_score.parquet")
except:
    try:
        df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/PDL1_SCORE.parquet")
    except:
        print("   ⚠ Không tìm thấy file pdl1_score")
        df_pdl1 = pd.DataFrame()
        
df_labs = df_clinical[clinical_predictors]

# Load radiomics
try:
    df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")
except:
    try:
        df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/LUNG_RADIOMICS_spacing1.0_MirpOn_Window1350.250_allImageTypes_bw20.parquet")
    except:
        print("   ⚠ Không tìm thấy file radiomics, sử dụng file trong code/")
        df_radiology = pd.read_parquet(f"lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")

df_radiology_by_site = decorate_with_site_index(df_radiology)

try:
    df_texture = pd.read_parquet(f"{BASE_DB_DIR}/new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet")
except:
    try:
        df_texture = pd.read_parquet(f"{BASE_DB_DIR}/NEW_LUNG_glcm_Autocorrelation_v2_20x_stain1_PDL1.parquet")
    except:
        print("   ⚠ Không tìm thấy file texture")
        df_texture = pd.DataFrame()

try:
    df_glcm = pd.read_parquet(f"{BASE_DB_DIR}/lung_pathology_pdl1_glcm_v3.parquet")
except:
    try:
        df_glcm = pd.read_parquet(f"{BASE_DB_DIR}/LUNG_PATHOLOGY_PDL1_GLCM_V3.parquet")
    except:
        print("   ⚠ Không tìm thấy file glcm")
        df_glcm = pd.DataFrame()

## ================== CHUẨN BỊ MẢNG GỐC =================
print("\n[4/6] Đang chuẩn bị modality mask (GỐC)...")
modality_MASK_original = df_clinical[[]].copy()
modality_dict_original = {}

# Radiology modalities
modality_PC_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK_original, 'PC', 'rad_lesion_pc')
modality_PL_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK_original, 'PL', 'rad_lesion_pl')
modality_LN_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK_original, 'LN', 'rad_lesion_ln')

# Other modalities
if len(df_texture) > 0: prepare_other_modalities(modality_dict_original, df_texture, modality_MASK_original, 'path_ihc_pdl1')
if len(df_glcm) > 0: prepare_other_modalities(modality_dict_original, df_glcm, modality_MASK_original, 'path_ihc_glcm')
if len(df_genomic) > 0: prepare_other_modalities(modality_dict_original, df_genomic, modality_MASK_original, 'gen_driver_mut_amp')
if len(df_nontmb) > 0: prepare_other_modalities(modality_dict_original, df_nontmb, modality_MASK_original, 'gen_driver_non_tmb')
if len(df_tmb) > 0: prepare_other_modalities(modality_dict_original, df_tmb, modality_MASK_original, 'gen_driver_tmb')
if len(df_pdl1) > 0: prepare_other_modalities(modality_dict_original, df_pdl1, modality_MASK_original, 'cnl_pdl1_score')

prepare_other_modalities(modality_dict_original, df_labs, modality_MASK_original, 'cnl_dem_labs')

if 'cnl_pdl1_score' in modality_dict_original:
    modality_dict_original['cnl_pdl1_score'] = - modality_dict_original['cnl_pdl1_score'] / 100.0
modality_MASK_original = modality_MASK_original.fillna(False)

modality_list_original = [modality_dict_original[key] for key in modality_MASK_original.columns]

## ================== CHUẨN BỊ MẢNG NLP =================
print("\n[5/6] Đang chuẩn bị modality mask (NLP CLINICAL)...")
modality_MASK_nlp = df_clinical[[]].copy()
modality_dict_nlp = {}

# Copy Radiology modalities to NLP dict
prepare_rad_modality_by_size(modality_dict_nlp, df_radiology_by_site, modality_MASK_nlp, 'PC', 'rad_lesion_pc')
prepare_rad_modality_by_size(modality_dict_nlp, df_radiology_by_site, modality_MASK_nlp, 'PL', 'rad_lesion_pl')
prepare_rad_modality_by_size(modality_dict_nlp, df_radiology_by_site, modality_MASK_nlp, 'LN', 'rad_lesion_ln')

# Copy other modalities
if len(df_texture) > 0: prepare_other_modalities(modality_dict_nlp, df_texture, modality_MASK_nlp, 'path_ihc_pdl1')
if len(df_glcm) > 0: prepare_other_modalities(modality_dict_nlp, df_glcm, modality_MASK_nlp, 'path_ihc_glcm')
if len(df_genomic) > 0: prepare_other_modalities(modality_dict_nlp, df_genomic, modality_MASK_nlp, 'gen_driver_mut_amp')
if len(df_nontmb) > 0: prepare_other_modalities(modality_dict_nlp, df_nontmb, modality_MASK_nlp, 'gen_driver_non_tmb')
if len(df_tmb) > 0: prepare_other_modalities(modality_dict_nlp, df_tmb, modality_MASK_nlp, 'gen_driver_tmb')
if len(df_pdl1) > 0: prepare_other_modalities(modality_dict_nlp, df_pdl1, modality_MASK_nlp, 'cnl_pdl1_score')

if 'cnl_pdl1_score' in modality_dict_nlp:
    modality_dict_nlp['cnl_pdl1_score'] = - modality_dict_nlp['cnl_pdl1_score'] / 100.0

# THAY THẾ df_labs (cnl_dem_labs) BẰNG NLP EMBEDDINGS (NLP Clinical)
prepare_nlp_clinical_modality(modality_dict_nlp, df_labs, modality_MASK_nlp, 'cnl_nlp_embedding')

modality_MASK_nlp = modality_MASK_nlp.fillna(False)

# Lưu ý: Tìm vị trí index của 'cnl_nlp_embedding' trong mask để pass vào cấu hình `no_scale` của mô hình (không scale raw NLP vector)
nlp_modality_index = list(modality_MASK_nlp.columns).index('cnl_nlp_embedding')

modality_list_nlp = [modality_dict_nlp[key] for key in modality_MASK_nlp.columns]


# Thiết lập tham số mô hình
print("\n[6/6] Thiết lập tham số mô hình & Bắt đầu huấn luyện...")
model_params_original = {
    'epochs': 100,
    'alpha': 1.0,
    'beta': 1.0,
    'lr': 0.01,
    'attention_gate_enabled': True
}

# Tham số cho model nlp: no_scale cnl_nlp_embedding
model_params_nlp = {
    'epochs': 100,
    'alpha': 1.0,
    'beta': 1.0,
    'lr': 0.01,
    'attention_gate_enabled': True,
    'no_scale': [nlp_modality_index] 
}

l1_dfs_filter = {}  
folds = 5  

# -------- EXPERIMENT 1: CHẠY MÔ HÌNH ATTENTION GỐC --------
print("\n" + "=" * 80)
print("1. CHẠY MÔ HÌNH ATTENTION GỐC (Dữ liệu Lâm sàng Số)")
print("=" * 80)
summary_df_original, coef_df_original = train(
    modality_list_in=modality_list_original,
    modality_mask=modality_MASK_original,
    outcomes=df_outcomes,
    l1_dfs_filter=l1_dfs_filter,
    model_params=model_params_original,
    folds=folds
)

if 1.0 in summary_df_original['label'].values and 0.0 in summary_df_original['label'].values:
    auc_original, ci_original = auc_roc_ci(summary_df_original['label'].values, summary_df_original['score'].values, 0.95)
    print(f"\n✓ Mô hình gốc - AUC: {auc_original:.4f} (95% CI: {ci_original[0]:.4f} - {ci_original[1]:.4f})")

# -------- EXPERIMENT 2: CHẠY MÔ HÌNH NLP CLINICAL --------
print("\n" + "=" * 80)
print("2. CHẠY MÔ HÌNH TÍCH HỢP NLP (Dữ liệu Lâm sàng Ngôn Ngữ Hóa)")
print("=" * 80)
summary_df_nlp, coef_df_nlp = train(
    modality_list_in=modality_list_nlp,
    modality_mask=modality_MASK_nlp,
    outcomes=df_outcomes,
    l1_dfs_filter=l1_dfs_filter,
    model_params=model_params_nlp,
    folds=folds
)

if 1.0 in summary_df_nlp['label'].values and 0.0 in summary_df_nlp['label'].values:
    auc_nlp, ci_nlp = auc_roc_ci(summary_df_nlp['label'].values, summary_df_nlp['score'].values, 0.95)
    print(f"\n✓ Mô hình NLP - AUC: {auc_nlp:.4f} (95% CI: {ci_nlp[0]:.4f} - {ci_nlp[1]:.4f})")

# ----------------- SO SÁNH KẾT QUẢ -------------------
print("\n" + "=" * 80)
print("KẾT QUẢ SO SÁNH")
print("=" * 80)

comparison_results = {
    'Model': ['Original Attention', 'NLP Clinical Attention'],
    'AUC': [auc_original, auc_nlp],
    'AUC_CI_Lower': [ci_original[0], ci_nlp[0]],
    'AUC_CI_Upper': [ci_original[1], ci_nlp[1]],
    'AUC_Difference': [0, auc_nlp - auc_original]
}

df_comparison = pd.DataFrame(comparison_results)
print("\n" + df_comparison.to_string(index=False))

# Tính metrics khác
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def find_best_threshold(y_true, y_score):
    from sklearn.metrics import roc_curve
    y_true = np.array(y_true).astype(float)
    y_score = np.array(y_score).astype(float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        return np.median(y_score)
    
    y_true_binary = (y_true == unique_labels[1]).astype(int) if not set(unique_labels).issubset({0,1,0.0,1.0}) else y_true.astype(int)
    
    if len(np.unique(y_true_binary)) < 2: return np.median(y_score)
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        return thresholds[np.argmax(tpr - fpr)]
    except:
        return np.median(y_score)

# So sánh nhãn/score hai mô hình
y_true_original = np.array(pd.to_numeric(summary_df_original['label'].values, errors='coerce'), dtype=np.float64)
y_score_original = np.array(pd.to_numeric(summary_df_original['score'].values, errors='coerce'), dtype=np.float64)
y_true_nlp = np.array(pd.to_numeric(summary_df_nlp['label'].values, errors='coerce'), dtype=np.float64)
y_score_nlp = np.array(pd.to_numeric(summary_df_nlp['score'].values, errors='coerce'), dtype=np.float64)

mask_original = ~(np.isnan(y_true_original) | np.isnan(y_score_original))
mask_nlp = ~(np.isnan(y_true_nlp) | np.isnan(y_score_nlp))
common_mask = mask_original & mask_nlp

y_true_clean = y_true_original[common_mask]
y_score_original_clean = y_score_original[common_mask]
y_score_nlp_clean = y_score_nlp[common_mask]

if not set(np.unique(y_true_clean)).issubset({0, 1, 0.0, 1.0}):
    y_true_clean = (y_true_clean == np.unique(y_true_clean)[1]).astype(int)

threshold_original = find_best_threshold(y_true_clean, y_score_original_clean)
threshold_nlp = find_best_threshold(y_true_clean, y_score_nlp_clean)

y_pred_original = (y_score_original_clean > threshold_original).astype(int)
y_pred_nlp = (y_score_nlp_clean > threshold_nlp).astype(int)
y_true_final = y_true_clean.astype(int)

metrics_comparison = {
    'Model': ['Original Attention', 'NLP Clinical Attention'],
    'Accuracy': [accuracy_score(y_true_final, y_pred_original), accuracy_score(y_true_final, y_pred_nlp)],
    'Precision': [precision_score(y_true_final, y_pred_original, zero_division=0), precision_score(y_true_final, y_pred_nlp, zero_division=0)],
    'Recall': [recall_score(y_true_final, y_pred_original, zero_division=0), recall_score(y_true_final, y_pred_nlp, zero_division=0)],
    'F1-Score': [f1_score(y_true_final, y_pred_original, zero_division=0), f1_score(y_true_final, y_pred_nlp, zero_division=0)]
}

df_metrics = pd.DataFrame(metrics_comparison)
print("\n" + "=" * 80)
print("CÁC METRICS KHÁC")
print("=" * 80)
print("\n" + df_metrics.to_string(index=False))

print("\n[V] Đang lưu kết quả...")
output_dir = Path('./excel')
output_dir.mkdir(exist_ok=True)

with pd.ExcelWriter(output_dir / 'nlp_comparison_results.xlsx', engine='xlsxwriter') as writer:
    df_comparison.to_excel(writer, sheet_name='AUC_Comparison', index=False)
    df_metrics.to_excel(writer, sheet_name='Metrics_Comparison', index=False)
    summary_df_original.to_excel(writer, sheet_name='Original_Predictions', index=True)
    summary_df_nlp.to_excel(writer, sheet_name='NLP_Predictions', index=True)
    coef_df_original.to_excel(writer, sheet_name='Original_Coefficients', index=True)
    coef_df_nlp.to_excel(writer, sheet_name='NLP_Coefficients', index=True)

print(f"✓ Đã lưu kết quả vào: {output_dir / 'nlp_comparison_results.xlsx'}")

print("\n" + "=" * 80)
print("KẾT LUẬN")
print("=" * 80)
if auc_nlp > auc_original:
    improvement = ((auc_nlp - auc_original) / auc_original) * 100
    print(f"✓ Mô hình NLP Attention cho kết quả tốt hơn {improvement:.2f}%")
    print(f"  (AUC tăng từ {auc_original:.4f} lên {auc_nlp:.4f})")
elif auc_original > auc_nlp:
    decrease = ((auc_original - auc_nlp) / auc_original) * 100
    print(f"⚠ Mô hình gốc cho kết quả tốt hơn {decrease:.2f}%")
    print(f"  (AUC giảm từ {auc_original:.4f} xuống {auc_nlp:.4f})")
else:
    print("≈ Hai mô hình cho kết quả tương đương")

print("\n" + "=" * 80)
print("HOÀN TẤT!")
print("=" * 80)
