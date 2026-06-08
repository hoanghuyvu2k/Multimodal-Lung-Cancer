"""
Script so sánh mô hình Attention gốc và OvO (One-Versus-Others) Attention

Script này sẽ:
1. Load dữ liệu từ datasets
2. Chạy cả hai mô hình (gốc và OvO)
3. So sánh kết quả AUC, accuracy, và các metrics khác
4. Lưu kết quả vào file Excel
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import importlib

# Import lung_helpers
import lung_helpers
importlib.reload(lung_helpers)
from lung_helpers import (
    train, train_ovo, auc_roc_ci, 
    get_clinical_table_v2, prepare_rad_modality_by_size, 
    prepare_other_modalities, decorate_with_site_index
)

print("=" * 80)
print("SO SÁNH MÔ HÌNH ATTENTION GỐC VÀ OVO ATTENTION")
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

df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
df_tmb = df_genomic[['TMB']]
df_nontmb = df_genomic.loc[:, ~df_genomic.columns.str.contains("TMB")]
try:
    df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/pdl1_score.parquet")
except:
    try:
        df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/PDL1_SCORE.parquet")
    except:
        print("   ⚠ Không tìm thấy file pdl1_score")
        df_pdl1 = pd.DataFrame()
df_labs = df_clinical[clinical_predictors]

# Try different possible file names
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

# Chuẩn bị modality mask và dictionary
print("\n[4/6] Đang chuẩn bị modality mask...")
modality_MASK = df_clinical[[]]
modality_dict = {}

# Radiology modalities
modality_PC_full = prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'PC', 'rad_lesion_pc')
modality_PL_full = prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'PL', 'rad_lesion_pl')
modality_LN_full = prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'LN', 'rad_lesion_ln')

modality_LU_full = prepare_rad_modality_by_size(
    modality_dict, 
    df_radiology_by_site, 
    modality_MASK, 
    sites=['PC', 'PL', 'LN'], 
    name='rad_lesion_lu',
    sort='original_shape_MeshVolume', 
    ascending=False,
    reduce=True
)

# Other modalities
prepare_other_modalities(modality_dict, df_texture, modality_MASK, 'path_ihc_pdl1')
prepare_other_modalities(modality_dict, df_glcm, modality_MASK, 'path_ihc_glcm')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, 'gen_driver_mut_amp')
prepare_other_modalities(modality_dict, df_nontmb, modality_MASK, 'gen_driver_non_tmb')
prepare_other_modalities(modality_dict, df_tmb, modality_MASK, 'gen_driver_tmb')
prepare_other_modalities(modality_dict, df_pdl1, modality_MASK, 'cnl_pdl1_score')
prepare_other_modalities(modality_dict, df_labs, modality_MASK, 'cnl_dem_labs')

modality_dict['cnl_pdl1_score'] = - modality_dict['cnl_pdl1_score'] / 100.0
modality_MASK = modality_MASK.fillna(False)

print(f"   - Số lượng modalities: {len(modality_dict)}")
print(f"   - Modality mask shape: {modality_MASK.shape}")
print(f"   - Modality availability:")
for col in modality_MASK.columns:
    count = modality_MASK[col].sum()
    print(f"     * {col}: {count}/{len(modality_MASK)} ({100*count/len(modality_MASK):.1f}%)")

# Chuẩn bị modality list
modality_list = [modality_dict[key] for key in modality_MASK.columns]

# Thiết lập tham số mô hình
print("\n[5/6] Thiết lập tham số mô hình...")
model_params = {
    'epochs': 100,
    'alpha': 1.0,
    'beta': 1.0,
    'lr': 0.01,
    'attention_gate_enabled': True
}

l1_dfs_filter = {}  # Không dùng L1 filter cho so sánh đơn giản
folds = 5  # Giảm số folds để chạy nhanh hơn

# Chạy mô hình gốc
print("\n" + "=" * 80)
print("CHẠY MÔ HÌNH ATTENTION GỐC")
print("=" * 80)
summary_df_original, coef_df_original = train(
    modality_list_in=modality_list,
    modality_mask=modality_MASK,
    outcomes=df_outcomes,
    l1_dfs_filter=l1_dfs_filter,
    model_params=model_params,
    folds=folds
)

# Tính AUC cho mô hình gốc
if 1.0 in summary_df_original['label'].values and 0.0 in summary_df_original['label'].values:
    auc_original, ci_original = auc_roc_ci(
        summary_df_original['label'].values, 
        summary_df_original['score'].values, 
        0.95
    )
    print(f"\n✓ Mô hình gốc - AUC: {auc_original:.4f} (95% CI: {ci_original[0]:.4f} - {ci_original[1]:.4f})")

# Chạy mô hình OvO
print("\n" + "=" * 80)
print("CHẠY MÔ HÌNH OVO ATTENTION")
print("=" * 80)
summary_df_ovo, coef_df_ovo = train_ovo(
    modality_list_in=modality_list,
    modality_mask=modality_MASK,
    outcomes=df_outcomes,
    l1_dfs_filter=l1_dfs_filter,
    model_params=model_params,
    folds=folds
)

# Tính AUC cho mô hình OvO
if 1.0 in summary_df_ovo['label'].values and 0.0 in summary_df_ovo['label'].values:
    auc_ovo, ci_ovo = auc_roc_ci(
        summary_df_ovo['label'].values, 
        summary_df_ovo['score'].values, 
        0.95
    )
    print(f"\n✓ Mô hình OvO - AUC: {auc_ovo:.4f} (95% CI: {ci_ovo[0]:.4f} - {ci_ovo[1]:.4f})")

# So sánh kết quả
print("\n" + "=" * 80)
print("KẾT QUẢ SO SÁNH")
print("=" * 80)

comparison_results = {
    'Model': ['Original Attention', 'OvO Attention'],
    'AUC': [auc_original, auc_ovo],
    'AUC_CI_Lower': [ci_original[0], ci_ovo[0]],
    'AUC_CI_Upper': [ci_original[1], ci_ovo[1]],
    'AUC_Difference': [0, auc_ovo - auc_original]
}

df_comparison = pd.DataFrame(comparison_results)
print("\n" + df_comparison.to_string(index=False))

# Tính các metrics khác
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Tìm threshold tối ưu cho mỗi mô hình
def find_best_threshold(y_true, y_score):
    from sklearn.metrics import roc_curve
    
    # Đảm bảo y_true là binary (0/1) và loại bỏ NaN
    y_true = np.array(y_true).astype(float)
    y_score = np.array(y_score).astype(float)
    
    # Loại bỏ NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    # Đảm bảo y_true là binary (0/1)
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        # Nếu không phải binary, thử convert
        if len(unique_labels) > 2:
            raise ValueError(f"y_true phải là binary nhưng có {len(unique_labels)} giá trị khác nhau: {unique_labels}")
        # Nếu chỉ có 1 lớp, không thể tính ROC
        return np.median(y_score)
    
    # Đảm bảo labels là 0 và 1
    if not (set(unique_labels) == {0, 1} or set(unique_labels) == {0.0, 1.0}):
        # Nếu labels không phải 0/1, convert về 0/1
        y_true_binary = (y_true == unique_labels[1]).astype(int)
    else:
        y_true_binary = y_true.astype(int)
    
    # Kiểm tra xem có đủ cả hai lớp không
    if len(np.unique(y_true_binary)) < 2:
        return np.median(y_score)
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    except Exception as e:
        print(f"Warning: Không thể tính ROC curve, sử dụng median: {e}")
        return np.median(y_score)

# Lấy labels và scores và convert sang numpy array numeric
y_true_original = pd.to_numeric(summary_df_original['label'].values, errors='coerce')
y_score_original = pd.to_numeric(summary_df_original['score'].values, errors='coerce')
y_true_ovo = pd.to_numeric(summary_df_ovo['label'].values, errors='coerce')
y_score_ovo = pd.to_numeric(summary_df_ovo['score'].values, errors='coerce')

# Convert sang numpy array float
y_true_original = np.array(y_true_original, dtype=np.float64)
y_score_original = np.array(y_score_original, dtype=np.float64)
y_true_ovo = np.array(y_true_ovo, dtype=np.float64)
y_score_ovo = np.array(y_score_ovo, dtype=np.float64)

# Đảm bảo cả hai có cùng labels
# Sử dụng y_true từ original làm reference
y_true = y_true_original.copy()

# Loại bỏ NaN
mask_original = ~(np.isnan(y_true_original) | np.isnan(y_score_original))
mask_ovo = ~(np.isnan(y_true_ovo) | np.isnan(y_score_ovo))

# Chỉ sử dụng các samples có trong cả hai
common_mask = mask_original & mask_ovo
y_true_clean = y_true_original[common_mask]
y_score_original_clean = y_score_original[common_mask]
y_score_ovo_clean = y_score_ovo[common_mask]

# Convert y_true về binary nếu cần
if not set(np.unique(y_true_clean)).issubset({0, 1, 0.0, 1.0}):
    unique_vals = np.unique(y_true_clean)
    if len(unique_vals) == 2:
        y_true_clean = (y_true_clean == unique_vals[1]).astype(int)
    else:
        print(f"Warning: y_true có {len(unique_vals)} giá trị khác nhau: {unique_vals}")
        y_true_clean = y_true_clean.astype(int)

threshold_original = find_best_threshold(y_true_clean, y_score_original_clean)
threshold_ovo = find_best_threshold(y_true_clean, y_score_ovo_clean)

y_pred_original = (y_score_original_clean > threshold_original).astype(int)
y_pred_ovo = (y_score_ovo_clean > threshold_ovo).astype(int)
y_true_final = y_true_clean.astype(int)

metrics_comparison = {
    'Model': ['Original Attention', 'OvO Attention'],
    'Accuracy': [
        accuracy_score(y_true_final, y_pred_original),
        accuracy_score(y_true_final, y_pred_ovo)
    ],
    'Precision': [
        precision_score(y_true_final, y_pred_original, zero_division=0),
        precision_score(y_true_final, y_pred_ovo, zero_division=0)
    ],
    'Recall': [
        recall_score(y_true_final, y_pred_original, zero_division=0),
        recall_score(y_true_final, y_pred_ovo, zero_division=0)
    ],
    'F1-Score': [
        f1_score(y_true_final, y_pred_original, zero_division=0),
        f1_score(y_true_final, y_pred_ovo, zero_division=0)
    ]
}

df_metrics = pd.DataFrame(metrics_comparison)
print("\n" + "=" * 80)
print("CÁC METRICS KHÁC")
print("=" * 80)
print("\n" + df_metrics.to_string(index=False))

# Lưu kết quả
print("\n[6/6] Đang lưu kết quả...")
output_dir = Path('./excel')
output_dir.mkdir(exist_ok=True)

with pd.ExcelWriter(output_dir / 'ovo_comparison_results.xlsx', engine='xlsxwriter') as writer:
    df_comparison.to_excel(writer, sheet_name='AUC_Comparison', index=False)
    df_metrics.to_excel(writer, sheet_name='Metrics_Comparison', index=False)
    summary_df_original.to_excel(writer, sheet_name='Original_Predictions', index=True)
    summary_df_ovo.to_excel(writer, sheet_name='OvO_Predictions', index=True)
    coef_df_original.to_excel(writer, sheet_name='Original_Coefficients', index=True)
    coef_df_ovo.to_excel(writer, sheet_name='OvO_Coefficients', index=True)

print(f"✓ Đã lưu kết quả vào: {output_dir / 'ovo_comparison_results.xlsx'}")

# Kết luận
print("\n" + "=" * 80)
print("KẾT LUẬN")
print("=" * 80)
if auc_ovo > auc_original:
    improvement = ((auc_ovo - auc_original) / auc_original) * 100
    print(f"✓ Mô hình OvO Attention cho kết quả tốt hơn {improvement:.2f}%")
    print(f"  (AUC tăng từ {auc_original:.4f} lên {auc_ovo:.4f})")
elif auc_original > auc_ovo:
    decrease = ((auc_original - auc_ovo) / auc_original) * 100
    print(f"⚠ Mô hình gốc cho kết quả tốt hơn {decrease:.2f}%")
    print(f"  (AUC giảm từ {auc_original:.4f} xuống {auc_ovo:.4f})")
else:
    print("≈ Hai mô hình cho kết quả tương đương")

print("\n" + "=" * 80)
print("HOÀN TẤT!")
print("=" * 80)

