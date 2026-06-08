"""
Script so sánh đầy đủ mô hình One-Versus-Others (OvO) Attention gốc 
và NLP + OvO Attention trên các tổ hợp modality có chứa dữ liệu lâm sàng (cnl_dem_labs).
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import importlib

import lung_helpers
importlib.reload(lung_helpers)
from lung_helpers import (
    train_ovo, auc_roc_ci, 
    get_clinical_table_v2, prepare_rad_modality_by_size, prepare_rad_modality,
    prepare_other_modalities, decorate_with_site_index,
    get_training_data
)

from clinical_nlp_embedding import prepare_nlp_clinical_modality

print("=" * 80)
print("SO SÁNH ĐẦY ĐỦ MÔ HÌNH OVO ATTENTION GỐC VÀ NLP + OVO ATTENTION")
print("=" * 80)

BASE_DB_DIR = '../datasets'

print("\n[1/7] Đang load dữ liệu cohort...")
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort']=='discovery']
print(f"   - Discovery cohort: {len(df_cohort_disc)} samples")

print("\n[2/7] Đang load clinical data...")
df_clinical = get_clinical_table_v2(
    path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv", 
    main_index_col='dmp_pt_id',
    cohort=df_cohort_disc
)
df_outcomes = df_clinical[['label']].copy(deep=True)
print(f"   - Outcomes: {len(df_outcomes)} samples")

print("\n[3/7] Đang load các modality...")
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

try:
    df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")
except:
    try:
        df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/LUNG_RADIOMICS_spacing1.0_MirpOn_Window1350.250_allImageTypes_bw20.parquet")
    except:
        print("   ⚠ Không tìm thấy file radiomics, sử dụng file trong code/")
        df_radiology = pd.read_parquet("lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")

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


# CHUẨN BỊ MẢNG GỐC VÀ NLP
print("\n[4/7] Đang chuẩn bị modality masks và NLP embeddings...")
modality_MASK = df_clinical[[]].copy()
modality_dict_original = {}
modality_dict_nlp = {}

# Radiology modalities
modality_PC_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK, 'PC', 'rad_lesion_pc')
modality_PL_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK, 'PL', 'rad_lesion_pl')
modality_LN_full = prepare_rad_modality_by_size(modality_dict_original, df_radiology_by_site, modality_MASK, 'LN', 'rad_lesion_ln')

modality_LU_full = prepare_rad_modality_by_size(
    modality_dict_original, 
    df_radiology_by_site, 
    modality_MASK, 
    sites=['PC', 'PL', 'LN'], 
    name='rad_lesion_lu',
    sort='original_shape_MeshVolume', 
    ascending=False,
    reduce=True
)

# Other modalities
if not df_texture.empty: prepare_other_modalities(modality_dict_original, df_texture, modality_MASK, 'path_ihc_pdl1')
if not df_glcm.empty: prepare_other_modalities(modality_dict_original, df_glcm, modality_MASK, 'path_ihc_glcm')
if not df_genomic.empty: prepare_other_modalities(modality_dict_original, df_genomic, modality_MASK, 'gen_driver_mut_amp')
if not df_nontmb.empty: prepare_other_modalities(modality_dict_original, df_nontmb, modality_MASK, 'gen_driver_non_tmb')
if not df_tmb.empty: prepare_other_modalities(modality_dict_original, df_tmb, modality_MASK, 'gen_driver_tmb')
if not df_pdl1.empty: prepare_other_modalities(modality_dict_original, df_pdl1, modality_MASK, 'cnl_pdl1_score')

prepare_other_modalities(modality_dict_original, df_labs, modality_MASK, 'cnl_dem_labs')

if 'cnl_pdl1_score' in modality_dict_original:
    modality_dict_original['cnl_pdl1_score'] = - modality_dict_original['cnl_pdl1_score'] / 100.0
modality_MASK = modality_MASK.fillna(False)

# Clone for NLP
for k, v in modality_dict_original.items():
    if k != 'cnl_dem_labs':
        modality_dict_nlp[k] = v

modality_MASK_nlp = modality_MASK.copy()
prepare_nlp_clinical_modality(modality_dict_nlp, df_labs, modality_MASK_nlp, 'cnl_nlp_embedding')
modality_MASK_nlp = modality_MASK_nlp.fillna(False)

print(f"   - Modalities gốc: {len(modality_dict_original)}")
print(f"   - Modalities NLP: {len(modality_dict_nlp)}")

print("\n[5/7] Thiết lập tham số mô hình...")
model_params = {'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0, 'cross_modality_enabled': False}
model_params_gate_off = {'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0, 'cross_modality_enabled': False, 'attention_gate_enabled': False}

dfs_rad_filters = {
    0: {'l1_selection_df': modality_PC_full, 'kwargs': {'l1_strength': 0.1}},
    1: {'l1_selection_df': modality_PL_full, 'kwargs': {'l1_strength': 0.1}},
    2: {'l1_selection_df': modality_LN_full, 'kwargs': {'l1_strength': 0.1}},
}
dfs_rad_filters_lu = {
    0: {'l1_selection_df': modality_PC_full, 'kwargs': {'l1_strength': 0.1}},
    1: {'l1_selection_df': modality_PL_full, 'kwargs': {'l1_strength': 0.1}},
    2: {'l1_selection_df': modality_LN_full, 'kwargs': {'l1_strength': 0.1}},
    3: {'l1_selection_df': modality_LU_full, 'kwargs': {'l1_strength': 0.1}},
}
folds = 10 

test_cases = [
    # Single modality
    {'name': 'Clinical_Only', 'modalities': ['cnl_dem_labs'], 'filters': {}, 'params': model_params_gate_off},
    
    # Two modalities
    {'name': 'Rad+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad-LU+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'rad_lesion_lu', 'cnl_dem_labs'], 'filters': dfs_rad_filters_lu, 'params': model_params},
    {'name': 'Gen+Labs', 'modalities': ['gen_driver_mut_amp', 'cnl_dem_labs'], 'filters': {}, 'params': model_params},
    {'name': 'IHC-A+Labs', 'modalities': ['path_ihc_pdl1', 'cnl_dem_labs'], 'filters': {}, 'params': model_params},
    {'name': 'IHC-G+Labs', 'modalities': ['path_ihc_glcm', 'cnl_dem_labs'], 'filters': {}, 'params': model_params},
    
    # Three modalities
    {'name': 'Rad+Gen+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'gen_driver_mut_amp', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-A+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    
    # Four+ modalities
    {'name': 'Rad+IHC-A+Gen+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_mut_amp', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-G+Gen+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_mut_amp', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'All_Best_Modality+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
]

valid_test_cases = []
for case in test_cases:
    missing = [m for m in case['modalities'] if m not in modality_dict_original]
    if not missing:
        valid_test_cases.append(case)
    else:
        print(f"   ⚠ Bỏ qua test case '{case['name']}': thiếu modalities {missing}")

print(f"\n   - Số test cases hợp lệ chứa Clinical Labs: {len(valid_test_cases)}/{len(test_cases)}")

print("\n[6/7] Chạy so sánh cho từng test case (GỌI HÀM TRAIN_OVO)...")
print("=" * 80)

results_summary = []

for idx, case in enumerate(valid_test_cases):
    print(f"\n[{idx+1}/{len(valid_test_cases)}] Test case: {case['name']}")
    print(f"   Modalities: {', '.join(case['modalities'])}")
    
    try:
        # DATA CHO MÔ HÌNH OVO LÂM SÀNG SỐ
        data_orig, mask_orig, labels_orig = get_training_data(case['modalities'], modality_dict_original, modality_MASK, df_outcomes)
        
        # DATA CHO MÔ HÌNH NLP OVO
        nlp_mods_list = [m if m != 'cnl_dem_labs' else 'cnl_nlp_embedding' for m in case['modalities']]
        data_nlp, mask_nlp, labels_nlp = get_training_data(nlp_mods_list, modality_dict_nlp, modality_MASK_nlp, df_outcomes)
        
        if len(data_orig) == 0 or len(labels_orig) == 0:
            print(f"   ⚠ Bỏ qua: không có dữ liệu")
            continue
        print(f"   - Samples: {len(labels_orig)}")
        
        # CHẠY MÔ HÌNH OvO GỐC (Dữ liệu Số)
        print(f"   - Đang chạy mô hình OvO gốc (Tham số số học)...")
        summary_orig, _ = train_ovo(
            modality_list_in=data_orig,
            modality_mask=mask_orig,
            outcomes=labels_orig,
            l1_dfs_filter=case['filters'],
            model_params=case['params'],
            folds=folds
        )
        if 1.0 in summary_orig['label'].values and 0.0 in summary_orig['label'].values:
            auc_orig, ci_orig = auc_roc_ci(summary_orig['label'].values, summary_orig['score'].values, 0.95)
        else: continue
        
        # CHẠY MÔ HÌNH NLP + OvO
        print(f"   - Đang chạy mô hình NLP + OvO (Text Embedding)...")
        nlp_params = case['params'].copy()
        if 'cnl_nlp_embedding' in mask_nlp.columns:
            # Lấy vị trí của cnl_nlp_embedding theo mask_nlp
            nlp_idx = list(mask_nlp.columns).index('cnl_nlp_embedding')
            nlp_params['no_scale'] = [nlp_idx]
            
        summary_nlp, _ = train_ovo(
            modality_list_in=data_nlp,
            modality_mask=mask_nlp,
            outcomes=labels_nlp,
            l1_dfs_filter=case['filters'], 
            model_params=nlp_params,
            folds=folds
        )
        if 1.0 in summary_nlp['label'].values and 0.0 in summary_nlp['label'].values:
            auc_nlp, ci_nlp = auc_roc_ci(summary_nlp['label'].values, summary_nlp['score'].values, 0.95)
        else: continue
        
        # LƯU KẾT QUẢ
        improvement = auc_nlp - auc_orig
        improvement_pct = (improvement / auc_orig * 100) if auc_orig > 0 else 0
        
        results_summary.append({
            'Test Case': case['name'],
            'Modalities': ', '.join(case['modalities']),
            'N_Samples': len(labels_orig),
            'OvO_Original_AUC': auc_orig,
            'OvO_Original_CI_Lower': ci_orig[0],
            'OvO_Original_CI_Upper': ci_orig[1],
            'NLP_OvO_AUC': auc_nlp,
            'NLP_OvO_CI_Lower': ci_nlp[0],
            'NLP_OvO_CI_Upper': ci_nlp[1],
            'AUC_Difference': improvement,
            'AUC_Improvement_%': improvement_pct,
            'Better_Model': 'NLP+OvO' if improvement > 0 else 'OvO_Original' if improvement < 0 else 'Equal'
        })
        
        print(f"   ✓ OvO Original AUC: {auc_orig:.4f}")
        print(f"   ✓ NLP + OvO AUC:    {auc_nlp:.4f}")
        print(f"   ✓ Improvement:      {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
    except Exception as e:
        print(f"   ✗ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print("\n[7/7] Tổng hợp kết quả...")
print("=" * 80)

if len(results_summary) == 0:
    print("⚠ Không có kết quả nào để tổng hợp!")
    sys.exit(1)

df_results = pd.DataFrame(results_summary)

comparison_table = []
for r in results_summary:
    comparison_table.append({
        'Test Case': r['Test Case'],
        'N_Samples': r['N_Samples'],
        'OvO_Original_AUC': f"{r['OvO_Original_AUC']:.4f}",
        'NLP_OvO_AUC': f"{r['NLP_OvO_AUC']:.4f}",
        'Difference': f"{r['AUC_Difference']:+.4f}",
        'Improvement_%': f"{r['AUC_Improvement_%']:+.2f}%",
        'Winner': r['Better_Model']
    })

df_comparison = pd.DataFrame(comparison_table)
print("\nBẢNG SO SÁNH CHI TIẾT (OVO GỐC VS NLP+OVO)\n")
print(df_comparison.to_string(index=False))

nlp_wins = sum(1 for r in results_summary if r['Better_Model'] == 'NLP+OvO')
original_wins = sum(1 for r in results_summary if r['Better_Model'] == 'OvO_Original')
equal = sum(1 for r in results_summary if r['Better_Model'] == 'Equal')

print("\nTHỐNG KÊ TỔNG HỢP")
print(f"Tổng số test cases chạy: {len(results_summary)}")
print(f"  - NLP + OvO tốt hơn: {nlp_wins} ({100*nlp_wins/len(results_summary):.1f}%)")
print(f"  - OvO Original tốt hơn: {original_wins} ({100*original_wins/len(results_summary):.1f}%)")

print("\nĐang lưu kết quả Excel...")
output_dir = Path('./excel')
output_dir.mkdir(exist_ok=True)

with pd.ExcelWriter(output_dir / 'nlp_ovo_comparison_full_results.xlsx', engine='xlsxwriter') as writer:
    df_results.to_excel(writer, sheet_name='All_Results', index=False)
    df_comparison.to_excel(writer, sheet_name='Comparison_Table', index=False)

print(f"✓ Hoàn tất! Đã lưu file tại {output_dir / 'nlp_ovo_comparison_full_results.xlsx'}")
