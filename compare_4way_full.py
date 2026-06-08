"""
Script So sánh toàn diện Đa mô hình:
1. Mô hình gốc (Original Attention + Clinical Labs Numeric)
2. Mô hình OvO (OvO Attention + Clinical Labs Numeric)
3. Mô hình NLP (Original Attention + NLP Clinical Text Embedding)
4. Mô hình OvO + NLP (OvO Attention + NLP Clinical Text Embedding)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import importlib

import lung_helpers
importlib.reload(lung_helpers)
from lung_helpers import (
    train, train_ovo, auc_roc_ci, 
    get_clinical_table_v2, prepare_rad_modality_by_size, prepare_rad_modality,
    prepare_other_modalities, decorate_with_site_index,
    get_training_data
)

from clinical_nlp_embedding import prepare_nlp_clinical_modality

print("=" * 100)
print("SO SÁNH 4 TRẠNG THÁI: GỐC | OVO | NLP | NLP + OVO")
print("=" * 100)

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
    # Baseline
    {'name': 'Clinical_Only', 'modalities': ['cnl_dem_labs'], 'filters': {}, 'params': model_params_gate_off},
    
    # 2-3 Modalities (Tiêu chuẩn)
    {'name': 'Rad+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Gen_Paper+Labs', 'modalities': ['gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_dem_labs'], 'filters': {}, 'params': model_params},
    
    # Paper's "Best Configurations" but with Clinical Labs appended
    {'name': 'Rad+IHC-A+Gen_Paper+Labs', 
     'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_dem_labs'], 
     'filters': dfs_rad_filters, 'params': model_params},
     
    {'name': 'Rad+IHC-G+Gen_Paper+Labs', 
     'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_dem_labs'], 
     'filters': dfs_rad_filters, 'params': model_params},
     
    # Báo cáo kỷ lục > 0.811 AUC (Thêm PDL1 Score)
    {'name': 'Rad+IHC-A+Gen_Paper+PDL1+Labs (Best Model 1)', 
     'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_pdl1_score', 'cnl_dem_labs'], 
     'filters': dfs_rad_filters, 'params': model_params},
     
    # Báo cáo kỷ lục > 0.821 AUC (Thêm PDL1 Score)
    {'name': 'Rad+IHC-G+Gen_Paper+PDL1+Labs (Best Model 2)', 
     'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_pdl1_score', 'cnl_dem_labs'], 
     'filters': dfs_rad_filters, 'params': model_params},
]

valid_test_cases = []
for case in test_cases:
    missing = [m for m in case['modalities'] if m not in modality_dict_original]
    if not missing:
        valid_test_cases.append(case)
    else:
        print(f"   ⚠ Bỏ qua test case '{case['name']}': thiếu modalities {missing}")

print(f"\n   - Số test cases hợp lệ chứa Clinical Labs: {len(valid_test_cases)}/{len(test_cases)}")

print("\n[6/7] Chạy so sánh TOÀN DIỆN cho từng test case...")
print("=" * 100)

results_summary = []

for idx, case in enumerate(valid_test_cases):
    print(f"\n[{idx+1}/{len(valid_test_cases)}] Test case: {case['name']}")
    print(f"   Modalities: {', '.join(case['modalities'])}")
    
    try:
        # ---- PREPARE DATA ----
        data_orig, mask_orig, labels_orig = get_training_data(case['modalities'], modality_dict_original, modality_MASK, df_outcomes)
        
        nlp_mods_list = [m if m != 'cnl_dem_labs' else 'cnl_nlp_embedding' for m in case['modalities']]
        data_nlp, mask_nlp, labels_nlp = get_training_data(nlp_mods_list, modality_dict_nlp, modality_MASK_nlp, df_outcomes)
        
        if len(data_orig) == 0 or len(labels_orig) == 0:
            print(f"   ⚠ Bỏ qua: không có dữ liệu")
            continue
        print(f"   - Samples: {len(labels_orig)}")
        
        
        # 1. CHUYẾN XA: MÔ HÌNH GỐC
        print(f"   (1/4) Chạy Mô hình Gốc...")
        summary_orig, _ = train(
            modality_list_in=data_orig, modality_mask=mask_orig, outcomes=labels_orig,
            l1_dfs_filter=case['filters'], model_params=case['params'], folds=folds
        )
        auc_orig = auc_roc_ci(summary_orig['label'].values, summary_orig['score'].values, 0.95)[0] if (1.0 in summary_orig['label'].values and 0.0 in summary_orig['label'].values) else 0

        # 2. CHUYẾN XA: MÔ HÌNH OVO
        print(f"   (2/4) Chạy Mô hình OvO Gốc...")
        summary_ovo, _ = train_ovo(
            modality_list_in=data_orig, modality_mask=mask_orig, outcomes=labels_orig,
            l1_dfs_filter=case['filters'], model_params=case['params'], folds=folds
        )
        auc_ovo = auc_roc_ci(summary_ovo['label'].values, summary_ovo['score'].values, 0.95)[0] if (1.0 in summary_ovo['label'].values and 0.0 in summary_ovo['label'].values) else 0

        # PARAMETER SETUP CHO NLP
        nlp_params = case['params'].copy()
        if 'cnl_nlp_embedding' in mask_nlp.columns:
            nlp_idx = list(mask_nlp.columns).index('cnl_nlp_embedding')
            nlp_params['no_scale'] = [nlp_idx]
            
        # 3. CHUYẾN XA: MÔ HÌNH NLP
        print(f"   (3/4) Chạy Mô hình NLP Attention...")
        summary_nlp, _ = train(
            modality_list_in=data_nlp, modality_mask=mask_nlp, outcomes=labels_nlp,
            l1_dfs_filter=case['filters'], model_params=nlp_params, folds=folds
        )
        auc_nlp = auc_roc_ci(summary_nlp['label'].values, summary_nlp['score'].values, 0.95)[0] if (1.0 in summary_nlp['label'].values and 0.0 in summary_nlp['label'].values) else 0

        # 4. CHUYẾN XA: MÔ HÌNH OVO + NLP
        print(f"   (4/4) Chạy Mô hình OvO + NLP...")
        summary_nlp_ovo, _ = train_ovo(
            modality_list_in=data_nlp, modality_mask=mask_nlp, outcomes=labels_nlp,
            l1_dfs_filter=case['filters'], model_params=nlp_params, folds=folds
        )
        auc_nlp_ovo = auc_roc_ci(summary_nlp_ovo['label'].values, summary_nlp_ovo['score'].values, 0.95)[0] if (1.0 in summary_nlp_ovo['label'].values and 0.0 in summary_nlp_ovo['label'].values) else 0

        # LƯU KẾT QUẢ
        winner_auc = max(auc_orig, auc_ovo, auc_nlp, auc_nlp_ovo)
        if winner_auc == auc_nlp_ovo: winner_name = "NLP + OvO"
        elif winner_auc == auc_nlp: winner_name = "NLP Attention"
        elif winner_auc == auc_ovo: winner_name = "OvO Original"
        elif winner_auc == auc_orig: winner_name = "Original"
        else: winner_name = "Unknown"

        results_summary.append({
            'Test Case': case['name'],
            'Modalities': ', '.join(case['modalities']),
            'N_Samples': len(labels_orig),
            'AUC_Original': auc_orig,
            'AUC_OvO': auc_ovo,
            'AUC_NLP': auc_nlp,
            'AUC_NLP_OvO': auc_nlp_ovo,
            'Max_AUC': winner_auc,
            'Winner': winner_name,
            'Improvement_(vs_Orig)': winner_auc - auc_orig
        })
        
        print(f"   » Kết quả (AUC): Orig={auc_orig:.4f} | OvO={auc_ovo:.4f} | NLP={auc_nlp:.4f} | NLP+OvO={auc_nlp_ovo:.4f}")
        print(f"   » TỐT NHẤT: {winner_name} (Tăng {winner_auc - auc_orig:.4f} so với mẫu cơ sở)")
        
    except Exception as e:
        print(f"   ✗ Lỗi nội bộ cho case này: {str(e)}")
        continue

print("\n[7/7] Tổng hợp kết quả TOÀN DIỆN...")
print("=" * 100)

if not results_summary:
    print("⚠ Không có kết quả nào để tổng hợp!")
    sys.exit(1)

df_results = pd.DataFrame(results_summary)

print("\nBẢNG XẾP HẠNG 4 MÔ HÌNH\n")
# Định dạng chuỗi hiển thị 4 chữ số thập phân
df_display = df_results[['Test Case', 'AUC_Original', 'AUC_OvO', 'AUC_NLP', 'AUC_NLP_OvO', 'Winner']].copy()
for c in ['AUC_Original', 'AUC_OvO', 'AUC_NLP', 'AUC_NLP_OvO']:
    df_display[c] = df_display[c].apply(lambda x: f"{x:.4f}" if x>0 else "-")
print(df_display.to_string(index=False))

winner_counts = df_results['Winner'].value_counts()
print("\nTHỐNG KÊ CHIẾN THẮNG THEO STATE")
for name, c in winner_counts.items():
    print(f"  - {name}: {c} lần ({100*c/len(results_summary):.1f}%)")

print("\nĐang lưu kết quả Excel tổng thể của dự án...")
output_dir = Path('./excel')
output_dir.mkdir(exist_ok=True)

with pd.ExcelWriter(output_dir / 'compare_4way_full_results.xlsx', engine='xlsxwriter') as writer:
    df_results.to_excel(writer, sheet_name='Summary_Bench', index=False)
    # Thống kê top win
    pd.DataFrame({'Model': winner_counts.index, 'Wins': winner_counts.values}).to_excel(writer, sheet_name='Win_Count', index=False)

print(f"✓ Bạn đã có thể xem bảng Benchmarking khổng lồ tại: {output_dir / 'compare_4way_full_results.xlsx'}")
