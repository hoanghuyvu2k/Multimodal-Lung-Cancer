"""
Script so sánh đầy đủ mô hình Attention gốc và OvO (One-Versus-Others) Attention
trên tất cả các trường hợp như trong notebook figures-finalized.ipynb

Script này sẽ:
1. Load dữ liệu từ datasets
2. Chạy cả hai mô hình (gốc và OvO) trên tất cả các tổ hợp modality
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
    prepare_other_modalities, decorate_with_site_index,
    get_training_data
)

print("=" * 80)
print("SO SÁNH ĐẦY ĐỦ MÔ HÌNH ATTENTION GỐC VÀ OVO ATTENTION")
print("=" * 80)

# Thiết lập đường dẫn dữ liệu
BASE_DB_DIR = '../datasets'

# Load dữ liệu cohort
print("\n[1/7] Đang load dữ liệu cohort...")
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort']=='discovery']
print(f"   - Discovery cohort: {len(df_cohort_disc)} samples")

# Load clinical data
print("\n[2/7] Đang load clinical data...")
df_clinical = get_clinical_table_v2(
    path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv", 
    main_index_col='dmp_pt_id',
    cohort=df_cohort_disc
)
df_outcomes = df_clinical[['label']].copy(deep=True)
print(f"   - Outcomes: {len(df_outcomes)} samples")
print(f"   - Class distribution: {df_outcomes['label'].value_counts().to_dict()}")

# Load các modality
print("\n[3/7] Đang load các modality...")
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

# Chuẩn bị modality mask và dictionary
print("\n[4/7] Đang chuẩn bị modality mask...")
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
if not df_texture.empty:
    prepare_other_modalities(modality_dict, df_texture, modality_MASK, 'path_ihc_pdl1')
if not df_glcm.empty:
    prepare_other_modalities(modality_dict, df_glcm, modality_MASK, 'path_ihc_glcm')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, 'gen_driver_mut_amp')
prepare_other_modalities(modality_dict, df_nontmb, modality_MASK, 'gen_driver_non_tmb')
prepare_other_modalities(modality_dict, df_tmb, modality_MASK, 'gen_driver_tmb')
if not df_pdl1.empty:
    prepare_other_modalities(modality_dict, df_pdl1, modality_MASK, 'cnl_pdl1_score')
prepare_other_modalities(modality_dict, df_labs, modality_MASK, 'cnl_dem_labs')

if 'cnl_pdl1_score' in modality_dict:
    modality_dict['cnl_pdl1_score'] = - modality_dict['cnl_pdl1_score'] / 100.0
modality_MASK = modality_MASK.fillna(False)

print(f"   - Số lượng modalities: {len(modality_dict)}")
print(f"   - Modality mask shape: {modality_MASK.shape}")

# Thiết lập tham số mô hình
print("\n[5/7] Thiết lập tham số mô hình...")
model_params = {'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0, 'cross_modality_enabled': False}
model_params_gate_off = {'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0, 'cross_modality_enabled': False, 'attention_gate_enabled': False}

# Thiết lập filters cho radiomics
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

folds = 10  # Sử dụng 10 folds như trong notebook gốc

# Định nghĩa tất cả các test cases
test_cases = [
    # Single modality
    {'name': 'TMB', 'modalities': ['gen_driver_tmb'], 'filters': {}, 'params': model_params_gate_off},
    {'name': 'PDL1', 'modalities': ['cnl_pdl1_score'], 'filters': {}, 'params': model_params_gate_off},
    {'name': 'IHC-A', 'modalities': ['path_ihc_pdl1'], 'filters': {}, 'params': model_params_gate_off},
    {'name': 'Gen', 'modalities': ['gen_driver_mut_amp'], 'filters': {}, 'params': model_params_gate_off},
    
    # Two modalities
    {'name': 'TMB+PDL1', 'modalities': ['gen_driver_tmb', 'cnl_pdl1_score'], 'filters': {}, 'params': model_params_gate_off},
    {'name': 'Rad', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad-LU', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'rad_lesion_lu'], 'filters': dfs_rad_filters_lu, 'params': model_params},
    {'name': 'PDL1+Gen', 'modalities': ['cnl_pdl1_score', 'gen_driver_mut_amp'], 'filters': {}, 'params': model_params},
    
    # Three modalities
    {'name': 'Rad+IHC-A', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-G', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+Gen', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'gen_driver_mut_amp'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'IHC-A+Gen', 'modalities': ['path_ihc_pdl1', 'gen_driver_mut_amp'], 'filters': {}, 'params': model_params},
    {'name': 'IHC-G+Gen', 'modalities': ['path_ihc_glcm', 'gen_driver_mut_amp'], 'filters': {}, 'params': model_params},
    
    # Four modalities
    {'name': 'Rad+IHC-A+Gen', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_mut_amp'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-G+Gen', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_mut_amp'], 'filters': dfs_rad_filters, 'params': model_params},
    
    # Five+ modalities - Cấu hình tốt nhất theo bài báo: Rad + Gen + Pathology (PD-L1 TPS)
    {'name': 'Rad+IHC-A+Gen+PDL1', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_mut_amp', 'cnl_pdl1_score'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-G+Gen+PDL1', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_mut_amp', 'cnl_pdl1_score'], 'filters': dfs_rad_filters, 'params': model_params},
    # Thêm các biến thể khác có thể đạt kết quả cao
    {'name': 'Rad+IHC-A+Gen+TMB+PDL1', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_non_tmb', 'gen_driver_tmb', 'cnl_pdl1_score'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-A+Gen+PDL1+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_pdl1', 'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
    {'name': 'Rad+IHC-G+Gen+PDL1+Labs', 'modalities': ['rad_lesion_pc', 'rad_lesion_pl', 'rad_lesion_ln', 'path_ihc_glcm', 'gen_driver_mut_amp', 'cnl_pdl1_score', 'cnl_dem_labs'], 'filters': dfs_rad_filters, 'params': model_params},
]

# Lọc các test cases có modalities hợp lệ
valid_test_cases = []
for case in test_cases:
    # Kiểm tra xem tất cả modalities có trong modality_dict không
    missing = [m for m in case['modalities'] if m not in modality_dict]
    if not missing:
        valid_test_cases.append(case)
    else:
        print(f"   ⚠ Bỏ qua test case '{case['name']}': thiếu modalities {missing}")

print(f"\n   - Số test cases hợp lệ: {len(valid_test_cases)}/{len(test_cases)}")

# Chạy so sánh cho từng test case
print("\n[6/7] Chạy so sánh cho từng test case...")
print("=" * 80)

results_summary = []

for idx, case in enumerate(valid_test_cases):
    print(f"\n[{idx+1}/{len(valid_test_cases)}] Test case: {case['name']}")
    print(f"   Modalities: {', '.join(case['modalities'])}")
    
    try:
        # Chuẩn bị dữ liệu
        data, mask, labels = get_training_data(case['modalities'], modality_dict, modality_MASK, df_outcomes)
        
        if len(data) == 0 or len(labels) == 0:
            print(f"   ⚠ Bỏ qua: không có dữ liệu")
            continue
        
        print(f"   - Samples: {len(labels)}")
        
        # Chạy mô hình gốc
        print(f"   - Đang chạy mô hình gốc...")
        summary_df_original, _ = train(
            modality_list_in=data,
            modality_mask=mask,
            outcomes=labels,
            l1_dfs_filter=case['filters'],
            model_params=case['params'],
            folds=folds
        )
        
        # Tính AUC cho mô hình gốc
        if 1.0 in summary_df_original['label'].values and 0.0 in summary_df_original['label'].values:
            auc_original, ci_original = auc_roc_ci(
                summary_df_original['label'].values, 
                summary_df_original['score'].values, 
                0.95
            )
        else:
            print(f"   ⚠ Không thể tính AUC cho mô hình gốc (thiếu một trong hai lớp)")
            continue
        
        # Chạy mô hình OvO
        print(f"   - Đang chạy mô hình OvO...")
        summary_df_ovo, _ = train_ovo(
            modality_list_in=data,
            modality_mask=mask,
            outcomes=labels,
            l1_dfs_filter=case['filters'],
            model_params=case['params'],
            folds=folds
        )
        
        # Tính AUC cho mô hình OvO
        if 1.0 in summary_df_ovo['label'].values and 0.0 in summary_df_ovo['label'].values:
            auc_ovo, ci_ovo = auc_roc_ci(
                summary_df_ovo['label'].values, 
                summary_df_ovo['score'].values, 
                0.95
            )
        else:
            print(f"   ⚠ Không thể tính AUC cho mô hình OvO (thiếu một trong hai lớp)")
            continue
        
        # Lưu kết quả
        improvement = auc_ovo - auc_original
        improvement_pct = (improvement / auc_original * 100) if auc_original > 0 else 0
        
        results_summary.append({
            'Test Case': case['name'],
            'Modalities': ', '.join(case['modalities']),
            'N_Samples': len(labels),
            'Original_AUC': auc_original,
            'Original_CI_Lower': ci_original[0],
            'Original_CI_Upper': ci_original[1],
            'OvO_AUC': auc_ovo,
            'OvO_CI_Lower': ci_ovo[0],
            'OvO_CI_Upper': ci_ovo[1],
            'AUC_Difference': improvement,
            'AUC_Improvement_%': improvement_pct,
            'Better_Model': 'OvO' if improvement > 0 else 'Original' if improvement < 0 else 'Equal'
        })
        
        print(f"   ✓ Original AUC: {auc_original:.4f} (95% CI: {ci_original[0]:.4f} - {ci_original[1]:.4f})")
        print(f"   ✓ OvO AUC: {auc_ovo:.4f} (95% CI: {ci_ovo[0]:.4f} - {ci_ovo[1]:.4f})")
        print(f"   ✓ Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
    except Exception as e:
        print(f"   ✗ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Tổng hợp kết quả
print("\n[7/7] Tổng hợp kết quả...")
print("=" * 80)

if len(results_summary) == 0:
    print("⚠ Không có kết quả nào để tổng hợp!")
    sys.exit(1)

df_results = pd.DataFrame(results_summary)

# Hiển thị bảng so sánh chi tiết cho từng loại data
print("\n" + "=" * 100)
print("BẢNG SO SÁNH CHI TIẾT CHO TỪNG LOẠI DATA")
print("=" * 100)

# Tạo bảng so sánh dễ đọc hơn
comparison_table = []
for r in results_summary:
    comparison_table.append({
        'Test Case': r['Test Case'],
        'N_Samples': r['N_Samples'],
        'Original_AUC': f"{r['Original_AUC']:.4f}",
        'Original_CI': f"[{r['Original_CI_Lower']:.4f}, {r['Original_CI_Upper']:.4f}]",
        'OvO_AUC': f"{r['OvO_AUC']:.4f}",
        'OvO_CI': f"[{r['OvO_CI_Lower']:.4f}, {r['OvO_CI_Upper']:.4f}]",
        'Difference': f"{r['AUC_Difference']:+.4f}",
        'Improvement_%': f"{r['AUC_Improvement_%']:+.2f}%",
        'Winner': r['Better_Model']
    })

df_comparison = pd.DataFrame(comparison_table)
print("\n" + df_comparison.to_string(index=False))

# Hiển thị từng test case một cách chi tiết
print("\n" + "=" * 100)
print("CHI TIẾT TỪNG TEST CASE")
print("=" * 100)

for idx, r in enumerate(results_summary, 1):
    print(f"\n{idx}. {r['Test Case']}")
    print(f"   Modalities: {r['Modalities']}")
    print(f"   Số samples: {r['N_Samples']}")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │ Mô hình gốc:  AUC = {r['Original_AUC']:.4f} (95% CI: {r['Original_CI_Lower']:.4f} - {r['Original_CI_Upper']:.4f}) │")
    print(f"   │ Mô hình OvO:  AUC = {r['OvO_AUC']:.4f} (95% CI: {r['OvO_CI_Lower']:.4f} - {r['OvO_CI_Upper']:.4f}) │")
    print(f"   │ Chênh lệch:   {r['AUC_Difference']:+.4f} ({r['AUC_Improvement_%']:+.2f}%)                                    │")
    if r['Better_Model'] == 'OvO':
        print(f"   │ ⭐ Mô hình OvO tốt hơn {abs(r['AUC_Improvement_%']):.2f}%                                    │")
    elif r['Better_Model'] == 'Original':
        print(f"   │ ⭐ Mô hình gốc tốt hơn {abs(r['AUC_Improvement_%']):.2f}%                                    │")
    else:
        print(f"   │ ⭐ Hai mô hình bằng nhau                                                      │")
    print(f"   └─────────────────────────────────────────────────────────┘")

# Thống kê tổng hợp
ovo_wins = sum(1 for r in results_summary if r['Better_Model'] == 'OvO')
original_wins = sum(1 for r in results_summary if r['Better_Model'] == 'Original')
equal = sum(1 for r in results_summary if r['Better_Model'] == 'Equal')

avg_improvement = df_results['AUC_Difference'].mean()
avg_improvement_pct = df_results['AUC_Improvement_%'].mean()

# Tính thống kê chi tiết hơn
ovo_better_cases = [r for r in results_summary if r['Better_Model'] == 'OvO']
original_better_cases = [r for r in results_summary if r['Better_Model'] == 'Original']

print(f"\n{'='*100}")
print("THỐNG KÊ TỔNG HỢP")
print(f"{'='*100}")
print(f"Tổng số test cases: {len(results_summary)}")
print(f"  - OvO tốt hơn: {ovo_wins} ({100*ovo_wins/len(results_summary):.1f}%)")
print(f"  - Original tốt hơn: {original_wins} ({100*original_wins/len(results_summary):.1f}%)")
print(f"  - Bằng nhau: {equal} ({100*equal/len(results_summary):.1f}%)")
print(f"\nCải thiện AUC trung bình: {avg_improvement:+.4f} ({avg_improvement_pct:+.2f}%)")

if ovo_better_cases:
    avg_ovo_improvement = np.mean([r['AUC_Difference'] for r in ovo_better_cases])
    print(f"\nKhi OvO tốt hơn:")
    print(f"  - Số trường hợp: {len(ovo_better_cases)}")
    print(f"  - Cải thiện trung bình: {avg_ovo_improvement:+.4f}")
    print(f"  - Cải thiện lớn nhất: {max([r['AUC_Difference'] for r in ovo_better_cases]):+.4f} ({max(ovo_better_cases, key=lambda x: x['AUC_Difference'])['Test Case']})")
    print(f"  - Cải thiện nhỏ nhất: {min([r['AUC_Difference'] for r in ovo_better_cases]):+.4f} ({min(ovo_better_cases, key=lambda x: x['AUC_Difference'])['Test Case']})")

if original_better_cases:
    avg_original_improvement = np.mean([r['AUC_Difference'] for r in original_better_cases])
    print(f"\nKhi Original tốt hơn:")
    print(f"  - Số trường hợp: {len(original_better_cases)}")
    print(f"  - Chênh lệch trung bình: {avg_original_improvement:+.4f}")
    print(f"  - Chênh lệch lớn nhất: {min([r['AUC_Difference'] for r in original_better_cases]):+.4f} ({min(original_better_cases, key=lambda x: x['AUC_Difference'])['Test Case']})")
    print(f"  - Chênh lệch nhỏ nhất: {max([r['AUC_Difference'] for r in original_better_cases]):+.4f} ({max(original_better_cases, key=lambda x: x['AUC_Difference'])['Test Case']})")

# Top 5 test cases có cải thiện tốt nhất với OvO
print(f"\n{'='*100}")
print("TOP 5 TEST CASES - OVO TỐT NHẤT")
print(f"{'='*100}")
top_ovo = sorted([r for r in results_summary if r['Better_Model'] == 'OvO'], 
                 key=lambda x: x['AUC_Difference'], reverse=True)[:5]
for idx, r in enumerate(top_ovo, 1):
    print(f"{idx}. {r['Test Case']}: +{r['AUC_Difference']:.4f} ({r['AUC_Improvement_%']:+.2f}%)")

# Top 5 test cases - Original tốt nhất
print(f"\n{'='*100}")
print("TOP 5 TEST CASES - ORIGINAL TỐT NHẤT")
print(f"{'='*100}")
top_original = sorted([r for r in results_summary if r['Better_Model'] == 'Original'], 
                      key=lambda x: x['AUC_Difference'])[:5]
for idx, r in enumerate(top_original, 1):
    print(f"{idx}. {r['Test Case']}: {r['AUC_Difference']:.4f} ({r['AUC_Improvement_%']:+.2f}%)")

# Lưu kết quả
print("\n[8/8] Đang lưu kết quả...")
output_dir = Path('./excel')
output_dir.mkdir(exist_ok=True)

with pd.ExcelWriter(output_dir / 'ovo_comparison_full_results.xlsx', engine='xlsxwriter') as writer:
    # Sheet 1: Tất cả kết quả
    df_results.to_excel(writer, sheet_name='All_Results', index=False)
    
    # Sheet 2: Sắp xếp theo improvement
    df_results_sorted = df_results.sort_values('AUC_Difference', ascending=False)
    df_results_sorted.to_excel(writer, sheet_name='Sorted_by_Improvement', index=False)
    
    # Sheet 3: Chỉ các test cases OvO tốt hơn
    df_ovo_better = df_results[df_results['Better_Model'] == 'OvO'].sort_values('AUC_Difference', ascending=False)
    if len(df_ovo_better) > 0:
        df_ovo_better.to_excel(writer, sheet_name='OvO_Better', index=False)
    
    # Sheet 4: Chỉ các test cases Original tốt hơn
    df_original_better = df_results[df_results['Better_Model'] == 'Original'].sort_values('AUC_Difference', ascending=True)
    if len(df_original_better) > 0:
        df_original_better.to_excel(writer, sheet_name='Original_Better', index=False)
    
    # Sheet 5: Bảng so sánh dễ đọc
    df_comparison.to_excel(writer, sheet_name='Comparison_Table', index=False)
    
    # Sheet 6: Thống kê tổng hợp
    stats_data = {
        'Metric': [
            'Total Test Cases',
            'OvO Better (Count)',
            'OvO Better (%)',
            'Original Better (Count)',
            'Original Better (%)',
            'Equal (Count)',
            'Equal (%)',
            'Average Improvement',
            'Average Improvement (%)',
            'Max Improvement (OvO)',
            'Max Improvement Case',
            'Max Difference (Original)',
            'Max Difference Case'
        ],
        'Value': [
            len(results_summary),
            ovo_wins,
            f"{100*ovo_wins/len(results_summary):.1f}%",
            original_wins,
            f"{100*original_wins/len(results_summary):.1f}%",
            equal,
            f"{100*equal/len(results_summary):.1f}%",
            f"{avg_improvement:.4f}",
            f"{avg_improvement_pct:.2f}%",
            f"{max([r['AUC_Difference'] for r in results_summary]):.4f}" if results_summary else "N/A",
            max(results_summary, key=lambda x: x['AUC_Difference'])['Test Case'] if results_summary else "N/A",
            f"{min([r['AUC_Difference'] for r in results_summary]):.4f}" if results_summary else "N/A",
            min(results_summary, key=lambda x: x['AUC_Difference'])['Test Case'] if results_summary else "N/A"
        ]
    }
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_excel(writer, sheet_name='Statistics', index=False)

print(f"✓ Đã lưu kết quả vào: {output_dir / 'ovo_comparison_full_results.xlsx'}")

print("\n" + "=" * 80)
print("HOÀN TẤT!")
print("=" * 80)

