"""Generate Supplementary Figure S5 — modality availability heatmap for the
discovery cohort (n=247 patients x 8 modalities/sub-modalities), built from
the real modality_MASK produced by lung_helpers data-prep functions, exactly
as in compare_ovo_attention_full.py.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/
CODE_DIR = os.path.dirname(PAPER_DIR)  # code/
SUPP_FIG_DIR = os.path.join(PAPER_DIR, "supplementary", "figures")

sys.path.insert(0, CODE_DIR)
from lung_helpers import (
    get_clinical_table_v2, prepare_rad_modality_by_size,
    prepare_other_modalities, decorate_with_site_index,
)

BASE_DB_DIR = os.path.join(CODE_DIR, "..", "datasets")

df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
df_cohort_disc = df_cohort[df_cohort["cohort"] == "discovery"]

df_clinical = get_clinical_table_v2(
    path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
    main_index_col="dmp_pt_id",
    cohort=df_cohort_disc,
)

clinical_predictors = ['age', 'pack_years', 'ecog', 'albumin', 'dnlr',
                        'brain_mets', 'liver_mets', 'tumor_burden', 'therapy_line',
                        'recieves_combo_therapy', 'site_lung', 'recieves_pdl1_therapy', 'hist_adeno']

df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
df_pdl1 = pd.read_parquet(f"{BASE_DB_DIR}/pdl1_score.parquet")
df_labs = df_clinical[clinical_predictors]

df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")
df_radiology_by_site = decorate_with_site_index(df_radiology)

df_texture = pd.read_parquet(f"{BASE_DB_DIR}/new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet")
df_glcm = pd.read_parquet(f"{BASE_DB_DIR}/lung_pathology_pdl1_glcm_v3.parquet")

modality_MASK = df_clinical[[]]
modality_dict = {}

prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'PC', 'rad_lesion_pc')
prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'PL', 'rad_lesion_pl')
prepare_rad_modality_by_size(modality_dict, df_radiology_by_site, modality_MASK, 'LN', 'rad_lesion_ln')

prepare_other_modalities(modality_dict, df_texture, modality_MASK, 'path_ihc_pdl1')
prepare_other_modalities(modality_dict, df_glcm, modality_MASK, 'path_ihc_glcm')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, 'gen_driver_mut_amp')
prepare_other_modalities(modality_dict, df_pdl1, modality_MASK, 'cnl_pdl1_score')
prepare_other_modalities(modality_dict, df_labs, modality_MASK, 'cnl_dem_labs')

modality_MASK = modality_MASK.fillna(False)

col_labels = {
    'rad_lesion_pc': 'CT radiomics (PC)',
    'rad_lesion_pl': 'CT radiomics (PL)',
    'rad_lesion_ln': 'CT radiomics (LN)',
    'path_ihc_pdl1': 'Pathology IHC-A',
    'path_ihc_glcm': 'Pathology IHC-G',
    'gen_driver_mut_amp': 'Genomics',
    'cnl_pdl1_score': 'PD-L1 TPS',
    'cnl_dem_labs': 'Clinical labs',
}
cols = list(col_labels.keys())
mat = modality_MASK[cols].astype(int).values

# Sort patients for a more readable banding pattern (by total availability, then by columns)
order = np.lexsort([-mat[:, i] for i in reversed(range(mat.shape[1]))] + [-mat.sum(axis=1)])
mat = mat[order]

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([col_labels[c] for c in cols], rotation=45, ha="right", fontsize=8)
ax.set_ylabel(f"Patients (n = {mat.shape[0]})")
ax.set_yticks([])
ax.set_title("Modality availability — discovery cohort", fontsize=10, fontweight="bold")

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="black", label="Available"),
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="Missing"),
]
ax.legend(handles=handles, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(SUPP_FIG_DIR, "figS5_modality_heatmap.pdf"))
plt.close(fig)

print("Figure S5 done. Mask shape:", modality_MASK[cols].shape)
print(modality_MASK[cols].sum().to_dict())
