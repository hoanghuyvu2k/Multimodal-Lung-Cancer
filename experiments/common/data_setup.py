"""
Harness dùng chung: load toàn bộ dữ liệu một lần, cache lại để các experiment sau
không phải đọc lại parquet radiomics 54MB.

Rút ra từ phần [1/7]-[4/7] của compare_ovo_attention_full.py.

Dùng:
    from common.data_setup import load_all
    ctx = load_all()
    ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes, ctx.rad_filters
"""

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Cho phép import lung_helpers từ code/ khi chạy từ experiments/
_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import (  # noqa: E402
    decorate_with_site_index,
    get_clinical_table_v2,
    prepare_other_modalities,
    prepare_rad_modality_by_size,
)

BASE_DB_DIR = str(_CODE_DIR / ".." / "datasets")

_CACHE_DIR = Path(
    os.environ.get(
        "ATTN_CACHE_DIR",
        r"C:\Users\Admin\AppData\Local\Temp\claude\D--code-master-doan-code"
        r"\30911c1f-27b8-40d7-9ab5-213f143105a1\scratchpad",
    )
)
_CACHE_PATH = _CACHE_DIR / "attn_redesign_data_cache.pkl"

CLINICAL_PREDICTORS = [
    "age", "pack_years", "ecog", "albumin", "dnlr",
    "brain_mets", "liver_mets", "tumor_burden", "therapy_line",
    "recieves_combo_therapy", "site_lung", "recieves_pdl1_therapy", "hist_adeno",
]


@dataclass
class DataContext:
    modality_dict: dict
    modality_MASK: pd.DataFrame
    df_outcomes: pd.DataFrame
    rad_filters: dict
    rad_filters_lu: dict


def _read_parquet_any(*names):
    """Parquet filenames có thể khác nhau về hoa/thường."""
    for name in names:
        path = os.path.join(BASE_DB_DIR, name)
        if os.path.exists(path):
            return pd.read_parquet(path)
    return pd.DataFrame()


def _build():
    df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    df_cohort_disc = df_cohort[df_cohort["cohort"] == "discovery"]

    df_clinical = get_clinical_table_v2(
        path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540"
             f"-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id",
        cohort=df_cohort_disc,
    )
    df_outcomes = df_clinical[["label"]].copy(deep=True)

    df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
    df_tmb = df_genomic[["TMB"]]
    df_nontmb = df_genomic.loc[:, ~df_genomic.columns.str.contains("TMB")]
    df_labs = df_clinical[CLINICAL_PREDICTORS]

    df_pdl1 = _read_parquet_any("pdl1_score.parquet", "PDL1_SCORE.parquet")
    df_radiology = _read_parquet_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet",
        "LUNG_RADIOMICS_spacing1.0_MirpOn_Window1350.250_allImageTypes_bw20.parquet",
    )
    df_texture = _read_parquet_any(
        "new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet",
        "NEW_LUNG_glcm_Autocorrelation_v2_20x_stain1_PDL1.parquet",
    )
    df_glcm = _read_parquet_any(
        "lung_pathology_pdl1_glcm_v3.parquet",
        "LUNG_PATHOLOGY_PDL1_GLCM_V3.parquet",
    )

    df_radiology_by_site = decorate_with_site_index(df_radiology)

    modality_MASK = df_clinical[[]]
    modality_dict = {}

    modality_PC_full = prepare_rad_modality_by_size(
        modality_dict, df_radiology_by_site, modality_MASK, "PC", "rad_lesion_pc")
    modality_PL_full = prepare_rad_modality_by_size(
        modality_dict, df_radiology_by_site, modality_MASK, "PL", "rad_lesion_pl")
    modality_LN_full = prepare_rad_modality_by_size(
        modality_dict, df_radiology_by_site, modality_MASK, "LN", "rad_lesion_ln")
    modality_LU_full = prepare_rad_modality_by_size(
        modality_dict, df_radiology_by_site, modality_MASK,
        sites=["PC", "PL", "LN"], name="rad_lesion_lu",
        sort="original_shape_MeshVolume", ascending=False, reduce=True,
    )

    if not df_texture.empty:
        prepare_other_modalities(modality_dict, df_texture, modality_MASK, "path_ihc_pdl1")
    if not df_glcm.empty:
        prepare_other_modalities(modality_dict, df_glcm, modality_MASK, "path_ihc_glcm")
    prepare_other_modalities(modality_dict, df_genomic, modality_MASK, "gen_driver_mut_amp")
    prepare_other_modalities(modality_dict, df_nontmb, modality_MASK, "gen_driver_non_tmb")
    prepare_other_modalities(modality_dict, df_tmb, modality_MASK, "gen_driver_tmb")
    if not df_pdl1.empty:
        prepare_other_modalities(modality_dict, df_pdl1, modality_MASK, "cnl_pdl1_score")
    prepare_other_modalities(modality_dict, df_labs, modality_MASK, "cnl_dem_labs")

    if "cnl_pdl1_score" in modality_dict:
        modality_dict["cnl_pdl1_score"] = -modality_dict["cnl_pdl1_score"] / 100.0
    modality_MASK = modality_MASK.fillna(False)

    rad_filters = {
        0: {"l1_selection_df": modality_PC_full, "kwargs": {"l1_strength": 0.1}},
        1: {"l1_selection_df": modality_PL_full, "kwargs": {"l1_strength": 0.1}},
        2: {"l1_selection_df": modality_LN_full, "kwargs": {"l1_strength": 0.1}},
    }
    rad_filters_lu = dict(rad_filters)
    rad_filters_lu[3] = {"l1_selection_df": modality_LU_full, "kwargs": {"l1_strength": 0.1}}

    return DataContext(modality_dict, modality_MASK, df_outcomes, rad_filters, rad_filters_lu)


def load_all(use_cache=True):
    """Trả về DataContext. Cache ra pickle trong scratchpad để lần sau load nhanh."""
    # Cache dạng dict, không phải dataclass: nếu pickle chính object DataContext thì
    # class bị gắn vào module tạo ra nó (__main__ khi chạy trực tiếp) và không load lại được.
    if use_cache and _CACHE_PATH.exists():
        with open(_CACHE_PATH, "rb") as fh:
            return DataContext(**pickle.load(fh))

    ctx = _build()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as fh:
        pickle.dump(ctx.__dict__, fh)
    return ctx


if __name__ == "__main__":
    ctx = load_all(use_cache=False)
    print(f"modalities        : {len(ctx.modality_dict)}")
    print(f"mask shape        : {ctx.modality_MASK.shape}")
    print(f"outcomes          : {len(ctx.df_outcomes)}")
    print(f"label distribution: {ctx.df_outcomes['label'].value_counts().to_dict()}")
    print(f"modality names    : {sorted(ctx.modality_dict.keys())}")
